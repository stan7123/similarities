from typing import Annotated
from uuid import UUID, uuid4

import redis
from decouple import config
from fastapi import APIRouter, Depends, HTTPException, UploadFile, responses, status
from rq import Queue, Retry
from sqlmodel import Session, select

from similarities.db import get_session
from similarities.models import Image, validate_image_content
from similarities.serializers import (
    ImageCreationResponse, SearchType, SimilarImageEntry, SimilarImagesResponse, SimilarResponseStatus,
    SEARCH_TYPE_TO_COLUMN_NAME
)
from similarities.processing import update_image_histograms
from similarities.storage import save_uploaded_file, get_image_public_url


SessionDep = Annotated[Session, Depends(get_session)]

redis_conn = redis.from_url(config("QUEUE_BROKER_URL"))
queue = Queue("default", connection=redis_conn)

router = APIRouter()


@router.post("/upload", status_code=status.HTTP_201_CREATED, response_model=ImageCreationResponse)
async def upload_image(image: UploadFile, session: SessionDep):
    await validate_image_content(image)

    unique_id = str(uuid4())
    image_path = await save_uploaded_file(unique_id, image)

    image_obj = Image(id=unique_id, path=str(image_path))
    session.add(image_obj)
    session.commit()

    retry_backoff = Retry(10, interval=[5 * 2**n for n in range(10)])  # Up to 2560 seconds between last retries
    queue.enqueue(update_image_histograms, str(image_obj.id), retry=retry_backoff)

    return image_obj


@router.get("/download/{image_id}")
async def download_image(image_id: UUID, session: SessionDep):
    image_obj = session.get(Image, image_id)
    if not image_obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found.")

    public_url = get_image_public_url(image_obj)
    return responses.RedirectResponse(str(public_url), status_code=status.HTTP_301_MOVED_PERMANENTLY)


@router.get("/similar/{image_id}/{search_type}", response_model=SimilarImagesResponse)
async def similar_images(
        image_id: UUID,
        search_type: SearchType,
        session: SessionDep,
        limit: int = 10,
        max_distance: float = None,
):
    image_obj = session.get(Image, image_id)
    if not image_obj:
        raise HTTPException(status_code=404, detail="Image not found.")

    column_name = SEARCH_TYPE_TO_COLUMN_NAME[search_type]
    image_histogram = getattr(image_obj, column_name)
    if image_histogram is None:
        return SimilarImagesResponse(
            status=SimilarResponseStatus.PROCESSING,
            image_url=get_image_public_url(image_obj),
            similar_images=[],
        )

    image_column = getattr(Image, column_name)
    query = (
        select(Image, image_column.l2_distance(image_histogram).label("distance"))
        .where(Image.id != image_obj.id)
        # Adding '0' to not use index because of the issue: https://github.com/pgvector/pgvector/issues/719
        .order_by(image_column.l2_distance(image_histogram) + 0)
        .limit(limit)
    )
    if max_distance:
        query = query.where(image_column.l2_distance(image_histogram) <= max_distance)

    query_result = session.exec(query)
    similar_images = [
        SimilarImageEntry(url=get_image_public_url(image), distance=distance)
        for image, distance in query_result
    ]

    return SimilarImagesResponse(
        status=SimilarResponseStatus.OK,
        image_url=get_image_public_url(image_obj),
        similar_images=similar_images
    )

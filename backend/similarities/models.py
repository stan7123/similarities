from datetime import datetime, UTC
from uuid import UUID, uuid4

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status
from pgvector.sqlalchemy import Vector
from sqlalchemy import Column
from sqlmodel import Field, SQLModel, Index

from similarities.histograms import (
    COLOR_HISTOGRAM_VECTOR_SIZE,
    HOG_HISTOGRAM_VECTOR_SIZE,
    TEXTURE_HISTOGRAM_VECTOR_SIZE,
)


ACCEPTED_CONTENT_TYPES = {
    "image/png",
    "image/jpeg",
    "image/jpg",
}


class Image(SQLModel, table=True):
    id: UUID = Field(default=uuid4, primary_key=True)
    path: str
    color_hist: list[float] = Field(sa_column=Column(Vector(COLOR_HISTOGRAM_VECTOR_SIZE)))
    hog_hist: list[float] = Field(sa_column=Column(Vector(HOG_HISTOGRAM_VECTOR_SIZE)))
    texture_hist: list[float] = Field(sa_column=Column(Vector(TEXTURE_HISTOGRAM_VECTOR_SIZE)))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    processed_at: datetime | None

    __table_args__ = (
        Index('ix_image_color', 'color_hist', postgresql_using='ivfflat'),
        Index('ix_image_hog', 'hog_hist', postgresql_using='ivfflat'),
        Index('ix_image_texture', 'texture_hist', postgresql_using='ivfflat'),
    )


async def validate_image_content(image: UploadFile):
    contents = await image.read()
    await image.seek(0)
    file_bytes = np.frombuffer(contents, dtype=np.uint8)
    im = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if im is None or image.content_type not in ACCEPTED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported content type or corrupted image. Supported content types: {ACCEPTED_CONTENT_TYPES}",
        )

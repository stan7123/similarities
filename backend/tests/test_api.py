from tempfile import NamedTemporaryFile
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import func
from sqlmodel import Session, select

from app import app
from similarities.models import Image
from similarities.processing import update_image_histograms
from similarities.serializers import SearchType
from tests import assets


client = TestClient(app)


def get_temp_image(extension: str = 'jpg') -> NamedTemporaryFile:
    image = np.zeros((100, 100, 3), np.uint8)
    tmp_file = NamedTemporaryFile(suffix=f'.{extension}')
    cv2.imwrite(tmp_file.name, image)
    return tmp_file


def get_temp_corrupted_image() -> NamedTemporaryFile:
    tmp_file = NamedTemporaryFile(suffix='.jpg')
    tmp_file.write(b'qweqweqwe')  # Writing text instead of image data
    return tmp_file


@patch("similarities.api.queue.enqueue")
def test_successful_image_upload(mocked_queue, session: Session, client: TestClient):
    assert session.scalar(select(func.count(Image.id))) == 0

    response = client.post("/upload", files={"image": get_temp_image()})

    assert response.status_code == 201
    response_json = response.json()
    assert "id" in response_json
    assert session.scalar(select(func.count(Image.id))) == 1
    image_obj = session.get(Image, response_json["id"])
    assert image_obj is not None
    assert image_obj.path is not None
    assert image_obj.created_at is not None
    assert image_obj.color_hist is None
    assert image_obj.hog_hist is None
    assert image_obj.texture_hist is None
    assert image_obj.processed_at is None
    assert mocked_queue.assert_called_once


@patch("similarities.api.queue.enqueue")
def test_if_returns_error_when_no_image_send(mocked_queue, session: Session, client: TestClient):
    assert session.scalar(select(func.count(Image.id))) == 0

    response = client.post("/upload")

    assert response.status_code == 422
    response_json = response.json()
    assert response_json["detail"][0]["type"] == "missing"
    assert response_json["detail"][0]["loc"] == ["body", "image"]
    assert response_json["detail"][0]["msg"] == "Field required"
    assert session.scalar(select(func.count(Image.id))) == 0
    assert mocked_queue.assert_not_called


@patch("similarities.api.queue.enqueue")
def test_returning_unsupported_media_type_when_uploaded_file_has_no_image_content(
        mocked_queue, session: Session, client: TestClient
):
    assert session.scalar(select(func.count(Image.id))) == 0

    response = client.post("/upload", files={"image": get_temp_corrupted_image()})

    assert response.status_code == 415
    response_json = response.json()
    assert "Unsupported content type or corrupted image" in response_json["detail"]
    assert session.scalar(select(func.count(Image.id))) == 0
    assert mocked_queue.assert_not_called


def test_returning_redirection_for_existing_image(session: Session, client: TestClient):
    unique_id = "a17b8434-a467-46d3-8f36-0c5863781f75"
    image = Image(id=unique_id, path="/storage/ab/cd/a17b8434-a467-46d3-8f36-0c5863781f75.jpg")
    session.add(image)
    session.commit()

    response = client.get(f"/download/{image.id}", follow_redirects=False)

    assert response.status_code == 301
    assert "location" in response.headers
    assert response.headers["location"].endswith(image.path)


def test_returning_redirection_for_nonexistent_image(session: Session, client: TestClient):
    nonexistent_id = "a17b8434-a467-46d3-8f36-0c5863781f75"
    assert session.scalar(select(func.count(Image.id)).where(Image.id == nonexistent_id)) == 0

    response = client.get(f"/download/{nonexistent_id}", follow_redirects=False)

    assert response.status_code == 404
    response_json = response.json()
    assert response_json["detail"] == "Image not found."


@patch("similarities.processing.get_session_instance")
def test_returning_similar_images(mock_get_session, session: Session, client: TestClient):
    mock_get_session.return_value = session

    images_to_load = _load_images(session)

    requested_kiwi = images_to_load[-1]
    response = client.get(f"/similar/{requested_kiwi['id']}/{SearchType.COLORS.value}")

    assert response.status_code == 200
    response_json = response.json()
    assert response_json["status"] == "ok"
    assert response_json["image_url"].endswith(requested_kiwi["path"])
    assert len(response_json["similar_images"]) == 8

    response = client.get(f"/similar/{requested_kiwi['id']}/{SearchType.OBJECTS.value}")

    assert response.status_code == 200
    response_json = response.json()
    assert response_json["status"] == "ok"
    assert response_json["image_url"].endswith(requested_kiwi["path"])
    assert len(response_json["similar_images"]) == 8

    response = client.get(f"/similar/{requested_kiwi['id']}/{SearchType.TEXTURE.value}")

    assert response.status_code == 200
    response_json = response.json()
    assert response_json["status"] == "ok"
    assert response_json["image_url"].endswith(requested_kiwi["path"])
    assert len(response_json["similar_images"]) == 8


@patch("similarities.processing.get_session_instance")
def test_limiting_similar_images_to_requested_max_distance(mock_get_session, session: Session, client: TestClient):
    mock_get_session.return_value = session

    images_to_load = _load_images(session)

    requested_kiwi = images_to_load[-1]
    no_max_distance_response = client.get(f"/similar/{requested_kiwi['id']}/{SearchType.COLORS.value}")

    assert no_max_distance_response.status_code == 200
    response_json = no_max_distance_response.json()
    assert len(response_json["similar_images"]) == 8
    max_distance = 1.2
    images_under_max_distance = [
        i for i in response_json["similar_images"]
        if i["distance"] <= max_distance
    ]
    assert len(images_under_max_distance) < 8, "Adjust max_distance value so that it filters out the results."

    max_distance_response = client.get(
        f"/similar/{requested_kiwi['id']}/{SearchType.COLORS.value}",
        params={"max_distance": max_distance}
    )

    assert max_distance_response.status_code == 200
    response_json = max_distance_response.json()
    assert len(response_json["similar_images"]) == len(images_under_max_distance)


def test_returning_processing_status_when_requested_histogram_not_ready(session: Session, client: TestClient):
    image_id = "77777777-4444-4444-1111-222263781f75"
    image = Image(id=image_id, path=str(assets.IMAGES["apples"][0]))
    session.add(image)
    session.commit()

    response = client.get(f"/similar/{image_id}/{SearchType.COLORS.value}")

    assert response.status_code == 200
    response_json = response.json()
    assert response_json["status"] == "processing"
    assert response_json["image_url"].endswith(image.path)
    assert response_json["similar_images"] == []


def test_similar_image_when_nonexistent_id_passed(session: Session, client: TestClient):
    nonexistent_id = "77777777-4444-4444-1111-0c5863781f75"

    response = client.get(f"/similar/{nonexistent_id}/{SearchType.COLORS.value}")

    assert response.status_code == 404


def _load_images(session: Session) -> list:
    images_to_load = [
        {
            "id": "00000000-a467-46d3-8f36-0c5863781f75",
            "path": str(assets.IMAGES["apples"][0]),
        },
        {
            "id": "11111111-a467-46d3-8f36-0c5863781f75",
            "path": str(assets.IMAGES["apples"][1]),
        },
        {
            "id": "22222222-a467-46d3-8f36-0c5863781f75",
            "path": str(assets.IMAGES["apples"][2]),
        },
        {
            "id": "33333333-a467-46d3-8f36-0c5863781f75",
            "path": str(assets.IMAGES["bananas"][0]),
        },
        {
            "id": "44444444-a467-46d3-8f36-0c5863781f75",
            "path": str(assets.IMAGES["bananas"][1]),
        },
        {
            "id": "55555555-a467-46d3-8f36-0c5863781f75",
            "path": str(assets.IMAGES["bananas"][2]),
        },
        {
            "id": "66666666-a467-46d3-8f36-0c5863781f75",
            "path": str(assets.IMAGES["kiwi"][0]),
        },
        {
            "id": "77777777-a467-46d3-8f36-0c5863781f75",
            "path": str(assets.IMAGES["kiwi"][1]),
        },
        {
            "id": "88888888-a467-46d3-8f36-0c5863781f75",
            "path": str(assets.IMAGES["kiwi"][2]),
        },
    ]
    for entry in images_to_load:
        image = Image(id=entry["id"], path=entry["path"])
        session.add(image)

    session.commit()

    for entry in images_to_load:
        update_image_histograms(entry["id"])

    return images_to_load

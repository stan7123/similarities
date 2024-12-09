from unittest.mock import patch

import cv2
from sqlmodel import Session

from similarities.histograms import(
    calculate_color_histogram,
    calculate_hog_histogram,
    calculate_texture_histogram,
    COLOR_HISTOGRAM_VECTOR_SIZE,
    HOG_HISTOGRAM_VECTOR_SIZE,
    TEXTURE_HISTOGRAM_VECTOR_SIZE,
)
from similarities.models import Image
from similarities.processing import update_image_histograms
from tests import assets


@patch("similarities.processing.get_session_instance")
def test_if_histograms_calculation_updates_fields_in_db(mock_get_session, session: Session):
    mock_get_session.return_value = session

    image_id = "cee6e8b5-6c21-47f8-8dc9-ea4bfcf07bfc"
    image_path = assets.IMAGES["apples"][0]
    image = Image(id=image_id, path=str(image_path))
    session.add(image)
    session.commit()

    image_obj = session.get(Image, image_id)
    assert image_obj.color_hist is None
    assert image_obj.hog_hist is None
    assert image_obj.texture_hist is None
    assert image_obj.processed_at is None

    update_image_histograms(image_id)

    image_obj = session.get(Image, image_id)
    assert image_obj.color_hist is not None
    assert image_obj.hog_hist is not None
    assert image_obj.texture_hist is not None
    assert image_obj.processed_at is not None


def test_color_histogram_returns_expected_vector_size():
    image_size_1 = cv2.imread(str(assets.IMAGES["apples"][0]))

    result = calculate_color_histogram(image_size_1)

    assert len(result) == COLOR_HISTOGRAM_VECTOR_SIZE

    image_size_2 = cv2.imread(str(assets.IMAGES["apples"][3]))

    result = calculate_color_histogram(image_size_2)

    assert len(result) == COLOR_HISTOGRAM_VECTOR_SIZE


def test_texture_histogram_returns_right_vector_size():
    image_size_1 = cv2.imread(str(assets.IMAGES["apples"][0]))

    result = calculate_texture_histogram(image_size_1)

    assert len(result) == TEXTURE_HISTOGRAM_VECTOR_SIZE

    image_size_2 = cv2.imread(str(assets.IMAGES["apples"][3]))

    result = calculate_texture_histogram(image_size_2)

    assert len(result) == TEXTURE_HISTOGRAM_VECTOR_SIZE


def test_hog_histogram_returns_right_vector_size():
    image_size_1 = cv2.imread(str(assets.IMAGES["apples"][0]))

    result = calculate_hog_histogram(image_size_1)

    assert len(result) == HOG_HISTOGRAM_VECTOR_SIZE

    image_size_2 = cv2.imread(str(assets.IMAGES["apples"][3]))

    result = calculate_hog_histogram(image_size_2)

    assert len(result) == HOG_HISTOGRAM_VECTOR_SIZE

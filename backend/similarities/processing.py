import logging
from datetime import datetime, UTC

import cv2

from similarities.db import get_session_instance
from similarities.models import Image
from similarities.histograms import (
    calculate_color_histogram, calculate_hog_histogram, calculate_texture_histogram
)


logger = logging.getLogger(__name__)


def update_image_histograms(image_id: str):
    logger.info("Processing image", image_id)

    session = get_session_instance()
    image_obj = session.get(Image, image_id)

    image = cv2.imread(image_obj.path)

    color_histogram = calculate_color_histogram(image)
    hog_histogram = calculate_hog_histogram(image)
    texture_histogram = calculate_texture_histogram(image)


    image_obj.color_hist = color_histogram
    image_obj.hog_hist = hog_histogram
    image_obj.texture_hist = texture_histogram
    image_obj.processed_at = datetime.now(UTC)

    session.add(image_obj)
    session.commit()

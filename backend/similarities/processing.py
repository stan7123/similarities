import logging
from datetime import datetime, UTC

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, hog
from sqlmodel import Session

from similarities.db import engine
from similarities.models import Image


logger = logging.getLogger(__name__)


def calculate_image_histograms(image_id: str):
    logger.info("Processing image", image_id)

    with Session(engine) as session:
        image_obj = session.get(Image, image_id)

        image = cv2.imread(image_obj.path)

        color_histogram = calculate_color_histogram(image)
        hog_histogram = calculate_hog_histogram(image)
        texture_histogram = calculate_texture_histogram(image)
        intensity_histogram = calculate_intensity_histogram(image)

        image_obj.color_hist = color_histogram
        image_obj.hog_hist = hog_histogram
        image_obj.texture_hist = texture_histogram
        image_obj.intensity_hist = intensity_histogram
        image_obj.processed_at = datetime.now(UTC)

        session.add(image_obj)
        session.commit()


"""
---------------------------------------
| Histogram     | When to use
---------------------------------------
| Color         | Comparing images based on dominant colors (e.g., landscapes, fashion, art).
| HOG           | Analyzing shapes and structures (e.g., people, objects, architecture).
| Texture       | Identifying texture patterns (e.g., fabrics, wood, medical images).
| Intensity     | Shadow detection, grayscale satellite imagery analysis.
---------------------------------------
"""

def calculate_color_histogram(image):
    """
    Color histogram.
    For similar color schema images.
    Returns Vector size: 512
    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    channels = [0, 1, 2]
    hist_size = [8, 8, 8]
    ranges = [0, 256, 0, 256, 0, 256]
    hist = cv2.calcHist([image], channels, None, hist_size, ranges)
    return cv2.normalize(hist, hist).flatten()


def calculate_intensity_histogram(image):
    """
    Intensity histogram.
    For black and white or scant variety of colors images.
    Returns Vector size: 256
    """

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    channels = [0]
    hist_size = [256]
    ranges = [0, 256]
    hist = cv2.calcHist([gray_image], channels, None, hist_size, ranges)
    return cv2.normalize(hist, hist).flatten()


def calculate_texture_histogram(image):
    """
    Haralick (Texture histogram)
    For images with similar texture schema.
    """

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    distances = [1, 2]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    levels = 256

    gray_image = np.clip(gray_image, 0, 255)
    gray_image = (gray_image / 255 * (levels - 1)).astype(np.uint8)

    glcm = graycomatrix(gray_image, distances=distances, angles=angles, levels=levels, symmetric=True, normed=True)

    feature_vector = []
    for feature_name in ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy', 'correlation']:
        feature_values = graycoprops(glcm, feature_name)
        feature_vector.extend(feature_values.ravel())

    return np.array(feature_vector)


def calculate_hog_histogram(image):
    """
    HOG (Histogram of Oriented Gradients)
    For images with similar objects
    Returns Vector size: 1764
    """

    target_size = (64, 64)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, target_size)
    return hog(resized_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), block_norm='L2-Hys')

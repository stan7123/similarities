import cv2
import numpy as np
from skimage.feature import hog


"""
---------------------------------------
| Histogram     | When to use
---------------------------------------
| Color         | Comparing images based on dominant colors (e.g., landscapes, fashion, art).
| HOG           | Analyzing shapes and structures (e.g., people, objects, architecture).
| Texture       | Identifying texture patterns (e.g., fabrics, wood, medical images).
---------------------------------------
"""

COLOR_HISTOGRAM_VECTOR_SIZE = 512
HOG_HISTOGRAM_VECTOR_SIZE = 1764
TEXTURE_HISTOGRAM_VECTOR_SIZE = 48


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


def calculate_texture_histogram(image):
    """
    Haralick (Texture histogram)
    For images with similar texture schema.
    Returns Vector size: 48
    """

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    num_orientations = 4
    num_bins = 12
    ksize = 31
    frequency = 0.2
    orientations = np.linspace(0, np.pi, num_orientations, endpoint=False)

    histograms = []
    for theta in orientations:
        kernel = cv2.getGaborKernel((ksize, ksize), 4.0, theta, frequency, 0.5, 0, ktype=cv2.CV_32F)
        filtered = cv2.filter2D(gray_image, cv2.CV_32F, kernel)
        hist, _ = np.histogram(filtered, bins=num_bins, range=(filtered.min(), filtered.max()), density=True)
        histograms.append(hist)

    full_histogram = np.concatenate(histograms)
    return full_histogram / np.sum(full_histogram)


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

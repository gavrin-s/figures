import pytest
import numpy as np
import cv2
from src import FiguresAnalyzer


@pytest.fixture()
def analyzer():
    return FiguresAnalyzer(
        eps=0.02,
        circle_color=(255, 0, 0),
        rectangle_color=(0, 255, 0),
        triangle_color=(0, 0, 255),
    )


@pytest.fixture()
def rgb_image():
    return cv2.imread('data/figures.png')[..., ::-1]


@pytest.fixture()
def gray_image():
    return cv2.imread('data/figures.png', 0)


@pytest.fixture()
def black_rgb_image():
    return np.zeros((512, 512, 3), dtype=np.uint8)


@pytest.fixture()
def black_gray_image():
    return np.zeros((512, 512), dtype=np.uint8)


@pytest.fixture()
def white_rgb_image():
    return np.ones((512, 512, 3), dtype=np.uint8) * 255


@pytest.fixture()
def white_gray_image():
    return np.ones((512, 512), dtype=np.uint8) * 255

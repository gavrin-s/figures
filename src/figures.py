from typing import Optional, Tuple
from warnings import warn
from enum import IntEnum
import cv2
import numpy as np


class Figure(IntEnum):
    """
    Class for storing figure types.
    """
    circle = 0
    rectangle = 1
    triangle = 2


class FiguresAnalyzer:
    """
    Class for figure analytics.
    """
    def __init__(
            self,
            eps: float = 0.02,
            circle_color: Tuple[int, int, int] = (255, 0, 0),
            rectangle_color: Tuple[int, int, int] = (0, 255, 0),
            triangle_color: Tuple[int, int, int] = (0, 0, 255),
    ):
        """ Initialization.

        :param eps: relative error tolerance;
        :param circle_color: fill color for circle;
        :param rectangle_color: fill color for rectangle;
        :param triangle_color: fill color triangle;
        """
        self.eps = eps
        self.figure_to_color = {
            Figure.circle: circle_color,
            Figure.rectangle: rectangle_color,
            Figure.triangle: triangle_color,
        }

    def figure_counts(self, image: np.ndarray) -> Tuple[int, int, int]:
        """ Figure counting.

        :param image: RGB or GRAY image;
        :return: count of circle, rectangle, triangle.
        """
        contours = self._get_contours(image)
        counter = {Figure.circle: 0, Figure.rectangle: 0, Figure.triangle: 0}

        for contour in contours:
            try:
                figure_type = self._get_figure_type(contour)
            except ValueError:
                continue
            counter[figure_type] += 1

        return counter[Figure.circle], counter[Figure.rectangle], counter[Figure.triangle]

    def fill_figures(
            self,
            image: np.ndarray,
            figures: Tuple[str] = ('circle', 'rectangle', 'triangle'),
    ):
        """ Figure filling.

        :param image: RGB or GRAY image;
        :param figures: tuple of figures for filling, default all.
        :return: RGB image with filled figures.
        """
        assert set(figures) - {'circle', 'rectangle', 'triangle'} == set(), \
            f"Available figures ['circle', 'rectangle'`', 'triangle']," \
            f" given {set(figures) - {'circle', 'rectangle', 'triangle'}}"

        contours = self._get_contours(image)
        result_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) if len(image.shape) == 2 else image.copy()

        for contour in contours:
            try:
                figure_type = self._get_figure_type(contour)
            except ValueError:
                continue
            if figure_type.name in figures:
                result_image = cv2.fillPoly(result_image, pts=[contour], color=self.figure_to_color[figure_type])
        return result_image

    @staticmethod
    def _get_contours(image: np.ndarray) -> Tuple[np.ndarray]:
        """
        :param image: RGB or GRAY image;
        :return: tuple of contours in cv2 format.
        """
        image_gray = image.copy() if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, image_gray = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(image_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def _get_figure_type(self, contour: np.ndarray) -> Figure:
        """
        :param contour: contour in cv2 format;
        :return: figure type;
        :raises ValueError: unable to determine figure type.
        """
        contour_area = cv2.contourArea(contour)
        if self._is_circle(contour, contour_area=contour_area):
            return Figure.circle
        if self._is_triangle(contour, contour_area=contour_area):
            return Figure.triangle
        if self._is_rectangle(contour, contour_area=contour_area):
            return Figure.rectangle
        raise ValueError(f'Unknown figure.')

    def _is_rectangle(self, contour: np.ndarray, contour_area: Optional[float]) -> bool:
        """
        :param contour: contour in cv2 format;
        :param contour_area: precalculated contour area;
        :return: True if figure is rectangle else False.
        """
        rectangle_contour = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.int0)
        rectangle_area = cv2.contourArea(rectangle_contour)
        if contour_area is None:
            contour_area = cv2.contourArea(contour)
        return (rectangle_area - contour_area) / rectangle_area <= self.eps

    def _is_circle(self, contour: np.ndarray, contour_area: Optional[float]) -> bool:
        """
        :param contour: contour in cv2 format;
        :param contour_area: precalculated contour area;
        :return: True if figure is circle else False.
        """
        _, radius = cv2.minEnclosingCircle(contour)
        circle_area = np.pi * (radius ** 2)
        if contour_area is None:
            contour_area = cv2.contourArea(contour)
        return (circle_area - contour_area) / circle_area <= self.eps

    def _is_triangle(self, contour: np.ndarray, contour_area: Optional[float]) -> bool:
        """
        :param contour: contour in cv2 format;
        :param contour_area: precalculated contour area;
        :return: True if figure is triangle else False.
        """
        _, triangle_contour = cv2.minEnclosingTriangle(contour)
        triangle_area = cv2.contourArea(triangle_contour)
        if contour_area is None:
            contour_area = cv2.contourArea(contour)
        return (triangle_area - contour_area) / triangle_area <= self.eps

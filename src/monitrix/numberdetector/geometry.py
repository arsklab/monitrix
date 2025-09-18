from typing import Literal

import cv2
from numba import jit
import numpy as np
from torch import Tensor

from ..abclass import Coordinate
from .douglas_peucker import douglas_peucker


class coordinate_cv2jit(Coordinate):
    """高速"""

    @classmethod
    def bounding_rectangle(cls, points: np.ndarray | Tensor) -> np.ndarray:
        return cls._rect_to_vertices(*cv2.boundingRect(cls.to_numpy(points)))

    @classmethod
    def polar_sort(
        cls, points: np.ndarray | Tensor, clockwise: bool = False
    ) -> np.ndarray:
        return cls._polar_sort_jit(cls.to_numpy(points), clockwise)

    @classmethod
    def change_aspect_ratio(
        cls, points: np.ndarray | Tensor, aspect_ratio: float
    ) -> np.ndarray:
        return cls._change_aspect_ratio_jit(cls.to_numpy(points), aspect_ratio)

    @classmethod
    def add_margin(
        cls, points: np.ndarray | Tensor, bottom_margin_rate: float = 0
    ) -> np.ndarray:
        _points = cls.to_numpy(points)
        _, height = np.max(_points, axis=0) - np.min(_points, axis=0)

        _, cy = _points.mean(axis=0)
        mask = _points[:, 1] > cy
        _points[:, 1][mask] = _points[:, 1][mask] + height * bottom_margin_rate
        return _points

    @classmethod
    def douglas_peucker(cls, points: np.ndarray | Tensor) -> np.ndarray:
        return douglas_peucker(cls.to_numpy(points, order="C"))

    @classmethod
    def to_numpy(
        cls,
        values: Tensor | np.ndarray,
        dtype=np.float32,
        order: Literal["C"] | None = None,
    ) -> np.ndarray:
        """numpy配列に変換"""
        array = (
            values.detach().numpy() if isinstance(values, Tensor) else values
        ).astype(dtype)
        if order == "C":
            return (
                np.ascontiguousarray(array)
                if not array.flags["C_CONTIGUOUS"]
                else array
            )
        else:
            return array

    @staticmethod
    @jit(nopython=True)
    def _rect_to_vertices(x, y, w, h) -> np.ndarray:
        return np.array(
            [[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float32
        )

    @staticmethod
    @jit(nopython=True)
    def _polar_sort_jit(points: np.ndarray, clockwise: bool = False) -> np.ndarray:
        center_x = np.sum(points[:, 0]) / points.shape[0]
        center_y = np.sum(points[:, 1]) / points.shape[0]
        angles = np.arctan2(points[:, 1] - center_y, points[:, 0] - center_x)
        if clockwise:
            angles = -angles
        return points[np.argsort(angles)]

    @staticmethod
    @jit(nopython=True)
    def _change_aspect_ratio_jit(points: np.ndarray, aspect_ratio: float) -> np.ndarray:
        n_points = points.shape[0]

        # 中心点を手動計算
        center_x = np.sum(points[:, 0]) / n_points
        center_y = np.sum(points[:, 1]) / n_points

        # min/max計算
        min_x, max_x = np.min(points[:, 0]), np.max(points[:, 0])
        min_y, max_y = np.min(points[:, 1]), np.max(points[:, 1])

        # 幅と高さ
        width = max_x - min_x
        height = max_y - min_y

        # 新しい幅と高さ計算
        current_area = width * height
        new_height = np.sqrt(current_area / aspect_ratio)
        new_width = new_height * aspect_ratio

        # スケーリング係数
        width_scale = new_width / width
        height_scale = new_height / height

        # 結果配列を事前確保
        result = np.empty_like(points)

        # ベクトル化された計算
        for i in range(n_points):
            result[i, 0] = (points[i, 0] - center_x) * width_scale + center_x
            result[i, 1] = (points[i, 1] - center_y) * height_scale + center_y

        return result


class coordinate(Coordinate):
    @classmethod
    def bounding_rectangle(cls, points: np.ndarray | Tensor) -> np.ndarray:
        # x座標とy座標の最小値と最大値を取得
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        # 長方形の4つの頂点を生成（反時計回り）
        return np.array(
            [
                [min_coords[0], min_coords[1]],  # 左下
                [max_coords[0], min_coords[1]],  # 右下
                [max_coords[0], max_coords[1]],  # 右上
                [min_coords[0], max_coords[1]],  # 左上
            ]
        )

    @classmethod
    def polar_sort(cls, points: np.ndarray, clockwise=False) -> np.ndarray:

        # 重心を計算
        center_x, center_y = np.mean(points.view(), axis=0)

        # 角度計算
        if clockwise:
            # 時計回りの場合、角度を反転
            angles = np.arctan2(center_y - points[:, 1], center_x - points[:, 0])
        else:
            # 反時計回り
            angles = np.arctan2(points[:, 1] - center_y, points[:, 0] - center_x)

        # 角度でソート（numpy.argsort は O(n log n) の複雑性）
        return points[np.argsort(angles)]

    @classmethod
    def change_aspect_ratio(
        cls, points: np.ndarray | Tensor, aspect_ratio: float
    ) -> np.ndarray:
        # 中心点を計算
        center = np.mean(points.view(), axis=0)

        # 現在の幅と高さを計算
        width, height = np.max(points, axis=0) - np.min(points, axis=0)

        # 新しい幅と高さを計算（面積を保存）
        current_area = width * height
        new_height = np.sqrt(current_area / aspect_ratio)
        new_width = new_height * aspect_ratio

        # スケーリング係数を計算
        width_scale = new_width / width
        height_scale = new_height / height

        # ベクトル化された計算
        centered_vertices = points.view() - center  # ビューを使用
        scale_factors = np.array([width_scale, height_scale])
        return centered_vertices * scale_factors + center

    @classmethod
    def add_margin(
        cls, points: np.ndarray | Tensor, bottom_margin_rate: float = 0
    ) -> np.ndarray:
        _points = points
        _, height = np.max(_points, axis=0) - np.min(_points, axis=0)

        _, cy = _points.mean(axis=0)
        mask = _points[:, 1] > cy
        _points[:, 1][mask] = _points[:, 1][mask] + height * bottom_margin_rate
        return _points

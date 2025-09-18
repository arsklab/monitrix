from typing import Self, TypeVar
from PIL import Image

from torch import Tensor
import numpy as np

import cv2

P = TypeVar("P", Tensor, np.ndarray)

from ..abclass import Projective


class ProjectiveMatrixCV2(Projective):
    """cv2による射影変換（高速）"""

    def __init__(
        self, homography_matrix: P, interpolation: int = cv2.INTER_CUBIC
    ) -> None:
        """
        interpolation:
        Preferable interpolation methods are cv.INTER_AREA for shrinking and cv.INTER_CUBIC (slow) & cv.INTER_LINEAR for zooming
        """
        self.homography_matrix = self.to_numpy(homography_matrix)
        self.interpolation = interpolation

    @classmethod
    def get_perspective_transform(
        cls, from_points: P, to_points: P, interpolation: int = cv2.INTER_CUBIC
    ) -> Self:
        """透視変換行列matrixを取得"""
        from_pts = cls.to_numpy(from_points, dtype=np.float32)
        to_pts = cls.to_numpy(to_points, dtype=np.float32)
        return cls(cv2.getPerspectiveTransform(from_pts, to_pts), interpolation)

    def transform(self, points: P) -> np.ndarray:
        """透視変換行列matrixを適応"""
        return np.squeeze(
            cv2.perspectiveTransform(
                src=self.to_numpy(points)[np.newaxis], m=self.homography_matrix
            )
        )

    def perspective(self, image: np.ndarray) -> np.ndarray:
        """透視投影を画像に適応"""
        return cv2.warpPerspective(
            src=image,
            M=self.homography_matrix,
            dsize=image.shape[:-1][::-1],
            flags=self.interpolation,
        )

    @classmethod
    def to_numpy(cls, values: Tensor | np.ndarray, dtype=np.float32) -> np.ndarray:
        array = (
            values.detach().numpy() if isinstance(values, Tensor) else values
        ).astype(dtype)
        return np.ascontiguousarray(array) if not array.flags["C_CONTIGUOUS"] else array


from torchvision.transforms.functional import (
    _get_perspective_coeffs,
    _interpolation_modes_from_int,
    pil_modes_mapping,
)
import torchvision.transforms._functional_pil as F_pil
import torchvision.transforms._functional_tensor as F_t
from torchvision.transforms import InterpolationMode
import torch


class ProjectiveMatrixTorch(Projective):
    """PyTorchを利用した射影変換"""

    def __init__(
        self,
        coeffs: list[float],
        interpolation: InterpolationMode | int = InterpolationMode.BILINEAR,
    ) -> None:
        """
        interpolation:
        Preferable interpolation methods are cv.INTER_AREA for shrinking and cv.INTER_CUBIC (slow) & cv.INTER_LINEAR for zooming
        """
        if isinstance(interpolation, int):
            interpolation = _interpolation_modes_from_int(interpolation)
        elif not isinstance(interpolation, InterpolationMode):
            raise TypeError(
                "Argument interpolation should be a InterpolationMode or a corresponding Pillow integer constant"
            )
        self.coeffs = coeffs
        self.H = torch.tensor(coeffs + [1.0]).reshape(3, 3)
        self.H_inverse = torch.inverse(
            self.H
        )  # 座標と画像の座標系が異なるため、逆方向の変換が必要
        self.interpolation = interpolation

    @classmethod
    def get_perspective_transform(
        cls,
        from_points: P,
        to_points: P,
        interpolation: InterpolationMode | int | None = None,
    ) -> Self:
        """透視変換行列matrixを取得"""
        coeffs = _get_perspective_coeffs(from_points, to_points)
        return cls(coeffs) if interpolation is None else cls(coeffs, interpolation)

    def transform(self, points: Tensor) -> Tensor:
        """透視変換行列matrixを適応"""
        # 同次座標に変換
        x = torch.cat([torch.tensor(points), torch.ones(points.shape[0], 1)], dim=1).to(
            torch.float
        )
        # 変換
        transformed = (self.H_inverse @ x.T).T
        return transformed[:, :2] / transformed[:, 2:3]

    def perspective(
        self,
        image: Image.Image | np.ndarray | Tensor,
        fill: list[float] | None = None,
    ) -> np.ndarray:
        """透視投影を画像に適応
        - imageがTensorの場合は、[C, H, W] shapeが期待される
        return の画像は[H, W, C] shape
        """
        if isinstance(image, Image.Image):
            return np.array(
                F_pil.perspective(
                    image,
                    self.coeffs,
                    interpolation=pil_modes_mapping[self.interpolation],
                    fill=fill,
                )
            )
        elif isinstance(image, np.ndarray):
            image_chw = torch.tensor(image).permute(2, 0, 1)
            return (
                F_t.perspective(
                    image_chw,
                    self.coeffs,
                    interpolation=self.interpolation.value,
                    fill=fill,
                )
                .permute(1, 2, 0)
                .numpy()
            )
        else:
            return (
                F_t.perspective(
                    image,
                    self.coeffs,
                    interpolation=self.interpolation.value,
                    fill=fill,
                )
                .permute(1, 2, 0)
                .numpy()
            )

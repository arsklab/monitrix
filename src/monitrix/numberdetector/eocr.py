from dataclasses import dataclass, field
from typing import TypeVar, cast, TypeAlias
import re
from pathlib import Path
from functools import lru_cache
import warnings


import easyocr
import numpy as np
import torch
from torch import Tensor
from ultralytics.engine.results import BaseTensor

from ..abclass import Text2Value, NumberReader, ConfigDataclass
from ..results_protocol import OcrsProtocol
from ..image.imageprocess import letterbox, UniformImageProcessor

S = TypeVar("S", np.ndarray, list[np.ndarray])

MODEL_DIR = Path(__file__).parent / "weights"
assert MODEL_DIR


class Nums(BaseTensor, OcrsProtocol):
    def __init__(self, nums: Tensor | np.ndarray, orig_shape: tuple[int, int]) -> None:
        if isinstance(nums, np.ndarray):
            nums = torch.tensor(nums)
        if nums.ndim == 1:
            nums = nums[None, :]
        n = nums.shape[-1]

        assert n >= 7, f"expected 8 values but got {n}"
        super().__init__(nums, orig_shape)

    # def update(self, indices: Tensor | None = None, is_outlier: Tensor | None = None):
    def update(self, indices: Tensor | None = None):
        """データの更新"""
        if indices is not None:
            self.data[:, -3] = indices

    @property
    def xyxy(self) -> Tensor:
        return cast(Tensor, self.data[:, :4])

    # @property
    # def is_outlier(self) -> Tensor:
    #     return cast(Tensor, self.data[:, -3])

    @property
    def index(self) -> Tensor:
        return cast(Tensor, self.data[:, -3])

    @property
    def conf(self) -> Tensor:
        return cast(Tensor, self.data[:, -2])

    @property
    def number(self) -> Tensor:
        return cast(Tensor, self.data[:, -1])

    # def __add__(self, other:Self)->Self:


class Text2Number(Text2Value):
    """テキストを数値に変換する"""

    search_pattern: re.Pattern = re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)")

    @lru_cache(maxsize=2048)
    def is_number(self, text: str) -> bool:
        """数値かどうか"""
        try:
            float(text)
            return True
        except ValueError:
            return False

    @lru_cache(maxsize=2048)
    def search(self, text: str) -> str | None:
        """数値を探す"""
        match_ = self.search_pattern.search(text)
        return match_.group() if match_ else None

    @lru_cache(maxsize=2048)
    def to_float(self, text: str, enable_search: bool = True) -> float:
        """数値に変換"""
        try:
            return float(text)
        except ValueError:
            pass

        if enable_search:
            if (search_text := self.search(text)) is not None:
                warnings.warn(
                    f"`{text}` is not a number, but converted to a number `{search_text}`",
                    DeprecationWarning,
                )
                return float(search_text)
        raise ValueError(f"`{text}` cannot be converted to float.")


@dataclass
class EasyocrInitConfig(ConfigDataclass):
    lang_list: list[str] = field(default_factory=lambda: ["en"])
    gpu: bool = True
    cudnn_benchmark: bool = True
    detector: bool = True  # テキスト領域検出ON/OFF
    model_storage_directory: Path | None = MODEL_DIR


@dataclass
class EasyocrPredictConfig(ConfigDataclass):
    beamWidth: int = 1
    allowlist: str = "0123456789."
    min_size: int = 10
    rotation_info: list[int] = field(default_factory=lambda: [0])
    contrast_ths: float = 0.3
    adjust_contrast: float = 0.5

    paragraph: bool = False
    batch_size: int = 1

    # private（推論時の直接のパラメータではない）
    max_mag_ratio: float = field(default=3.0, metadata={"private": True})
    min_image_size: tuple[int, int] = field(
        default=(100, 300), metadata={"private": True}
    )
    min_text_box_height_size_rate: float = field(
        default=0.2, metadata={"private": True}
    )
    image_size_normalizer: UniformImageProcessor = field(
        default=letterbox, metadata={"private": True}
    )


standard_type: TypeAlias = tuple[list[list[np.int32]], str, float]


class EasyocrNumberReader(NumberReader):
    def __init__(
        self,
        init_config: EasyocrInitConfig = EasyocrInitConfig(),
        default_predict_conf: EasyocrPredictConfig = EasyocrPredictConfig(),
        text2value: Text2Value = Text2Number(),
    ):
        self.init_config = init_config
        self.default_predict_conf = default_predict_conf
        self._reader_instance = easyocr.Reader(**self.init_config.to_dict())
        self.text2value = text2value

    def _calculate_processing_params(
        self, image_shape: tuple[int, int]
    ) -> tuple[float, int]:
        """処理パラメータの事前計算"""
        height, width = self.default_predict_conf.min_image_size
        _height, _width = image_shape
        mag_ratio = min(
            width / _width, height / _height, self.default_predict_conf.max_mag_ratio
        )
        min_size = max(
            self.default_predict_conf.min_size,
            int(_height * self.default_predict_conf.min_text_box_height_size_rate),
        )

        return mag_ratio, min_size

    def _flatten(
        self,
        points: list[list[int]] | list[list[np.int32]],
    ) -> list[int]:
        (x1, y1), (x2, y2) = np.min(points, axis=0), np.max(points, axis=0)
        return [x1, y2, x2, y1]

    @classmethod
    def _sorted_score(cls, standard_type: standard_type) -> float:
        """画像中の最も大きな数（面積ではなく高さで判定）× confスコア"""
        points, _, conf = standard_type
        y_coords = [p[1] for p in points]
        height = max(y_coords) - min(y_coords)
        return float(height) * conf

    @classmethod
    def __empty(cls) -> list:
        return [
            torch.nan,
            torch.nan,
            torch.nan,
            torch.nan,
            torch.nan,
            torch.nan,
            torch.nan,
        ]

    def _to_tensor(
        self, standard_types: list[standard_type], top_k: int | None = 1
    ) -> Tensor:
        data = []
        for i, nums in enumerate(standard_types):
            count = 0
            _top_k = top_k if top_k else len(nums)
            if len(nums) != 0:
                _data = None
                for n in sorted(
                    nums,
                    key=lambda x: self._sorted_score(x),
                    reverse=True,
                ):
                    if count >= _top_k:
                        break
                    try:
                        _data = [
                            *self._flatten(n[0]),
                            i,
                            n[2],
                            self.text2value.to_float(n[1]),
                        ]
                        count += 1
                    except Exception as e:
                        continue
                if _data is None:
                    _data = self.__empty()
                    _data[4] = i
                data.append(_data)
            else:
                empty = self.__empty()
                empty[4] = i
                data.append(empty)
        return torch.tensor(data, dtype=torch.float64)

    def readnumber(self, image: np.ndarray, top_k: int | None = 1, **kwargs) -> Nums:
        # 処理パラメータ
        default_paramas = self.default_predict_conf.to_dict()

        # 処理パラメータの事前計算
        mag_ratio, min_size = self._calculate_processing_params(image.shape[:2])
        default_paramas["mag_ratio"] = mag_ratio
        default_paramas["min_size"] = min_size

        default_paramas.update(kwargs)

        # 変更不可なパラメータ
        default_paramas["output_format"] = "standard"

        # 文字認識
        results: standard_type = cast(
            standard_type,
            self._reader_instance.readtext(image=image, **default_paramas),
        )
        # 信頼度順でソート
        data = self._to_tensor([results], top_k)

        # count = 0
        # _top_k = top_k if top_k else len(results)
        # data = []
        # for n in sorted(
        #     results,
        #     key=lambda x: self._sorted_score(x),
        #     reverse=True,
        # ):
        #     if count >= _top_k:
        #         break
        #     try:
        #         data.append(
        #             [
        #                 *self._flatten(n[0]),
        #                 0,
        #                 n[2],
        #                 self.text2value.to_float(n[1]),
        #             ]
        #         )
        #         count += 1
        #     except Exception as e:
        #         print(e)
        #         continue
        if len(data) != 0:
            return Nums(data, image.shape[:2])
        else:
            return Nums(
                torch.full((1, 7), fill_value=torch.nan, dtype=torch.float64),
                image.shape[:2],
            )

    def readnumber_batch(
        self, images: list[np.ndarray], top_k: int | None = 1, **kwargs
    ) -> Nums:
        shape, resize_images = self.default_predict_conf.image_size_normalizer(images)

        # 処理パラメータ
        default_paramas = self.default_predict_conf.to_dict()

        # 処理パラメータの事前計算
        mag_ratio, min_size = self._calculate_processing_params(shape)
        default_paramas["mag_ratio"] = mag_ratio
        default_paramas["min_size"] = min_size

        default_paramas.update(kwargs)

        # 変更不可なパラメータ
        default_paramas["output_format"] = "standard"

        # 文字認識
        results: list[standard_type] = cast(
            list[standard_type],
            self._reader_instance.readtext_batched(
                image=resize_images, **default_paramas
            ),
        )

        # data = []
        # for i, nums in enumerate(results):
        #     count = 0
        #     _top_k = top_k if top_k else len(nums)
        #     for n in sorted(
        #         nums,
        #         key=lambda x: self._sorted_score(x),
        #         reverse=True,
        #     ):
        #         if count >= _top_k:
        #             break
        #         try:
        #             data.append(
        #                 [
        #                     *self._flatten(n[0]),
        #                     i,
        #                     n[2],
        #                     self.text2value.to_float(n[1]),
        #                 ]
        #             )
        #             count += 1
        #         except:
        #             continue
        data = self._to_tensor(results, top_k)
        return Nums(data, shape)

    def predict(
        self,
        source: list[np.ndarray] | np.ndarray,
        top_k: int | None = 1,
        batched: bool = True,
        **kwargs,
    ) -> Nums:
        """sourceが複数の画像で、batched=Falseの場合
        sizeが異なる画像を一枚一枚readnumberするので、Numのshapeを決定できない
        そのため(-1,-1)とする"""
        with self._cuda_context():
            if isinstance(source, list):
                if batched:
                    return self.readnumber_batch(source, top_k=top_k, **kwargs)
                else:
                    data = [
                        self.readnumber(image, top_k=top_k, **kwargs).data
                        for image in source
                    ]
                    for i, d in enumerate(data):
                        d[:, 4] = i
                    return Nums(
                        torch.cat(
                            data,
                            dim=0,
                        ),
                        (-1, -1),
                    )
            else:
                return self.readnumber(source, top_k=top_k, **kwargs)

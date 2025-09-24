from abc import ABC, abstractmethod
from typing import TypeAlias, TypeVar, Generator, Any, Self
import gc
from contextlib import contextmanager
from pathlib import Path
from dataclasses import dataclass, asdict, fields


from PIL import Image
from torch import Tensor
import torch
from numpy import ndarray
import numpy as np

from .results_protocol import ResultsProtocol, OcrsProtocol

# class ResultdataProtocol(Protocol):
#     """検出・認識結果プロトコル"""

#     @property
#     def data(self) -> Tensor | np.ndarray: ...


R = TypeVar("R", bound=ResultsProtocol)
P = TypeVar("P", Tensor, np.ndarray)
Source: TypeAlias = str | Path | int | Image.Image | ndarray | Tensor


class Config(ABC):
    """設定抽象クラス"""

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """（設定情報の辞書化）"""


class Detector(ABC):
    """オブジェクト検出器抽象クラス"""

    @abstractmethod
    def predict(
        self,
        source: Source | list[Source] | tuple[Source, ...],
        stream: bool,
        config: Config | None,
    ) -> Generator[R, None, None] | list[R]:
        """検出の実行（推定実行）"""

    @contextmanager
    def _cuda_context(self):
        """CUDA メモリ管理用のコンテキストマネージャー"""
        try:
            yield
        finally:
            if torch.cuda is not None:
                torch.cuda.empty_cache()
            gc.collect()


class PostProcess(ABC):
    """推論後のResultsに対する後処理抽象クラス"""

    @abstractmethod
    def __init__(self, config: Config) -> None:
        """初期化"""

    @abstractmethod
    def apply(self, results: list[R]) -> list[R]:
        """処理の実行（バッチ処理）
        メソッド内でのみ状態を持つ（インスタンスが保持する状態は関係しない）
        """

    def reset(self):
        """インスタンスの状態を初期化"""
        pass

    @abstractmethod
    def stream(self, result: R) -> R:
        """ストリーミング処理
        インスタンスが保持する状態を更新する
        インスタンスの状態は`reset`で初期化可能
        """

    def __repr__(self) -> str:
        return self.__class__.__name__

    # def to_dict(self, only_param: bool = True) -> dict[str, Any]:
    #     return {repr(self): self.config.to_dict(only_param)}


class Text2Value(ABC):
    """数字→数値変換抽象クラス"""

    @abstractmethod
    def is_number(self, text: str) -> bool:
        """数値かどうか"""

    @abstractmethod
    def search(self, text: str) -> str | None:
        """数値を探す"""

    @abstractmethod
    def to_float(self, text: str, **kwargs) -> float:
        """数値に変換"""


class NumberReader(Detector):
    """数字認識抽象クラス"""

    @abstractmethod
    def __init__(
        self, init_config: Config, default_predict_conf: Config, text2value: Text2Value
    ):
        """初期化"""

    @abstractmethod
    def readnumber(
        self, image: np.ndarray, top_k: int | None = 1, **kwargs
    ) -> OcrsProtocol:
        """画像中の数値読み取り"""

    @abstractmethod
    def readnumber_batch(
        self, images: list[np.ndarray], top_k: int | None = 1, **kwargs
    ) -> OcrsProtocol:
        """画像中の数値読み取り（複数画像）"""

    @abstractmethod
    def predict(
        self,
        source: list[np.ndarray] | np.ndarray,
        top_k: int | None = 1,
        batched: bool = True,
        **kwargs,
    ) -> OcrsProtocol:
        """数値読み取り"""


class Projective(ABC):
    """射影変換抽象クラス"""

    @classmethod
    @abstractmethod
    def get_perspective_transform(
        cls, from_points: P, to_points: P, interpolation
    ) -> Self:
        """座標からホモグラフィ行列算出"""

    @abstractmethod
    def transform(self, points: P) -> P:
        """座標を射影変換"""

    @abstractmethod
    def perspective(self, image: np.ndarray) -> np.ndarray:
        """画像を射影変換"""


class Coordinate(ABC):
    @classmethod
    @abstractmethod
    def bounding_rectangle(cls, points: P) -> np.ndarray:
        """点群を包含する最小の矩形（回転なし）"""

    @classmethod
    @abstractmethod
    def polar_sort(cls, points: P, clockwise: bool = False) -> np.ndarray:
        """
        極座標によるソート（中心点からの角度でソート）

        Parameters:
        -----------
        clockwise : bool, optional
            Trueの場合は時計回りでソート、Falseの場合は反時計回りでソート

        Returns:
        --------
        np.ndarray
            角度でソートされた点群、形状は (n, 2)
        """

    @classmethod
    @abstractmethod
    def change_aspect_ratio(cls, points: P, aspect_ratio: float) -> np.ndarray:
        """点群（矩形とは限らない）のアスペクト比を変更"""

    @classmethod
    @abstractmethod
    def add_margin(cls, points: P, bottom_margin_rate: float) -> np.ndarray:
        """下部のマージンを追加する"""

    @classmethod
    @abstractmethod
    def douglas_peucker(cls, points: P) -> np.ndarray:
        """
        Ramer–Douglas–Peucker アルゴリズムを使用した凸四角形の近似

        Args:
            polygon: 入力ポリゴン

        Returns:
            近似された4点の凸四角形
        """


@dataclass
class ConfigDataclass(Config):
    """基底設定クラス"""

    def to_dict(self, only_param: bool = True) -> dict[str, Any]:
        if only_param:
            _d = {}
            for f in fields(self):
                if not f.metadata.get("private", False):
                    value = getattr(self, f.name)
                    if issubclass(value.__class__, ConfigDataclass):
                        value = value.to_dict(only_param)
                    _d[f.name] = value
            return _d
        else:
            return asdict(self)

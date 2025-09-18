from pathlib import Path
from typing import Literal, Generator, TypeAlias, Self, cast, Callable, Any
from dataclasses import dataclass, field
from functools import partialmethod

from ultralytics.engine.results import Results, Boxes, Masks, BaseTensor
from ultralytics.utils import ops
from PIL import Image
from torch import Tensor
import torch
from numpy import ndarray

from ..abclass import Detector, ConfigDataclass, PostProcess

from ..results_protocol import (
    BoxesProtocol,
    AggregateProtocol,
    ResultsProtocol,
    OcrsProtocol,
)
from ..postprocess import StationaryTracker, ThresholdConfig, ClassConfFilter

DEFALUIT_WEIGHT = Path(__file__).parent / "weights/yolo11n_seg_c300.pt"
assert DEFALUIT_WEIGHT.exists()


Source: TypeAlias = str | Path | int | ndarray | Tensor
# Image.Image |


class exBoxes(Boxes, BoxesProtocol):
    """BoxesProtocol準拠"""

    def __init__(
        self, boxes: Tensor | ndarray | Boxes, orig_shape: tuple[int, int]
    ) -> None:
        if isinstance(boxes, Boxes):
            boxes = boxes.data
        if isinstance(boxes, ndarray):
            boxes = torch.tensor(boxes)
        if boxes.ndim == 1:
            boxes = boxes[None, :]
        n = boxes.shape[-1]
        assert n in {6, 7, 9}, f"expected 6, 7 or 9 values but got {n}"
        if n in {6, 7}:
            empty = torch.full((boxes.shape[0], 9), fill_value=torch.nan)
            empty[:, :4] = boxes[:, :4]
            empty[:, 7] = boxes[:, -2]
            empty[:, 8] = boxes[:, -1]
            match n:
                case 7:
                    empty[:, 6] = boxes[:, -3]
            boxes = empty

        BaseTensor.__init__(self, boxes, orig_shape)

        self.is_track = ~torch.isnan(boxes[:, -3]).all()
        self.orig_shape = orig_shape

    @property
    def id(self) -> Tensor:
        return cast(Tensor, super().id)

    @property
    def xyxy(self) -> Tensor:
        return cast(Tensor, super().xyxy)

    @property
    def conf(self) -> Tensor:
        return cast(Tensor, super().conf)

    @property
    def cls(self) -> Tensor:
        return cast(Tensor, super().cls)

    @property
    def is_base_box(self) -> Tensor:
        """ベースになる（包含する）インスタンスかどうか"""
        return cast(Tensor, self.data[:, -5])

    @property
    def base_box_id(self) -> Tensor:
        """包含するベースクラスのid"""
        return cast(Tensor, self.data[:, -4])

    def update(self, _id=None, is_base_box=None, base_box_id=None):
        """データの更新"""
        if _id is not None:
            self.data[:, -3] = _id
            self.is_track = ~torch.isnan(_id).all()

        if is_base_box is not None:
            self.data[:, -5] = is_base_box
        if base_box_id is not None:
            self.data[:, -4] = base_box_id


class Agg(BaseTensor, AggregateProtocol):
    def __init__(self, data: Tensor, orig_shape: tuple[int, int]) -> None:
        if data.ndim == 1:
            data = data[None, :]
        n = data.shape[-1]
        assert n == 8, f"expected 8 values but got {n}"
        super().__init__(data, orig_shape)


class exResults(Results, ResultsProtocol):
    """ResultsProtocol準拠"""

    def __init__(
        self, frame_no: int, orig_img, path, names, boxes, masks, ocrs=None, speed=None
    ) -> None:
        self.frame_no = frame_no
        self.orig_img: Tensor | ndarray = orig_img
        self.orig_shape: tuple[int, int] = orig_img.shape[:2]
        self.boxes: exBoxes = exBoxes(boxes, self.orig_shape)
        self.masks: Masks = Masks(masks, self.orig_shape)
        self.ocrs: OcrsProtocol | None = ocrs
        self.speed = (
            speed
            if speed is not None
            else {"preprocess": None, "inference": None, "postprocess": None}
        )
        self.names = names
        self.path = path
        self.save_dir = None

        self.probs = None
        self.keypoints = None
        self.obb = None

        self._keys = "boxes", "masks", "ocrs"

    def update(self, boxes=None, masks=None, ocrs=None):
        if boxes is not None:
            self.boxes = exBoxes(
                ops.clip_boxes(boxes, self.orig_shape), self.orig_shape
            )
        if masks is not None:
            self.masks = Masks(masks, self.orig_shape)
        if ocrs is not None:
            self.ocrs = ocrs

    @classmethod
    def cast(cls, frame_no: int, result: Results) -> Self:
        return cls(
            frame_no=frame_no,
            orig_img=result.orig_img,
            path=result.path,
            names=result.names,
            boxes=None if result.boxes is None else result.boxes.data,
            masks=None if result.masks is None else result.masks.data,
            speed=result.speed,
        )

    def new(self) -> Self:
        return self.__class__(
            frame_no=self.frame_no,
            orig_img=self.orig_img,
            path=self.path,
            names=self.names,
            boxes=None if self.boxes is None else self.boxes.data,
            masks=None if self.masks is None else self.masks.data,
            ocrs=None if self.ocrs is None else self.ocrs.data,
            speed=self.speed,
        )

    def cuda(self) -> Self:
        return cast(Self, super().cuda())

    def numpy(self) -> Self:
        return cast(Self, super().numpy())

    def cpu(self) -> Self:
        return cast(Self, super().cpu())

    def __getitem__(self, idx) -> Self:
        return cast(Self, super().__getitem__(idx))

    def to(self, *args, **kwargs) -> Self:
        return cast(Self, super().to(self, *args, **kwargs))

    @property
    def aggregate(self) -> Agg:
        _empty = torch.full((len(self.boxes), 8), torch.nan, dtype=torch.float64)
        _empty[:, 0] = self.frame_no
        _empty[:, 1:6] = self.boxes.data[:, 4:10]
        if self.ocrs is not None:
            _empty[:, 6:] = self.ocrs.data[:, -2:]

        return Agg(_empty, self.orig_shape)


@dataclass
class YOLOPredictConfig(ConfigDataclass):
    """yolo 推論時パラメータ"""

    conf: float = 0.83
    iou: float = 0.7

    max_det: int = 15

    half: bool = True
    imgsz: int = 736  # 32 * k
    batch: int = 16
    agnostic_nms: bool = True

    save: bool = False


@dataclass
class YOLOInitConfig(ConfigDataclass):
    """yolo 初期化時パラメータ"""

    model: Path = DEFALUIT_WEIGHT
    task: Literal["detect", "segment", "classify", "pose", "obb"] = "segment"
    verbose: bool = False

    tracker: Path | Literal["botsort.yaml", "bytetrack.yaml"] | list[PostProcess] = (
        field(
            default_factory=lambda: [
                ClassConfFilter(),  # confフィルタ
                StationaryTracker(
                    ThresholdConfig(0.80)
                ),  # トラッキング（インスタンスid割り振り）
            ],
            metadata={"private": True},
        )
    )

    default_config: ConfigDataclass = field(
        default_factory=YOLOPredictConfig, metadata={"private": True}
    )


class YOLOc300(Detector):
    """c300モニタ検出器（YOLO）"""

    def __init__(self, config: YOLOInitConfig = YOLOInitConfig()) -> None:
        super().__init__()
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics package is required for YOLODetector")

        self.model: YOLO = YOLO(**config.to_dict())
        self.config = config

    # def predict(
    #     self,
    #     source: Source | list[Source] | tuple[Source, ...],
    #     stream: bool = True,
    #     config: YOLOPredictConfig | None = None,
    # ) -> Generator[exResults, None, None]:
    #     """
    #     YOLO推論を実行

    #     Args:
    #         source: 入力ソース（画像、動画、パスなど）
    #         stream: ストリーミングモードかどうか
    #         config: 推論設定

    #     Yields:
    #         R: 推論結果（CPUに移動済み）
    #     """

    #     with self._cuda_context():
    #         try:
    #             if isinstance(self.config.tracker, PostProcess):
    #                 # トラキングは後処理に一任
    #                 results_generator = self.model.predict(
    #                     source,
    #                     stream=stream,
    #                     predictor=None,  #!!
    #                     **(config if config else self.config.default_config).to_dict(),
    #                 )
    #             else:
    #                 results_generator = self.model.track(
    #                     source,
    #                     stream=stream,
    #                     persist=True,  #!!
    #                     **(config if config else self.config.default_config).to_dict(),
    #                 )
    #             for i, result in enumerate(results_generator):
    #                 try:
    #                     # 結果をCPUに移動
    #                     yield exResults.cast(i, result.cpu())
    #                 except Exception as e:
    #                     print(f"Error processing result {i}: {e}")
    #                     continue
    #         except Exception as e:
    #             print(f"Prediction failed: {e}")
    #             raise
    #         finally:
    #             pass

    def predict(
        self,
        source: Source | list[Source] | tuple[Source, ...],
        stream: bool = True,
        config: YOLOPredictConfig | None = None,
    ) -> Generator[exResults, None, None]:
        """
        YOLO推論を実行

        Args:
            source: 入力ソース（画像、動画、パスなど）
            stream: ストリーミングモードかどうか
            config: 推論設定

        Yields:
            R: 推論結果（CPUに移動済み）
        """
        with self._cuda_context():
            if isinstance(self.config.tracker, list):
                return self.defertrack(source, stream, config)
            else:
                return self.track(source, stream, config)

    def defertrack(
        self,
        source: Source | list[Source] | tuple[Source, ...],
        stream: bool = True,
        config: YOLOPredictConfig | None = None,
    ) -> Generator[exResults, None, None]:
        """トラキングは後処理に一任"""

        if not isinstance(self.config.tracker, list):
            raise ValueError

        results_generator = self.model.predict(
            source,
            stream=stream,
            predictor=None,  #!!
            **(config if config else self.config.default_config).to_dict(),
        )

        for process in self.config.tracker:
            process.reset()

        try:
            for i, result in enumerate(results_generator):
                try:
                    # 結果をCPUに移動
                    result = exResults.cast(i, result.cpu())
                    # 後処理でトラッキング
                    for process in self.config.tracker:
                        result = process.stream(result)
                    yield result
                except Exception as e:
                    print(f"Error processing result {i}: {e}")
                    continue
        except Exception as e:
            print(f"Prediction failed: {e}")
            raise

    def track(
        self,
        source: Source | list[Source] | tuple[Source, ...],
        stream: bool = True,
        config: YOLOPredictConfig | None = None,
    ) -> Generator[exResults, None, None]:
        """model.trackを利用してトラッキング"""

        results_generator = self.model.track(
            source,
            stream=stream,
            persist=True,  #!!
            **(config if config else self.config.default_config).to_dict(),
        )
        try:
            for i, result in enumerate(results_generator):
                try:
                    # 結果をCPUに移動
                    yield exResults.cast(i, result.cpu())
                except Exception as e:
                    print(f"Error processing result {i}: {e}")
                    continue
        except Exception as e:
            print(f"Prediction failed: {e}")
            raise

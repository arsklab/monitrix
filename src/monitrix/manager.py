import gc
from typing import Literal, Any

from torch import Tensor
import torch
import numpy as np
from tqdm import tqdm

from monitrix.abclass import (
    NumberReader,
    Projective,
    Coordinate,
    Detector,
    PostProcess,
    Config,
)
from monitrix.abclass import Source, R
from monitrix.results_protocol import ResultsProtocol, BoxesProtocol, OcrsProtocol
from monitrix.image.imageprocess import crop
from monitrix.manager import Model, ImageGroupTextReader
from monitrix.objectdetector import yolo
from monitrix.postprocess import (
    InclusionProperty,
    InclusionConfig,
    ThresholdConfig,
    ClassConfFilter,
    StationaryTracker,
)
from monitrix.numberdetector.projective import ProjectiveMatrixCV2
from monitrix.numberdetector.geometry import coordinate_cv2jit
from monitrix.numberdetector import eocr
from monitrix.objectdetector.yolo import exResults
from monitrix.videoresult.resultsobject import VideoResults
from monitrix.postprocess import ConfidenceDefalut, IoU_Default


class Model:
    def __init__(self, detector: Detector, postprocess: list[PostProcess]) -> None:
        super().__init__()
        self.detector = detector
        self.postprocess = postprocess

    def applies(self, results: list[R]) -> list[R]:
        for process in self.postprocess:
            results = process.apply(results)
        return results

    def predict(
        self,
        source: Source | list[Source] | tuple[Source, ...],
        stream: bool = True,
        config: Config | None = None,
        context_release: bool = True,
    ) -> list[ResultsProtocol]:

        return self.applies(
            [
                _r
                for _r in self.detector.predict(
                    source,
                    stream=stream,
                    config=config,
                    context_release=context_release,
                )
            ]
        )


class ImageGroupTextReader:
    def __init__(
        self,
        reader: NumberReader,
        result_type: type[OcrsProtocol],
        projecter: type[Projective],
        coordinate: type[Coordinate],
        perspective: bool = False,
        aspect_ratio: float | None = 4 / 3,
        bottom_margin_rate: float | None = None,
        batched: bool = True,
    ) -> None:
        self.reader = reader
        self.result_type = result_type
        self.projecter = projecter
        self.coordinate = coordinate
        self.perspective = perspective
        self.aspect_ratio: float | None = aspect_ratio
        self.bottom_margin_rate = bottom_margin_rate
        self.batched = batched

    def base_ids(self, boxes: BoxesProtocol) -> Tensor:
        """ベースのid"""
        if boxes.id is not None:
            return boxes.id[boxes.is_base_box.to(torch.bool)].unique()
        raise ValueError

    def no_base_masks(self, boxes: BoxesProtocol, base_id: int | None = None) -> Tensor:
        """ベース以外のインスタンスインデックス"""
        if base_id is None:
            return ~boxes.is_base_box.to(torch.bool)
        else:
            return (~boxes.is_base_box.to(torch.bool)) & (boxes.base_box_id == base_id)

    def get_perspective(
        self, base_xy: Tensor, aspect_ratio: float | None = None
    ) -> Projective:
        """射影変換インスタンス生成"""
        base_approx_mask = self.approx_tetragon(base_xy)
        base_box = self.bounding_rectangle(base_approx_mask)

        return self.projecter.get_perspective_transform(
            base_approx_mask,
            (
                self.change_aspect_ratio(base_box, aspect_ratio)
                if aspect_ratio
                else base_box
            ),
            None,
        )

    def crop_image_list(
        self,
        image: np.ndarray,
        xy_list: list[Tensor],
        bottom_margin_rate: float | None = None,
        _projecter: Projective | None = None,
    ) -> list[np.ndarray]:
        """画像クリップ"""
        if _projecter is None:
            return [
                self.crop(
                    image,
                    self.add_margin(
                        self.bounding_rectangle(_xy), bottom_margin_rate
                    ).astype(int),
                )
                for _xy in xy_list
            ]
        else:
            _p_img = _projecter.perspective(image)
            return [
                self.crop(
                    _p_img,
                    self.add_margin(
                        self.bounding_rectangle(_projecter.transform(_xy)),
                        bottom_margin_rate,
                    ).astype(int),
                )
                for _xy in xy_list
            ]

    def apply(self, result: ResultsProtocol, **kwargs) -> ResultsProtocol:
        # if result.boxes is None:raise
        # if result.masks is None:raise

        boxes = result.boxes
        _ocr_results = torch.full((len(boxes), 7), torch.nan, dtype=torch.float64)

        try:
            if not self.perspective:
                no_base_masks: Tensor = self.no_base_masks(boxes)

                if no_base_masks.sum() == 0:
                    return result
                crip_image_list: list[np.ndarray] = self.crop_image_list(
                    result.orig_img,
                    result.masks[no_base_masks].xy,
                    self.bottom_margin_rate,
                )

                ocr_results = self.reader.predict(
                    crip_image_list, top_k=1, batched=self.batched, **kwargs
                )
                ocr_results.update(torch.where(no_base_masks)[0])

                for _i, values in zip(ocr_results.index, ocr_results.data):
                    _ocr_results[int(_i)] = values

            else:

                if boxes.id is not None:
                    base_ids = boxes.base_box_id[self.no_base_masks(boxes)]
                    unique_base_ids = base_ids.unique()

                    # baseごとに射影変換
                    for base in unique_base_ids:
                        base_indices = torch.where(boxes.id == base)[0]
                        if len(base_indices) == 1:
                            # ベースインスタンスが存在する

                            no_base_masks = self.no_base_masks(boxes, base)
                            if no_base_masks.sum() == 0:
                                continue

                            base_xy = result.masks.xy[base_indices[0]]
                            _projecter = self.get_perspective(
                                base_xy, self.aspect_ratio
                            )

                            crip_image_list: list[np.ndarray] = self.crop_image_list(
                                result.orig_img,
                                result.masks[no_base_masks].xy,
                                self.bottom_margin_rate,
                                _projecter,
                            )
                            ocr_results = self.reader.predict(
                                crip_image_list, top_k=1, batched=self.batched, **kwargs
                            )
                            ocr_results.update(torch.where(no_base_masks)[0])
                            for _i, values in zip(ocr_results.index, ocr_results.data):
                                _ocr_results[int(_i)] = values

                        elif len(base_indices) == 0:
                            # ベースインスタンスなし
                            no_base_masks = self.no_base_masks(boxes) & (
                                boxes.base_box_id == base
                            )
                            if no_base_masks.sum() == 0:
                                return result
                            crip_image_list: list[np.ndarray] = self.crop_image_list(
                                result.orig_img,
                                result.masks[no_base_masks].xy,
                                self.bottom_margin_rate,
                            )

                            ocr_results = self.reader.predict(
                                crip_image_list, top_k=1, batched=self.batched, **kwargs
                            )
                            ocr_results.update(torch.where(no_base_masks)[0])

                            for _i, values in zip(ocr_results.index, ocr_results.data):
                                _ocr_results[int(_i)] = values
                        else:
                            raise ValueError

            # !!! 属性強制追加 !!!
            result.update(
                ocrs=self.result_type(_ocr_results, orig_shape=result.orig_shape)
            )

        except Exception as e:
            raise e
        # finally:
        return result

    def approx_tetragon(self, mask_xy: np.ndarray | Tensor) -> np.ndarray:
        return self.coordinate.polar_sort(self.coordinate.douglas_peucker(mask_xy))

    def bounding_rectangle(self, mask_xy: np.ndarray | Tensor) -> np.ndarray:
        return self.coordinate.polar_sort(self.coordinate.bounding_rectangle(mask_xy))

    def change_aspect_ratio(
        self, mask_xy: np.ndarray | Tensor, aspect_ratio: float
    ) -> np.ndarray:
        return self.coordinate.change_aspect_ratio(mask_xy, aspect_ratio)

    def add_margin(
        self, mask_xy: np.ndarray | Tensor, bottom_margin_rate: float | None = None
    ) -> np.ndarray:
        if bottom_margin_rate is not None:
            return self.coordinate.add_margin(mask_xy, bottom_margin_rate)
        else:
            return mask_xy

    def crop(self, iamge: np.ndarray, sorted_mask_xy: np.ndarray) -> np.ndarray:
        return crop(iamge, sorted_mask_xy[0], sorted_mask_xy[2])

    @classmethod
    def to_tensor(cls, values: Tensor | np.ndarray) -> Tensor:
        """配列をTensorに変換"""
        if isinstance(values, Tensor):
            return values.detach()  # .clone()
        else:
            return torch.tensor(values)


def get_model(
    tracker: Literal["stationary", "botsort.yaml", "bytetrack.yaml"] = "stationary",
    predict_config: dict[str, Any] | yolo.YOLOPredictConfig = yolo.YOLOPredictConfig(),
    track_classconf_thresholds: (
        dict[int, float] | float | ThresholdConfig
    ) = ConfidenceDefalut,
    track_iou_thresholds: dict[int, float] | float | ThresholdConfig = IoU_Default,
    inclusion_ioa_thresholds: dict[int, float] | float | ThresholdConfig = 0.8,
) -> Model:

    if not isinstance(predict_config, yolo.YOLOPredictConfig):
        predict_config = yolo.YOLOPredictConfig(**predict_config)
    if not isinstance(inclusion_ioa_thresholds, ThresholdConfig):
        inclusion_ioa_thresholds = ThresholdConfig(inclusion_ioa_thresholds)

    match tracker:
        case "stationary":
            if not isinstance(track_classconf_thresholds, ThresholdConfig):
                track_classconf_thresholds = ThresholdConfig(track_classconf_thresholds)
            if not isinstance(track_iou_thresholds, ThresholdConfig):
                track_iou_thresholds = ThresholdConfig(track_iou_thresholds)

            init_config = yolo.YOLOInitConfig(
                tracker=[
                    ClassConfFilter(track_classconf_thresholds),  # confフィルタ
                    StationaryTracker(
                        track_iou_thresholds
                    ),  # トラッキング（インスタンスid割り振り）
                ],
                default_config=predict_config,
            )
        case "botsort.yaml" | "bytetrack.yaml":
            init_config = yolo.YOLOInitConfig(
                tracker=tracker, default_config=predict_config
            )
    return Model(
        yolo.YOLOc300(init_config),
        # 後処理
        [InclusionProperty(InclusionConfig(ioa=inclusion_ioa_thresholds))],  # 包含関係
    )


def get_textreader(
    perspective: bool = False, batched: bool = True
) -> ImageGroupTextReader:
    return ImageGroupTextReader(
        eocr.EasyocrNumberReader(
            init_config=eocr.EasyocrInitConfig(
                gpu=True, detector=True, cudnn_benchmark=True  # Falseに対応していない
            ),
            default_predict_conf=eocr.EasyocrPredictConfig(batch_size=16),
        ),
        result_type=eocr.Nums,
        projecter=ProjectiveMatrixCV2,
        coordinate=coordinate_cv2jit,
        perspective=perspective,
        aspect_ratio=4 / 3,
        # bottom_margin_rate = 0.05,
        batched=batched,
    )


class predictor:
    def __init__(
        self,
        tracker: Literal["stationary", "botsort.yaml", "bytetrack.yaml"] = "stationary",
        predict_config: (
            dict[str, Any] | yolo.YOLOPredictConfig
        ) = yolo.YOLOPredictConfig(),
        track_classconf_thresholds: (
            dict[int, float] | float | ThresholdConfig
        ) = ConfidenceDefalut,
        track_iou_thresholds: dict[int, float] | float | ThresholdConfig = IoU_Default,
        inclusion_ioa_thresholds: dict[int, float] | float | ThresholdConfig = 0.8,
        perspective: bool = False,
        batched: bool = True,
    ) -> None:
        self.model = get_model(
            tracker,
            predict_config,
            track_classconf_thresholds,
            track_iou_thresholds,
            inclusion_ioa_thresholds,
        )
        self.igtr = get_textreader(perspective, batched)

    def detect(self, source) -> list[exResults]:
        return list(self.model.predict(source))

    def ocr(self, detects: list[exResults]) -> VideoResults:
        with tqdm(total=len(detects), desc="read number", leave=True) as pbar:
            with self.igtr.reader._cuda_context(True):
                for k, result in enumerate(detects):
                    try:
                        self.igtr.apply(result, context_release=False)
                    except Exception as e:
                        print(k)
                        print(e)
                    finally:
                        pbar.update()
        return VideoResults(detects)

    def predict(self, source) -> VideoResults:
        return self.ocr(self.detect(source))

    def release(self):
        del self.model
        del self.igtr

        gc.collect()
        if torch.cuda is not None:
            torch.cuda.empty_cache()

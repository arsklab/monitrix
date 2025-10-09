from torch import Tensor
import torch
import numpy as np

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

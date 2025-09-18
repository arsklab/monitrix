from typing import TypeVar
from dataclasses import field, dataclass

# from ultralytics.engine.results import Boxes
# from ultralytics.engine.results import Results
from ultralytics.utils.metrics import box_iou, bbox_ioa

from torch import Tensor
import torch
import numpy as np


from .abclass import PostProcess, ConfigDataclass
from .results_protocol import BoxesProtocol, ResultsProtocol

R = TypeVar("R", bound=ResultsProtocol)


@dataclass
class ThresholdConfig(ConfigDataclass):
    thresholds: dict[int, float] | float


ConfidenceDefalut = ThresholdConfig(
    {
        0: 0.94,
        1: 0.7,
        2: 0.7,
        3: 0.7,
        4: 0.7,
        5: 0.7,
    }
)
IoU_Default = ThresholdConfig(0.85)


@dataclass
class InclusionConfig(ConfigDataclass):
    base_box1_cls: int = 0
    ioa: ThresholdConfig = field(default_factory=lambda: ThresholdConfig(0.8))


class ClassConfFilter(PostProcess):
    """クラスごとのconfフィルタ"""

    def __init__(self, config: ThresholdConfig = ConfidenceDefalut) -> None:
        self.config = config

    def stream(self, result: R) -> R:
        """ストリーミング処理"""
        indices = self.conf_filter(result.boxes, self.config.thresholds)
        if len(indices) != len(result.boxes.conf):
            return result[indices]
        return result

    def apply(self, results: list[R]) -> list[R]:
        """バッチ処理"""
        for i, result in enumerate(results):
            indices = self.conf_filter(result.boxes, self.config.thresholds)
            if len(indices) != len(result.boxes.conf):
                results[i] = result[indices]

        return results

    def conf_filter(
        self, boxes: BoxesProtocol, conf_thersholds: dict[int, float] | float
    ) -> Tensor:
        """クラスごとの確信度フィルタ"""
        thersholds = torch.tensor(
            [conf_thersholds[int(_cls)] for _cls in boxes.cls]
            if isinstance(conf_thersholds, dict)
            else [conf_thersholds] * len(boxes.cls)
        )

        if isinstance(boxes.conf, Tensor):
            conf = boxes.conf
        else:
            conf = torch.tensor(boxes.conf)

        return torch.where(conf >= thersholds)[0]


class StationaryTracker(PostProcess):
    """固定視座、静止オブジェクトトラッカー"""

    def __init__(self, config: ThresholdConfig = IoU_Default) -> None:
        self.config = config
        self.reset()

    def reset(self):
        self._state: Tensor | None = None
        self._k: int = 1

    def stream(self, result: R) -> R:
        """ストリーミング処理"""

        iou_base_id, _, state = self.iou_continuity(result.boxes, self._k, self._state)

        # !!! boxes の id更新 !!!
        result.boxes.update(_id=iou_base_id)

        self._k += 1
        self._state = state

        return result

    def apply(self, results: list[R]) -> list[R]:
        """
        静止状態オブジェクトのトラッキング
        Boxesにインスタンスidを設定する

        - 前回t-1の各オブジェクトとのIoU
        # - これまでの座標位置の平均との比較
        """
        state = None
        for k, result in enumerate(results):
            if boxes := result.boxes:
                iou_base_id, iou_score, state = self.iou_continuity(boxes, k, state)

                boxes.update(_id=iou_base_id)
                # n = boxes.shape[-1]
                # data = self.to_tensor(boxes.data)
                # match n:
                #     case 6:
                #         data = torch.cat(
                #             [data[:, :4], iou_base_id.unsqueeze(-1), data[:, 4:7]],
                #             dim=1,
                #         )
                #     case 7:
                #         data[:, -3] = iou_base_id
                # result.update(boxes=boxes.data)

        return results

    @classmethod
    def to_tensor(cls, values: Tensor | np.ndarray) -> Tensor:
        """配列をTensorに変換"""
        if isinstance(values, torch.Tensor):
            return values.detach().clone()
        else:
            return torch.tensor(values)

    def _iou_thresholds(self, boxes_cls: Tensor | np.ndarray) -> Tensor:
        """IoU閾値配列"""
        return torch.tensor(
            [self.config.thresholds[int(_cls)] for _cls in boxes_cls]
            if isinstance(self.config.thresholds, dict)
            else [self.config.thresholds] * len(boxes_cls)
        )

    def iou_continuity(
        self, boxes: BoxesProtocol, k: int, past_xyxy_state: Tensor | None = None
    ):
        """IoUによるインスタンスの連続性判定"""

        if past_xyxy_state is None:
            past_xyxy_state = self.to_tensor(boxes.xyxy)
            n = len(past_xyxy_state)
            return torch.arange(n), torch.zeros(n), past_xyxy_state
        else:
            object_ids, scores = [], []
            thresholds = self._iou_thresholds(boxes.cls)

            biou = box_iou(past_xyxy_state, boxes.xyxy)
            max_iou, index = biou.max(axis=0)

            for i, iou, threshold, xyxy in zip(index, max_iou, thresholds, boxes.xyxy):
                if iou >= threshold:
                    # 状態更新
                    past_xyxy_state[i] = (xyxy + k * past_xyxy_state[i]) / (
                        k + 1
                    )  # 平均位置
                    object_ids.append(int(i))
                    scores.append(iou)
                else:
                    # 状態更新
                    past_xyxy_state = torch.cat([past_xyxy_state, xyxy.unsqueeze(0)])
                    object_ids.append(len(past_xyxy_state))
                    scores.append(0)
            return torch.tensor(object_ids), torch.tensor(scores), past_xyxy_state


class InclusionProperty(PostProcess):
    """包含関係"""

    def __init__(self, config: InclusionConfig = InclusionConfig()) -> None:
        # self.base_box1_cls = base_box1_cls
        self.config = config
        self.reset()

    def reset(self):
        self._state: dict | None = None

    def stream(self, result: R) -> R:
        """ストリーミング処理"""
        if result.boxes.id is None:
            return result
        else:
            is_base_box, base_box_id, state = self.ioa_inclusion(
                result.boxes, self._state
            )
            # !!! boxに属性追加 !!!
            result.boxes.update(is_base_box=is_base_box, base_box_id=base_box_id)

            self._state = state

            return result

    def apply(self, results: list[R]) -> list[R]:
        """ベースインスタンスが包含するインスタンスを判定
        is_base_box: ベースインスタンスかどうか
        base_box_id: そのインスタンスを包含するベースインスタンスid
        """
        state = None
        for result in results:
            if boxes := result.boxes:
                if boxes.id is None:
                    continue
                is_base_box, base_box_id, state = self.ioa_inclusion(boxes, state)

                # !!! boxに属性追加 !!!
                boxes.update(is_base_box=is_base_box, base_box_id=base_box_id)
                # result.update(boxes=boxes.data)
            # boxes.is_base_box = is_base_box
            # boxes.base_box_id = base_box_id

        return results

    def _ioa_thresholds(self, boxes_cls: Tensor | np.ndarray) -> Tensor:
        """IoA閾値配列"""
        return torch.tensor(
            [self.config.ioa.thresholds[int(_cls)] for _cls in boxes_cls]
            if isinstance(self.config.ioa.thresholds, dict)
            else [self.config.ioa.thresholds] * len(boxes_cls)
        )

    def ioa_inclusion(self, boxes: BoxesProtocol, inclusion_state: dict | None = None):
        """IoAによる包含関係の判定
        IoAによる推定ができない場合：（ベースが存在しないが過去包含されていたインスタンスは存在するなど）
        - k-1の状態から推定する
        """
        if boxes.id is None:
            raise ValueError

        if inclusion_state is None:
            inclusion_state = {}

        base_box_mask = boxes.cls == self.config.base_box1_cls
        is_base_box = (base_box_mask).to(torch.float32)

        base_box_index = torch.where(base_box_mask)[0]

        if len(base_box_index) != 0:
            bioa = bbox_ioa(boxes[base_box_index].xyxy, boxes.xyxy)
            ioas = bioa.max(axis=0)[0]
            indies = bioa.argmax(axis=0)

            thresholds = self._ioa_thresholds(boxes.cls)

            base_box_id = []
            for index, ioa, threshold, _id in zip(indies, ioas, thresholds, boxes.id):
                if ioa >= threshold:
                    inclusion_id = boxes.id[base_box_index][index]
                    base_box_id.append(inclusion_id)
                    # 状態更新
                    inclusion_state[int(_id)] = int(inclusion_id)
                else:
                    # 過去の状態から推定
                    base_box_id.append(inclusion_state.get(int(_id), np.nan))

        else:
            # 過去の状態から推定
            base_box_id = [inclusion_state.get(int(_id), np.nan) for _id in boxes.id]

        return is_base_box, torch.tensor(base_box_id), inclusion_state

from dataclasses import field, dataclass

from ..abclass import PostProcess, ConfigDataclass, PostProcessPipeline
from ..postprocess import (
    StationaryTracker,
    ThresholdConfig,
    ClassConfFilter,
    InclusionProperty,
    InclusionConfig,
)

from ..postprocess import ConfidenceDefalut, IoU_Default


@dataclass
class FVSOTrackerConfig(ConfigDataclass):
    conf_threshold: ThresholdConfig = field(default_factory=lambda: ConfidenceDefalut)
    iou_threshold: ThresholdConfig = field(default_factory=lambda: IoU_Default)


class FVSOTracker(PostProcessPipeline):
    """Fixed Viewpoint Stationary Object Tracker"""

    def __init__(self, config: FVSOTrackerConfig):
        super().__init__(
            config,
            [
                ClassConfFilter(config.conf_threshold),  # confフィルタ
                StationaryTracker(
                    config.iou_threshold
                ),  # トラッキング（インスタンスid割り振り）
            ],
        )


class Inclusion(PostProcessPipeline):
    def __init__(self, config: InclusionConfig):
        super().__init__(config, [InclusionProperty(config)])  # 包含関係

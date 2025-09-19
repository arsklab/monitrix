from pathlib import Path
import pandas as pd
import torch
import cv2
from tqdm import tqdm
import numpy as np

from monitrix.results_protocol import ResultsProtocol
from monitrix.draw import plot
from monitrix.metric import mdataframe


class VideoWriter:
    """動画保存クラス"""

    # OpenCVのコーデックマッピング
    CODEC_MAPPING = {
        ".mp4": "mp4v",  # または 'avc1' (H.264)
        ".avi": "XVID",
        ".mov": "mp4v",
    }

    def __init__(self, width: int, height: int, fps: float, output_path: Path):
        """
        初期化
        Args:
            config: 動画出力の設定
        """

        # コーデックの決定
        fourcc_code = self.CODEC_MAPPING.get(output_path.suffix.lower(), "mp4v")
        fourcc = cv2.VideoWriter_fourcc(*fourcc_code)

        # VideoWriterの初期化
        self.writer = cv2.VideoWriter(
            str(output_path.as_posix()),
            fourcc,
            fps,
            (width, height),
            isColor=True,
        )

        if not self.writer.isOpened():
            raise RuntimeError("VideoWriterの初期化に失敗しました")

    def write(self, frame: np.ndarray):
        """
        フレームを書き込む
        Args:
            frame: BGR形式の画像
        """
        # フレームの書き込み
        self.writer.write(frame)

    def release(self):
        """リソースの解放"""
        if self.writer:
            self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()


class VideoResults(list[ResultsProtocol]):
    columns: list[str] = [
        "frame",
        "is_monitor",
        "monitor_id",
        "id",
        "object_conf",
        "class",
        "ocr_conf",
        "pred_value",
    ]

    preset_colors = {
        0: ("#F69642", "#01204E"),
        1: ("#68C9FA", "#01204E"),
        2: ("#8868FA", "#01204E"),
        3: ("#FA68D5", "#01204E"),
        4: ("#F10674", "#EDE1E9"),
    }
    preset_classnames = {
        0: "main screen",
        1: "oxygen%",
        2: "MV L/min",
        3: "VT mL",
        4: "breathing rate",
    }

    @property
    def names(self) -> dict[int, str]:
        return self[0].names

    @property
    def path(self) -> Path:
        return Path(self[0].path)

    @property
    def size(self) -> tuple[int, int]:
        return self[0].orig_shape

    def to_df(self, decimals: int = 6, cast: bool = True) -> pd.DataFrame | mdataframe:
        _df = pd.DataFrame(
            torch.round(
                torch.cat([result.aggregate.data for result in self]), decimals=decimals
            ).numpy(),
            columns=self.columns,
        ).astype(
            {
                "frame": int,
                "id": int,
                "class": int,
                #  "monitor_id":int,
                "is_monitor": bool,
            }
        )
        if cast:
            return mdataframe(_df)
        else:
            return _df

    def to_csv(self, decimals: int = 6, /, **kwargs):
        return self.to_df(decimals=decimals).to_csv(**kwargs)

    def to_video(
        self,
        output_path: str | Path,
        resize: float = 1,
        fps: float = 30,
        trues: dict[int, dict[int, int | float]] | None = None,
    ):
        _size: tuple[int, int] = tuple(int(s * resize) for s in self.size[::-1])

        with VideoWriter(*_size, fps, Path(output_path)) as writer:
            with tqdm(total=len(self), desc="write frame", leave=True) as pbar:
                _trues = {} if trues is None else trues
                for frame in self:
                    if resize != 1:
                        writer.write(
                            cv2.resize(
                                plot(
                                    frame,
                                    class_names=self.preset_classnames,
                                    colors=self.preset_colors,
                                    trues=_trues.get(frame.frame_no, None),
                                ),
                                _size,
                            )
                        )
                    else:
                        writer.write(
                            plot(
                                frame,
                                class_names=self.preset_classnames,
                                colors=self.preset_colors,
                                trues=_trues.get(frame.frame_no, None),
                            )
                        )
                    pbar.update()

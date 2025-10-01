from pathlib import Path
from io import BytesIO
from PIL import Image

import pandas as pd
import torch
import cv2
from tqdm import tqdm
import numpy as np
import h5py

from monitrix.results_protocol import ResultsProtocol
from monitrix.draw import plot
from monitrix.metric import mdataframe
from monitrix.numberdetector.eocr import Nums
from monitrix.objectdetector.yolo import exResults


class VideoWriter:
    """動画保存クラス"""

    def __init__(
        self,
        width: int,
        height: int,
        fps: float,
        output_path: Path,
        fourcc_code: Literal["h264", "mp4v", "avc1", "XVID"] = "h264",
    ):
        """
        初期化
        Args:
            config: 動画出力の設定
        """

        # コーデックの決定
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

    def to_df(
        self,
        decimals: int = 6,
        cast: bool = True,
        sort: bool = True,
        drop_duplicate: bool = True,
    ) -> pd.DataFrame | mdataframe:
        _df = (
            pd.DataFrame(
                torch.round(
                    torch.cat([result.aggregate.data for result in self]),
                    decimals=decimals,
                ).numpy(),
                columns=self.columns,
            )
            .dropna(subset="id")
            .astype(
                {
                    "frame": int,
                    "id": int,
                    "class": int,
                    #  "monitor_id":int,
                    "is_monitor": bool,
                }
            )
        )
        if sort:
            _df["conf_area"] = 1 - _df["object_conf"].fillna(1e-10) * _df[
                "ocr_conf"
            ].fillna(1e-10)
            _df = (
                _df.sort_values(["frame", "monitor_id", "class", "conf_area"])
                .drop("conf_area", axis=1)
                .reset_index(drop=True)
            )
        if drop_duplicate:
            mask = _df.duplicated(["frame", "monitor_id", "class"], keep="first")
            _df = _df[~mask].reset_index(drop=True)
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

    def to_hdf5(self, output_path: str | Path, format: str = "jpeg", **kwargs):
        with h5py.File(output_path, "w") as file:
            for frame in self:
                group = file.create_group(f"{frame.frame_no}")

                # 属性
                group.attrs["frame_no"] = frame.frame_no
                group.attrs["path"] = frame.path
                subgroup = group.create_group("speed")
                for k, v in frame.speed.items():
                    subgroup.attrs[k] = v
                subgroup = group.create_group("names")
                for k, v in frame.names.items():
                    subgroup.attrs[str(k)] = v

                # 画像
                byte_io = BytesIO()
                Image.fromarray(frame.orig_img).save(byte_io, format=format, **kwargs)
                group.create_dataset(
                    "orig_img",
                    data=np.frombuffer(byte_io.getvalue(), dtype=np.uint8),
                    compression="gzip",
                )

                # boxes
                group.create_dataset(
                    "boxes",
                    data=frame.boxes.data,
                    compression="gzip",
                )
                # masks
                group.create_dataset(
                    "masks",
                    data=frame.masks.data,
                    compression="gzip",
                )
                # ocrs
                if frame.ocrs is not None:
                    group.create_dataset(
                        "ocrs",
                        data=frame.ocrs.data,
                        compression="gzip",
                    )

    @classmethod
    def load_hdf5(
        cls, output_path: str | Path, result_type: type[ResultsProtocol] = exResults
    ) -> "VideoResults":
        with h5py.File(output_path, "r") as file:
            buf = []
            for key in sorted(file.keys()):
                group = file[key]
                orig_img = np.array(Image.open(BytesIO(group["orig_img"][:].tobytes())))
                data = dict(
                    frame_no=group.attrs["frame_no"],
                    path=group.attrs["path"],
                    names={int(k): v for k, v in group["names"].attrs.items()},
                    speed={k: v for k, v in group["speed"].attrs.items()},
                    orig_img=orig_img,
                    boxes=group["boxes"][:],
                    masks=group["masks"][:],
                    ocrs=(
                        Nums(group["ocrs"][:], orig_img.shape[:2])
                        if "ocrs" in group.keys()
                        else None
                    ),
                )
                buf.append(result_type(**data))
        return VideoResults(buf)

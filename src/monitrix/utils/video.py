# Copyright (C) 2024 Arisaka Naoya
# SPDX-License-Identifier: AGPL-3.0-or-later

from pathlib import Path
from dataclasses import dataclass
from typing import Generator

import numpy as np
import numpy.typing as npt
import cv2


from monitrix.utils.logger import get_logger

logger = get_logger("monitrix.utils")


def to_video(
        frames:list[npt.NDArray[np.uint8]],
        fps:float = 30,
        output_path:Path = Path("output.mp4")
):
    """画像（frame）をまとめて動画に変換し保存する"""

    height, width,_ = frames[0].shape

    # 設定
    config = VideoWriterConfig(
            width=width,
            height=height,
            fps=fps,
            output_path=output_path,
    )

    with (
        SimpleVideoWriter(config) as writer
        ):
        # 順次読み込み
        for frame in frames:
            # フレームの処理
            writer.write(frame)



class VideoFrameGenerator:
    """動画フレームを読み込むためのジェネレータクラス"""
    
    def __init__(self, video_path: str):
        """
        初期化
        
        Args:
            video_path (str): 動画ファイルのパス
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"動画ファイルを開けません: {video_path}")
            
        # 動画の基本情報を取得
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()
    
    def get_frame_sequential(self) -> Generator[npt.NDArray[np.uint8], None, None]:
        """
        フレームを1つずつ順番に読み込むジェネレータ
        
        Yields:
            画像データ(BGRチャンネル付きHWCフォーマット): npt.NDArray[np.uint8]
        """

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
    
            yield frame.astype(np.uint8)

    
    def get_frame_batch(self, 
                       batch_size: int = 32,
                       overlap: int = 0
                       ) -> Generator[list[npt.NDArray[np.uint8]], None, None]:
        """
        フレームをバッチで読み込むジェネレータ
        
        Args:
            batch_size (int): バッチサイズ
            overlap (int): バッチ間のオーバーラップするフレーム数
        
        Yields:
            List[npt.NDArray[np.uint8]]: フレームのリスト
        """
        if overlap >= batch_size:
            raise ValueError("オーバーラップはバッチサイズより小さくする必要があります")
            
        batch = []
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                if batch:  # 最後のバッチがある場合は出力
                    yield batch
                break
            
            batch.append(frame.astype(np.uint8))

            if len(batch) == batch_size:
                yield batch
                # オーバーラップを考慮して次のバッチの開始位置を設定
                if overlap > 0:
                    batch = batch[-overlap:]
                else:
                    batch = []


    def get_frame_skip(self, 
                      skip_frames: int = 5
                      ) -> Generator[npt.NDArray[np.uint8], None, None]:
        """
        指定したフレーム数をスキップしながら読み込むジェネレータ
        
        Args:
            skip_frames (int): スキップするフレーム数
        
        Yields:
            画像データ(BGRチャンネル付きHWCフォーマット): npt.NDArray[np.uint8]
        """
        frame_number = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if frame_number % (skip_frames + 1) == 0:
                yield frame.astype(np.uint8)

            frame_number += 1
    
    def seek(self, frame_number: int) -> bool:
        """
        指定したフレーム番号にシーク
        
        Args:
            frame_number (int): シークするフレーム番号
        
        Returns:
            bool: シークが成功したかどうか
        """
        return self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)





@dataclass
class VideoWriterConfig:
    """動画出力の設定"""
    width: int
    height: int
    fps: float
    output_path: str | Path

class SimpleVideoWriter:
    """シンプルな動画保存クラス"""
    
    # OpenCVのコーデックマッピング
    CODEC_MAPPING = {
        '.mp4': 'mp4v',  # または 'avc1' (H.264)
        '.avi': 'XVID',
        '.mov': 'mp4v',
    }
    
    def __init__(self, config: VideoWriterConfig):
        """
        初期化
        Args:
            config: 動画出力の設定
        """
        self.config = config
        self.output_path = Path(config.output_path)
        
        # コーデックの決定
        fourcc_code = self.CODEC_MAPPING.get(self.output_path.suffix.lower(), 'mp4v')
        fourcc = cv2.VideoWriter_fourcc(*fourcc_code)
        
        # VideoWriterの初期化
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            config.fps,
            (config.width, config.height),
            isColor=True
        )
        
        if not self.writer.isOpened():
            logger.error("VideoWriterの初期化に失敗しました")
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

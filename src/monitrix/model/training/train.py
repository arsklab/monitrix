# Copyright (C) 2024 Arisaka Naoya
# SPDX-License-Identifier: AGPL-3.0-or-later

import subprocess
import os
import yaml
from pathlib import Path

from monitrix.model.dnyolocr import TaskType, SegmentModel, load_seg
from monitrix.utils.logger import get_logger

logger = get_logger("monitrix.model")


def command(args):
    """
    コマンドを実行し、出力を取得
    ! 実行には注意する
    """
    result = subprocess.run(args, capture_output=True, text=True)
    return result.stdout



def load_yaml(file_path):
    with open(file_path, 'r', encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config

def save_yaml(config, file_path):
    with open(file_path, 'w', encoding="utf-8") as file:
        yaml.dump(config, file)


def labelme_to_yolo_data_format(
        labelme_json_files_folder:Path|str,
        task:TaskType,
        val_size:float = 0.2,
        test_size:float = 0.0,
        seed:int = 42,
        # exist_ok:bool=False
    )->Path|None:
    """
    labelme形式 を yolo形式 にデータ構造を変換

    - 動作未確認（実際に学習できるか試していない）-> OK
        - dataset.yamlの一部が元のデータと異なる
            - test: null

    https://pypi.org/project/labelme2yolo/
    """

    # LabelMe JSON files チェック
    labelme_path = Path(labelme_json_files_folder)
    assert labelme_path.is_dir(), f"存在しないディレクトリ（ファイルパスが間違っている）：{labelme_path.resolve()}"
    assert len(list(labelme_path.glob("*.json")))!=0, "jsonファイルが存在しない"

    # 変換コマンド作成
    assert 0. <= val_size <= 1.
    assert 0. <= test_size <= 1.
    args = [
        "labelme2yolo",
        "--json_dir", labelme_path.as_posix(),
        "--val_size", str(val_size),
        "--test_size", str(test_size),
        "--output_format", task.format_option(),
        "--seed", str(seed)
        ]

    # 変換後のフォルダ`YOLODataset`が既に存在しているかどうか（出力するフォルダ名はコマンドから変更できない）
    # if (labelme_path / "YOLODataset").is_dir() and (not exist_ok):
    #     raise FileExistsError("保存先のフォルダが既に存在する")

    # 最終確認
    if input(f"yolo形式のデータセットを{labelme_path.as_posix()}に作成しますか? (y/n): ").lower() != 'y':
        return
    
    # 実行
    logger.info(command(args))

    # フォルダ名の変更
    folder = labelme_path / ("YOLODataset" + task.format_option().capitalize())
    os.rename(
        labelme_path / "YOLODataset",
        folder)

    # yamlの書き換え
    yaml_path = folder / "dataset.yaml"
    config = load_yaml(yaml_path)
    config['path'] = config['path'].replace("YOLODataset", "YOLODataset" + task.format_option().capitalize())

    save_yaml(config, yaml_path)
    logger.info(f"完了:{folder.as_posix()}")

    return yaml_path



def train_seg_model(
        yaml_config:Path,
        model:SegmentModel|Path|str=SegmentModel.YOLO11n_seg,
        epochs:int = 50,
        imgsz:int = 640,
        device:int = 0,
        project:str|None = None, # 保存したいディレクトリパス
        name:str|None = None, # 実験名（これがサブディレクトリになります）
):
    """学習の実行"""
    assert yaml_config.is_file()

    # モデル読み込み
    s_model = load_seg(model)

    return s_model.train(
        data=yaml_config, 
        epochs=epochs, 
        imgsz=imgsz, 
        device=device,
        project=project,
        name=name
        )


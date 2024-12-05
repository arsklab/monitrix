# Copyright (C) 2024 Arisaka Naoya
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
from pathlib import Path

import cv2

from monitrix.model import dnyolocr

def predict():
    parser = argparse.ArgumentParser(description='モニター画面上の数値認識（開発版）')
    parser.add_argument('image', help='ファイルパス')
    parser.add_argument('--save_folder', '-s', default=None, help='画像保存先パス(オプション)')
    parser.add_argument('--model', '-m', default=None, help='モデルの重みパス(オプション)')
    args = parser.parse_args()

    if args.image is not None:
        image_path = Path(args.image)
        if not image_path.is_file():
            print(f"画像ファイルが存在しない：{image_path.as_posix()}")
            return
    else:
        return
    
    if args.save_folder is not None:
        save_folder = Path(args.save_folder)
    else:
        save_folder = image_path.parent / "pred"
    if not save_folder.is_dir():
        save_folder.mkdir(parents=True,exist_ok=True)

    if args.model is not None:
        if not Path(args.model).is_file():
            print(f"モデルファイルが存在しない：{Path(args.model).as_posix()}")
            return

    model = dnyolocr.DNYOLOCR(model=args.model)
    result = model.predict(args.image)[0]
    
    # 結果の表示
    print(result.values())

    # 画像出力
    output_path = save_folder / ("p_" + image_path.name)
    cv2.imwrite(str(output_path),result.draw("total",screen_numbers_margin=100,to_rgb=False))
    print(f"save: {output_path.as_posix()}")


def predict_video():
    parser = argparse.ArgumentParser(description='モニター画面上の数値認識（開発版）')
    parser.add_argument('video', help='ファイルパス')
    parser.add_argument('--save_folder', '-s', default=None, help='画像保存先パス(オプション)')
    parser.add_argument('--model', '-m', default=None, help='モデルの重みパス(オプション)')
    args = parser.parse_args()

    if args.video is not None:
        video_path = Path(args.video)
        if not video_path.is_file():
            print(f"画像ファイルが存在しない：{video_path.as_posix()}")
            return
    else:
        return
    
    if args.save_folder is not None:
        save_folder = Path(args.save_folder)
    else:
        save_folder = video_path.parent / "pred"
    if not save_folder.is_dir():
        save_folder.mkdir(parents=True,exist_ok=True)

    if args.model is not None:
        if not Path(args.model).is_file():
            print(f"モデルファイルが存在しない：{Path(args.model).as_posix()}")
            return

    model = dnyolocr.DNYOLOCR(model=args.model)
    _ = model.predict_video(args.video, save_folder)


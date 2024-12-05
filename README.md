
# Monitrix


Monitrix is a numerical recognition system for monitors and displays utilizing image recognition and OCR technology. Using a YOLO-based custom model, it can extract numerical information from videos and images. / Monitrixは画像認識とOCR技術を活用したモニターやディスプレイ上の数値認識システムです。

## Requirements / 動作環境
- Python 3.10+
- ultralytics 8.x
- easyocr 1.7.x

## Install / インストール

1. [Install pytorch](https://pytorch.org/get-started/locally/)
2. [Install ultralytics](https://docs.ultralytics.com/quickstart/)
3. Install via pip (pipを使用してインストール):
    ```bash
    pip install monitrix
    ```

## Usage / 使用例

### Command Line Interface / コマンドライン
- Image / 画像
```bash
monitrix_predict [image_path]
```
- Video / 動画
```bash
monitrix_predict_video [video_path]
```

### Python Script / Pythonスクリプト
For simple usage examples, please refer to the Jupyter notebook at `example/example.ipynb`. / 簡単な使用例は`example/example.ipynb`のJupyterノートブックを参照してください。

## Features / 主な機能
- Segmentation using custom YOLO model / カスタムYOLOモデルによるセグメンテーション
- Numerical recognition OCR / 数値認識OCR
- Command line interface / コマンドラインインターフェース

## License / ライセンス
This project is licensed under the GNU Affero General Public License v3 or later (AGPLv3+).

### Third-party Libraries / 使用ライブラリ
- Ultralytics (AGPL-3.0)
- easyocr (Apache License 2.0)

## Source Code / ソースコード
The complete source code is available at:
- [GitHub](https://github.com/arsklab/monitrix)

## Disclaimer / 免責事項

WARNING: This software is intended for non-medical use only, specifically for monitoring and reading numerical values from standard displays and monitors.

注意: 本ソフトウェアは非医療用途に限定され、特に標準的なディスプレイやモニターからの数値読み取りを目的としています。

The accuracy and reliability of readings may be affected by various factors including lighting conditions, display quality, and camera settings.
（読み取りの精度と信頼性は、照明条件、ディスプレイの品質、カメラ設定など、様々な要因の影響を受ける可能性があります）

In addition to the warranty disclaimer and limitation of liability provided by the AGPL-3.0 license:
AGPL-3.0ライセンスで提供される保証の免責事項および責任の制限に加えて：

- This software is not certified as a medical device and is provided for research purposes only
（本ソフトウェアは医療機器として認証されておらず、研究目的でのみ提供されています）
- No warranty is made regarding the accuracy or reliability of any numerical readings
（数値の正確性、信頼性については一切保証いたしません）
- The developers shall not be liable for any medical decisions or actions taken based on the output of this software
（開発者は、本ソフトウェアの出力に基づいて行われるいかなる医療上の決定または行動についても責任を負いません）
- Users acknowledge that they use this software at their own risk for non-medical purposes only
（利用者は、医療目的でない場合に限り、自己の責任において本ソフトウェアを使用することを認めます）




## Citation / 引用

If you use Monitrix in your research, please cite it as follows:
研究でMonitrixを使用する場合は、以下のように引用してください：

### BibTeX
```bibtex
@software{monitrix2024,
  title = {Monitrix: A Numerical Recognition System for Displays},
  author = {Arisaka Naoya},
  year = {2024},
  url = {https://github.com/arsklab/monitrix},
}
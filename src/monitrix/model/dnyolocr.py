# Copyright (C) 2024 Arisaka Naoya
# SPDX-License-Identifier: AGPL-3.0-or-later

from enum import Enum, auto
from pathlib import Path
from typing import TypeVar,Type,Sequence,Literal,Any
from functools import singledispatch
import gc

from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from ultralytics.engine.results import Results as uResults

from monitrix.core.plane_geometry import Rectangle2D
from monitrix.core.main_screen import ExtractionProtocol, MaximumProbClasses, OnScreenNumbers, OnScreenNumbersOCR
from monitrix.core.results import Result
from monitrix.ocr import number
from monitrix.utils import video
from monitrix.utils.logger import get_logger

logger = get_logger("monitrix.model")



class TaskType(Enum):
    CLASSIFY = auto()
    DETECT = auto()
    SEGMENT = auto()
    POSE = auto()
    OBB = auto()

    def format_option(self)->str:
        """labelme2yolo output_format option"""
        if self == TaskType.DETECT:
            return "bbox"
        elif self == TaskType.SEGMENT:
            return "polygon"
        else:
            # 未定義
            raise NotImplementedError()
        
    def __str__(self)->str:
        return self.name.lower()

class Model(Enum):
    def __repr__(self)->str:
        return self.name.replace("_","-").lower() + ".pt"
    @classmethod
    def task(cls)->TaskType:
        ...



class SegmentModel(Model):
    """物体領域"""
    # v9
    YOLOv9c_seg = auto()
    YOLOv9e_seg = auto()
    # v11
    YOLO11n_seg = auto()
    YOLO11s_seg = auto()
    YOLO11m_seg = auto()
    YOLO11l_seg = auto()
    YOLO11x_seg = auto()

    @classmethod
    def task(cls)->TaskType:
        return TaskType.SEGMENT

'''
#### Segment

| モデル                                                       | サイズ<br/>(ピクセル) | **mAPbox**<br/>50-95 | **mAPmask**<br/>50-95 | **速度**<br/>CPU ONNX (ms) | **Speed**<br/>T4 TensorRT10 (ms) | params<br>(M) | FLOPs<br>(B) |
| ------------------------------------------------------------ | --------------------- | -------------------- | --------------------- | -------------------------- | -------------------------------- | ------------- | ------------ |
| [YOLOv9c-seg](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9c-seg.pt) | 640                   | 52.4                 | 42.2                  |                            |                                  | 27.9          | 159.4        |
| [YOLOv9e-seg](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov9e-seg.pt) | 640                   | 55.1                 | 44.3                  |                            |                                  | 60.5          | 248.4        |
| [YOLO11n-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-seg.pt) | 640                   | 38.9                 | 32                    | 65.9  ± 1.1                | 1.8  ± 0.0                       | 2.9           | 10.4         |
| [YOLO11s-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s-seg.pt) | 640                   | 46.6                 | 37.8                  | 117.6  ± 4.9               | 2.9  ± 0.0                       | 10.1          | 35.5         |
| [YOLO11m-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11m-seg.pt) | 640                   | 51.5                 | 41.5                  | 281.6  ± 1.2               | 6.3  ± 0.1                       | 22.4          | 123.3        |
| [YOLO11l-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l-seg.pt) | 640                   | 53.4                 | 42.9                  | 344.2  ± 3.2               | 7.8  ± 0.2                       | 27.6          | 142.2        |
| [YOLO11x-seg](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-seg.pt) | 640                   | 54.7                 | 43.8                  | 664.5  ± 3.2               | 15.8  ± 0.7                      | 62.1          | 319          |
'''




T = TypeVar("T", bound=Model)
def __load_yolo(yolo:T, limit_type:Type[T]=SegmentModel)->YOLO:
    """
    yoloのモデルのロード
    - この関数を直接呼ばないこと
    """
    if isinstance(yolo, limit_type):
        if yolo.task == limit_type.task: # 常にTrue(Modelクラスを追加した際のミスをチェック)
            return YOLO(repr(yolo),task=str(yolo.task))
    message = f"{limit_type.__name__}を選択してください:{repr(yolo)}"
    logger.error(message)
    raise ValueError(message)



@singledispatch
def load_seg(yolo)->YOLO:
    """segment yoloのロード"""
    raise NotImplementedError(f"yolo:{type(yolo)}")

@load_seg.register
def _(yolo:SegmentModel)->YOLO:
    return __load_yolo(yolo)

@load_seg.register
def _(yolo:Path)->YOLO:
    if yolo.is_file():
        return YOLO(yolo, task=str(TaskType.SEGMENT))
    else:
        message = f"ファイルが見つかりません:{yolo.absolute().as_posix()}"
        logger.error(message)
        raise FileNotFoundError(message)

@load_seg.register
def _(yolo:str)->YOLO:
    if Path(yolo).is_file():
        return load_seg(Path(yolo))
    else:
        return YOLO(yolo, task=str(TaskType.SEGMENT))





class DNYOLOCR:
    def __init__(self, 
                 model:SegmentModel|Path|str|None=None,
                 extraction:ExtractionProtocol[uResults, uResults] = MaximumProbClasses,
    ) -> None:
        if model is None:
            default_model = Path(__file__).parent / "weights/yolo11n_seg_c300.pt"
            self.yolo:YOLO = YOLO(default_model, task=str(TaskType.SEGMENT))
        else:
            self.yolo:YOLO = load_seg(model)
        
        self.extraction:ExtractionProtocol[uResults, uResults] = extraction
    
    def predict(
        self,
        source: str|Path|int|Image.Image|list|tuple|np.ndarray|torch.Tensor,
        stream: bool = False,
        predictor=None,
        # make OnScreenNumbers object
        screen_class:int=0,
        number_classes:Sequence[int] | None=None,
        # screen coordinates
        adjust_screen_aspect_ratio: float | None = 4/3,
        mode: Literal['expand', 'shrink', 'auto'] = "auto",
        margin: int = 0,
        # OCR
        number_recognition_fun = number.NumberRecognition.readnum,
        with_preprocess: bool = True,
        **yolo_kwargs,
    ) -> list[Result]:

        # 領域検出
        screen_segments:list[uResults] = self.yolo.predict(source, stream, predictor, **yolo_kwargs)

        total = len(source) if isinstance(source,(list,tuple)) else None

        results:list[Result] = []

        with tqdm(total = total) as pbar:
            for screen_segment in screen_segments:
                ocr_results:list[tuple[OnScreenNumbers[Rectangle2D], OnScreenNumbersOCR[Rectangle2D]]] = []
                try:
                    # 結果のフィルタリング
                    extract_classes = self.extraction.extract(screen_segment)

                    # OCR
                    for ex_classes in extract_classes:
                        try:
                            screen_numbers:OnScreenNumbers[Rectangle2D] = OnScreenNumbers.from_ultralytics_result(ex_classes, screen_class, number_classes)
                            ocr_screen:OnScreenNumbersOCR[Rectangle2D] = (
                                screen_numbers
                                .get_screen_coordinates(adjust_screen_aspect_ratio, mode, margin)
                                .read_number(number_recognition_fun,with_preprocess)
                            )
                            ocr_results.append((screen_numbers, ocr_screen))
                        except Exception as e:
                            continue

                except Exception as e:
                    # logger.error(e)
                    continue
                finally:
                    m_result = Result(ocr_results)
                    m_result.org_image = screen_segment.orig_img.copy()
                    results.append(m_result)

                    pbar.update()
        
        torch.cuda.empty_cache()
        gc.collect()
        return results


    def predict_video(
            self,
            video_path:Path|str, 
            save_folder:Path|str|None=None,
            # predict settings
            screen_class:int=0,
            number_classes:Sequence[int] | None=None,
            # screen coordinates
            adjust_screen_aspect_ratio: float | None = 4/3,
            mode: Literal['expand', 'shrink', 'auto'] = "auto",
            margin: int = 0,
            # OCR
            number_recognition_fun = number.NumberRecognition.readnum,
            with_preprocess: bool = True,
            # draw settings
            target: Literal['yolo', 'ocr', 'total'] = "total",
            draw_kwargs:dict[str,Any]={}
    )->list[Result]:
        
        if not Path(video_path).is_file():
            message = f"ファイルが存在しない：{video_path}"
            logger.error(message)
            raise FileNotFoundError(message)

        if save_folder is not None:
            if not Path(save_folder).is_dir():
                Path(save_folder).mkdir(parents=True, exist_ok=True)

        # フレーム読み込み
        fps = 30
        with video.VideoFrameGenerator(str(video_path)) as gen:
            frames = [frame for frame in gen.get_frame_sequential()]
            try:
                fps = gen.fps
            except:
                pass
        
        # dnyolocr predict
        results:list[Result] = self.predict(
            frames, save=False, imgsz=640, stream=True,
            screen_class=screen_class,number_classes=number_classes,adjust_screen_aspect_ratio=adjust_screen_aspect_ratio,
            mode=mode,margin=margin,number_recognition_fun=number_recognition_fun,with_preprocess=with_preprocess
            )

        # 動画を保存
        if save_folder is not None:
            try:
                # draw
                images = [result.draw(target=target, to_rgb=False,**draw_kwargs) for result in results]
                # to video
                video.to_video(images, fps, Path(save_folder) / (Path(video_path).name))
            except Exception as e:
                logger.error(e)

        return results
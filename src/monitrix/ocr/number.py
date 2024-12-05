# Copyright (C) 2024 Arisaka Naoya
# SPDX-License-Identifier: AGPL-3.0-or-later

import re
from dataclasses import dataclass
from typing import Callable, Any


import easyocr
import cv2

from numpy import uint8, array
import numpy.typing as npt


from monitrix.core.plane_geometry import Rectangle2D
from monitrix.core.mask import Masks, BBox
from monitrix.utils.logger import get_logger

logger = get_logger("monitrix.ocr")



@dataclass(frozen=True, slots=True)
class NBox(BBox):
    """NumBoxの結果データクラス"""
    value:float|None

    def __str__(self)->str:
        # return f"{self.value} p{self.prob:.2f}"
        return f"{self.value}"

    def __repr__(self) -> str:
        return f"[text='{self.pred}', prob={self.prob:.2%}, value={self.value}]"


class BBoxes(Masks[NBox]):
    ...


class NumberRecognition:
    """https://www.jaided.ai/easyocr/documentation/"""
    reader:easyocr.Reader = easyocr.Reader(['en'])

    # General Parameters
    beamWidth:int = 1 
    allowlist:str = "0123456789+-." # Force EasyOCR to recognize only subset of characters. Useful for specific problem (E.g. license plate, etc.)
    min_size:int = 10 # Filter text box smaller than minimum value in pixel
    rotation_info:list[int] = [0] # Allow EasyOCR to rotate each text box and return the one with the best confident score. Eligible values are 90, 180 and 270. For example, try [90, 180 ,270] for all possible text orientations.
    
    # Contrast Parameters
    contrast_ths:float = 0.3
    adjust_contrast:float = 0.5

    # Text Detection (from CRAFT) Parameters
    # mag_ratio:float = 1 # Image magnification ratio

    # Bounding Box Merging Parameters


    # Other（独自のパラメータ）
    min_text_box_height_size_rate:float = 0.2 # 画像に対するテキストボックスのサイズ
    min_image_size:tuple[int,int] = (100, 300)#(50,150) # 最低画像サイズ（mag_ratio算出用）(height, width)
    max_mag_ratio:float = 3 # Image magnification ratio

    number_pattern:re.Pattern = re.compile(r"^[-+]?[0-9]+\.?([0-9]+)?$")
    search_number_pattern:re.Pattern = re.compile(r"[-+]?[0-9]+\.?([0-9]+)?")

    sorted_key:Callable[[NBox],Any] = lambda x: (x.polygon.height, x.prob)
    sorted_reverse:bool = True

    @classmethod
    def readnum(cls,
                image:npt.NDArray[uint8],
                with_preprocess:bool = True
    # )->NResult:
    )->BBoxes:
        """数値の認識"""
        
        height, width = cls.min_image_size
        _height,_width = image.shape[:2]
        mag_ratio:float = min(width / _width, height / _height, cls.max_mag_ratio)

        _image = cls.preprocess(image) if with_preprocess else image

        _min_size = max(cls.min_size, int(_height*cls.min_text_box_height_size_rate))

        results = cls.reader.readtext(
            _image,
            # general_parameter
            beamWidth=cls.beamWidth, min_size=_min_size,
            paragraph = False,
            allowlist=cls.allowlist, rotation_info = cls.rotation_info,
            # contrast_parameter
            contrast_ths=cls.contrast_ths, adjust_contrast=cls.adjust_contrast,
            # text_detection_from_CRAFT_parameter
            mag_ratio=mag_ratio,
            # bounding_box_merging_parameter
        )

        num_box:list[NBox] = []
        for res in results:
            try:
                bbox, text, prob = res
                value = cls.to_float(text)
                num_box.append(NBox(polygon=Rectangle2D(bbox),pred=text,prob=prob,value=value))
            except Exception as e:
                error_message = f"{e}: {res}"
                logger.info(error_message)
                continue
        return BBoxes(sorted(num_box, key=cls.sorted_key, reverse=cls.sorted_reverse))
    
    @classmethod
    def is_number(cls, text:str)->bool:
        """数値かどうか"""
        return cls.number_pattern.match(text) is not None

    @classmethod
    def search(cls, text:str)->str|None:
        """数値を探す"""
        match_ = cls.search_number_pattern.search(text)
        if match_ is None:
            return None
        else:
            return match_.group()

    @classmethod
    def to_float(cls, text:str)->float:
        """数値に変換"""
        if cls.is_number(text):
            return float(text)
        else:
            if (search_text := cls.search(text)) is not None:  
                message = f"`{text}` is not a number, but converted to a number `{search_text}`"
                logger.info(message)
                # warnings.warn(message,UserWarning)
                return float(search_text)
            else:
                erroe_message = f"`{text}` cannot be converted to float."
                logger.info(erroe_message)
                raise ValueError(erroe_message)

    @classmethod
    def preprocess(cls, image:npt.NDArray[uint8])->npt.NDArray[uint8]:
        # グレースケール変換
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        # ノイズ除去
        denoised = cv2.fastNlMeansDenoising(gray)
            
        # コントラスト強調
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)

        return array(enhanced, dtype=uint8)



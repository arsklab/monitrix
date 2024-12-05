# Copyright (C) 2024 Arisaka Naoya
# SPDX-License-Identifier: AGPL-3.0-or-later

from dataclasses import dataclass, asdict
from functools import cached_property
from typing import Literal, Any

import cv2
import numpy as np
from numpy import uint8
from numpy import typing as npt

from monitrix.core.plane_geometry import Rectangle2D
from monitrix.core.main_screen import OnScreenNumbers, OnScreenNumbersOCR, pDist, mpDists
from monitrix.utils import plotting
from monitrix.utils.logger import get_logger

logger = get_logger("monitrix.core")




@dataclass(frozen=True)
class recode(mpDists):
    screen_id:int

class Result(list[tuple[OnScreenNumbers[Rectangle2D], OnScreenNumbersOCR[Rectangle2D]]]):
    """一枚の画像に対する YOLO + OCR 結果クラス(1つのタプルが一つのスクリーン)"""

    @property
    def org_image(self)->npt.NDArray[uint8]:
        try:
            return self.__org_image
        except:
            if not self.is_empty:
                return self[0][0].orig_img.copy()
            else:
                message = "オリジナル画像を取得出来なかった"
                logger.info(message)
                raise IndexError(message)

    @org_image.setter
    def org_image(self, value:npt.NDArray[uint8]):
        self.__org_image = value

    @cached_property
    def is_empty(self)->bool:
        """空かどうか"""
        return len(self) == 0

    #@cached_property
    def values(self,state:Literal["all","best"]="best")->list[dict[pDist,dict[pDist,list[pDist]]]]:
        """
        key: yoloによる推定されたクラス
        value: (yoloによる推定されたクラス確率, ocrの推定確率, ocr推定結果)
        """
        return [ocr_screen.to_dict(state) for _, ocr_screen in self]
    #@cached_property
    def recodes(self, state:Literal["all","best"]="best")->list[recode]:
        _l = []
        for s_id,(_, ocr_screen) in enumerate(self):
            for v in ocr_screen.to_list(state):
                _l.append(recode(**asdict(v),screen_id=s_id))
        return _l

    def draw(
            self, 
            target:Literal["yolo","ocr","total"] = "ocr",
            screen_numbers_margin: int = 100,
            screen_numbers_draw_kwargs:dict[str, Any] = {},
            ocr_screen_draw_kwargs:dict[str, Any] = {},
            to_rgb:bool = True,
            resize:float=1.

    )->npt.NDArray[uint8]:
        """結果の描画"""
        
        def bgr_to_rgb(_image:npt.NDArray[uint8])->npt.NDArray[uint8]:
            return np.array(cv2.cvtColor(_image, cv2.COLOR_BGR2RGB), dtype=uint8)
        
        if target == "yolo":
            """yoloの検出結果"""
            org_image = self.org_image
            height, width = org_image.shape[:2]
            x1, y1, y2, x2 = 0, 0, height, width
        
            for screen_numbers, _ in self:
                try:
                    org_image = screen_numbers.draw_screen(org_image, **screen_numbers_draw_kwargs)
                    if screen_numbers.screen is not None:
                        _x1, _y2, _x2, _y1 = screen_numbers.screen.bbox.xyxy.astype(int)
                        if _y1 > _y2:
                            _y1, _y2 = _y2, _y1
                        _x1, _x2, _y1, _y2 = max(0, _x1 - screen_numbers_margin),min(width, _x2 + screen_numbers_margin),max(0, _y1 - screen_numbers_margin),min(height, _y2 + screen_numbers_margin)
                        # 妥当性のチェック
                        if _x1 < _x2 and _y1 < _y2:
                            x1, y1, y2, x2 = max(x1,_x1),max(y1,_y1),min(y2,_y2),min(x2,_x2)
                except:
                    continue
            if to_rgb:
                return bgr_to_rgb(org_image[y1:y2, x1:x2])
            else:
                return org_image[y1:y2, x1:x2]
        
        elif target == "ocr":
            """スクリーンごとのOCR結果"""
            images = []
            for _, ocr_screen in self:
                try:
                    images.append(ocr_screen.draw_screen(**ocr_screen_draw_kwargs))
                except:
                    continue
            #images = [ocr_screen.draw_screen(**ocr_screen_draw_kwargs) for _, ocr_screen in self]
            if to_rgb:
                return bgr_to_rgb(plotting.create_image_grid(images))
            else:
                return plotting.create_image_grid(images)

        
        elif target == "total":
            """まとめ"""

            org_image = self.org_image
            height , width = org_image.shape[:2]
            try:
                proccess1 = self.draw("yolo",
                                    screen_numbers_margin=screen_numbers_margin,
                                    screen_numbers_draw_kwargs=screen_numbers_draw_kwargs,to_rgb=False)
            except:
                proccess1 = None

            try:
                proccess2 = self.draw("ocr",
                                      ocr_screen_draw_kwargs=ocr_screen_draw_kwargs,to_rgb=False)
            except :
                proccess2 = None
            
            result_image:npt.NDArray[uint8]
            if height > width:
                target_size = (height//2, width)

                proccess1 = plotting.letterbox(proccess1, target_size)
                proccess2 = plotting.letterbox(proccess2, target_size)
                p_images = (proccess1, proccess2)
                result_image = np.hstack((org_image,np.vstack(p_images)))
            else:
                target_size = (height//2, width)
                proccess1 = plotting.letterbox(proccess1, target_size)
                proccess2 = plotting.letterbox(proccess2, target_size)
                p_images = (proccess1, proccess2)
                
                result_image = np.vstack((org_image, np.hstack(p_images)))

            if to_rgb:
                result_image = bgr_to_rgb(result_image)

            if resize != 1.:
                result_image = np.array(cv2.resize(result_image,None,None,fx=resize,fy=resize),dtype=uint8)

            return result_image


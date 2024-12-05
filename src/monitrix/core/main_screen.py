# Copyright (C) 2024 Arisaka Naoya
# SPDX-License-Identifier: AGPL-3.0-or-later

from dataclasses import dataclass, asdict
from typing import Sequence, TypeVar, Generic, Literal, Callable
from typing import Protocol, Generic, TypeVar
from collections import defaultdict

from ultralytics.engine.results import Results
from numpy import uint8, int32
from numpy import typing as npt
import numpy as np
import cv2

from monitrix.core.plane_geometry import Polygon2D, Rectangle2D, Point2D
from monitrix.core.mask import Mask,Masks
from monitrix.utils import plotting
from monitrix.ocr import number
from monitrix.utils.logger import get_logger

logger = get_logger("monitrix.core")

T_contra = TypeVar("T_contra", contravariant=True)
R = TypeVar("R")

class ExtractionProtocol(Protocol, Generic[T_contra,R]):
    """Resultsから必要な結果を取り出すクラスのプロトコル"""
    @classmethod
    def extract(cls, result:T_contra)->list[R]:
        """取り出し実行
        画面に複数写っている場合も考慮して出力は`list[R]`とした
        """
        ...

class MaximumProbClasses(ExtractionProtocol[Results, Results]):
    """各クラスごとの、予測確率が最大値の結果を抽出"""
    @classmethod
    def extract(cls, result):
        """各クラスごとの、予測確率が最大値の結果を抽出"""
    
        u_result = result.cpu()
        if (bbox := u_result.boxes) is None:
            raise ValueError("Results.boxes do not exist.")

        bbox = bbox.numpy()
        # 予測クラス
        predicted_classes = bbox.cls
        # 予測確率
        predicted_probabilities = bbox.conf

        # 各クラスごとの、予測確率が最大値のインデックス取得
        sorted_indies = np.lexsort((predicted_probabilities, predicted_classes))
        mask = np.diff(predicted_classes[sorted_indies], append=predicted_classes.max()+1).astype(bool)
        maximum_prob_indies = sorted_indies[mask]

        return [u_result[maximum_prob_indies]]
    


P = TypeVar("P", bound=Polygon2D, contravariant=False)

@dataclass
class OnScreenNumbers(Generic[P]):
    """メインスクリーンと数値領域

    1つのメインスクリーンに複数の数値領域がある状態を想定
    """
    screen:Mask[P]|None
    numbers:Masks[Mask[P]]
    orig_img:npt.NDArray[uint8]

    def perspective_transform(
            self,
            adjust_screen_aspect_ratio:float|None=None,
            mode:Literal["expand","shrink","auto"]="expand"
    )->"OnScreenNumbers[P]":
        """
        透視投影（スクリーン）後のOnScreenNumbersを生成する

        Args:
            adjust_screen_aspect_ratio: 目標のアスペクト比 (width / height)
            mode: 調整モード
                - 'expand': 小さい方の辺を広げてアスペクト比を調整
                - 'shrink': 大きい方の辺を縮めてアスペクト比を調整
                - 'auto': 元の面積により近くなる方を選択
        """
        if self.screen is not None:
            screen_tetragon = self.screen.polygon.approx_tetragon
            
            screen_rectangle:Rectangle2D = screen_tetragon.bounding_rectangle

            # アスペクト比の修正
            if adjust_screen_aspect_ratio:
                screen_rectangle = screen_rectangle.adjust_bbox_aspect_ratio(adjust_screen_aspect_ratio,mode)
            
            # 変換行列
            _matrix = screen_tetragon.get_perspective_transform(screen_rectangle)

            # 透視変換を画像に適用
            p_img:npt.NDArray = cv2.warpPerspective(
                    src = self.orig_img, 
                    M = _matrix, 
                    dsize = self.orig_img.shape[:-1][::-1],
                    flags=cv2.INTER_LANCZOS4)
    
            return OnScreenNumbers(
                screen = self.screen.transform_points(_matrix),
                numbers = Masks(num.transform_points(_matrix) for num in self.numbers),
                orig_img = p_img
            )
        else:
            error_message = "スクリーンが存在しない"
            logger.error(error_message)
            raise ValueError(error_message)

    def shift(self, origin:Point2D, image:npt.NDArray[np.uint8])->"OnScreenNumbers[P]":
        """平行移動"""
        return OnScreenNumbers(
                screen = self.screen.shift(origin) if self.screen else None,
                numbers = self.numbers.shift(origin),
                orig_img = image
            )

    def slice_image(self,margin:int=0)->npt.NDArray[uint8]:
        """画像の切り出し"""
        if self.screen is not None:
            x1, y2, x2, y1 = abs(self.screen.bbox.xyxy.astype(int32))
            return self.orig_img[y1-margin:y2+margin, x1-margin:x2+margin]#.copy()
        else:
            error_message = "スクリーンが存在しない"
            logger.error(error_message)
            raise ValueError(error_message)
    
    def draw_screen(self, 
                    image:npt.NDArray[np.uint8]|None=None,
                    cmap:str="tab10", 
                    alpha=0.3, 
                    polygon_thickness=2, 
                    polygon_vertex_size=8,
                    tetragon_color:tuple[int,int,int]|None=None, 
                    tetragon_vertex_rate:int = 110,
                    with_numbers:bool = True
    )->npt.NDArray[uint8]:
        """スクリーンの描画"""

        _image:npt.NDArray[np.uint8] = self.orig_img.copy() if image is None else image.copy()

        color_gen = plotting.Color(cmap)
        _color = color_gen.get_next_color()
        
        if self.screen is not None:
            _image = self.screen.draw_polygon(_image, _color, alpha, polygon_thickness, polygon_vertex_size)
            _image = self.screen.draw_box(_image, str(self.screen.pred),color=_color)
            t_color = tetragon_color if tetragon_color else _color
            _image = self.screen.draw_approx_tetragon(_image, color=t_color, alpha=0, thickness=0,vertex_rate=tetragon_vertex_rate)

        if with_numbers:
            for num in self.numbers:
                _color = color_gen.get_next_color()
                _image = num.draw_polygon(_image, _color, alpha, polygon_thickness, polygon_vertex_size)
                _image = num.draw_box(_image, str(num.pred), loc="right",color=_color)
                t_color = tetragon_color if tetragon_color else _color
                _image = num.draw_approx_tetragon(_image, color=t_color, alpha=0, thickness=0,vertex_rate=tetragon_vertex_rate)

        return _image
    

    def get_screen_coordinates(self,
                               adjust_screen_aspect_ratio:float|None=None,
                               mode:Literal["expand","shrink","auto"]="auto",
                               margin:int=0
    )->"OnScreenNumbers[Rectangle2D]":
        """スクリーン取り出し"""

        # 透視投影変換
        _per_screen = self.perspective_transform(adjust_screen_aspect_ratio, mode)

        if _per_screen.screen is None:
            error_message = "スクリーンが存在しない"
            logger.error(error_message)
            raise ValueError(error_message)

        # スクリーン画像
        screen_image = _per_screen.slice_image(margin=margin)

        # 座標変換（原点の平行移動）
        screen_coord = OnScreenNumbers[Rectangle2D](
                screen = _per_screen.screen.bounding_rectangle,
                numbers = Masks(num.bounding_rectangle for num in _per_screen.numbers),
                orig_img = screen_image
            )

        if screen_coord.screen is None:
            error_message = "スクリーンが存在しない"
            logger.error(error_message)
            raise ValueError(error_message)

        # 座標変換（原点の平行移動）
        origin:Point2D = Point2D(screen_coord.screen.bbox.leftbottom - margin)
        
        # screen_coord.screen.polygon = Rectangle2D(screen_coord.screen.polygon - origin)
        # for num in screen_coord.numbers:
        #     num.polygon = Rectangle2D(num.polygon - origin)

        return screen_coord.shift(origin, screen_image)


    @classmethod
    def from_ultralytics_result(cls, 
                                ultralytics_result:Results, 
                                screen_class:int=0, 
                                number_classes:Sequence[int]|None=None,
    )->"OnScreenNumbers":
        """YOLOの予測結果からデータを取得"""

        u_result = ultralytics_result.cpu()
        if (bbox := u_result.boxes) is None:
            error_message = "Results.boxes do not exist."
            logger.error(error_message)
            raise ValueError(error_message)
        if (masks := u_result.masks) is None:
            error_message = "Results.masks do not exist."
            logger.error(error_message)
            raise ValueError(error_message)
        # クラス名
        names:dict[int,str] = u_result.names

        bbox_np = bbox.numpy()
        # 予測クラス
        predicted_classes:np.ndarray = np.array(bbox_np.cls).astype(int)

        if not (np.sum(predicted_classes == screen_class) <= 1):
            error_message = f"Maximum number of screens is 1: {len(np.sum(predicted_classes == screen_class))}"
            logger.error(error_message)
            raise ValueError(error_message)

        # 予測確率
        predicted_probabilities:np.ndarray = np.array(bbox_np.conf)
        # 境界
        predicted_polygon:list[np.ndarray] = [np.array(xy) for xy in masks.xy]

        _screen:Mask|None = None
        _numbers:list[Mask] = []
        for p_polygon, p_prob, p_class in zip(predicted_polygon, predicted_probabilities, predicted_classes):
            
            mask = Mask(Polygon2D(p_polygon), p_prob, names[p_class])
            if p_class == screen_class:
                _screen = mask
            elif number_classes is not None:
                if p_class in number_classes:
                    _numbers.append(mask)
            else:
                _numbers.append(mask)

        return OnScreenNumbers(_screen, Masks(_numbers), orig_img = u_result.orig_img)


    def read_number(self, 
                    number_recognition_func:Callable[[npt.NDArray[uint8],bool], number.BBoxes] = number.NumberRecognition.readnum,
                    with_preprocess:bool = True
    )->"OnScreenNumbersOCR[P]":
        """数値の認識"""
        orig_img = self.orig_img.copy()
        ocr_result = []
        for num in self.numbers:
            _image = num.slice_image(orig_img)

            # OCR実行
            num_results = number_recognition_func(_image, with_preprocess)

            # 座標移動
            origin = Point2D(-num.bbox.leftbottom)
            num_results_shift = [n.shift(origin) for n in num_results]

            ocr_result.append(OCRMask(**asdict(num), ocrs=number.BBoxes(num_results_shift)))
        
        return OnScreenNumbersOCR[P](screen=self.screen,numbers=OCRMasks(ocr_result),orig_img=self.orig_img.copy())


@dataclass(frozen=True)
class OCRMask(Mask[P]):
    ocrs:number.BBoxes

class OCRMasks(Masks[OCRMask[P]]):
    # def shift(self, origin:Point2D)->"Masks[P]":
    #     """座標の平行移動"""
    #     return Masks(m.shift(origin) for m in self)
    ...

@dataclass(frozen=True)
class pDist:
    """予測と確率のペア(予測分布;predictive distribution)"""
    label:str|float|int
    prob:float

    def __repr__(self) -> str:
        return f"{self.label}({self.prob:.2f})"


@dataclass(frozen=True)
class mpDists:
    """"""
    screen:str|float|int
    screen_prob:float
    label:str|float|int
    label_prob:float
    number:float
    number_prob:float

    def __repr__(self) -> str:
        return f"{self.screen}>{self.label}:{self.number}(p={self.number_prob:.2f})"

@dataclass
class OnScreenNumbersOCR(OnScreenNumbers[P]):

    numbers:OCRMasks[P]

    def draw_screen(self, 
                    image:npt.NDArray[np.uint8]|None=None,
                    cmap:str="tab10", 
                    alpha=0.3, 
                    polygon_thickness=2, 
                    polygon_vertex_size=8,
                    tetragon_color:tuple[int,int,int]|None=None, 
                    tetragon_vertex_rate:int = 110,
                    with_numbers:bool = True,

                    representative:bool = True, # ocrの先頭のboxesみ描画
                    num_bbox_color:tuple[int,int,int] = (255,0,0)

    )->npt.NDArray[uint8]:
        """スクリーンの描画"""
        _image = super().draw_screen(image,
                    cmap, 
                    alpha, 
                    polygon_thickness, 
                    polygon_vertex_size,
                    tetragon_color, 
                    tetragon_vertex_rate,
                    with_numbers)
        
        if with_numbers:
            for num in self.numbers:
                for nbox in num.ocrs:
                    _image = nbox.draw_box(_image, None, None, num_bbox_color, "top")
                    if representative:
                        break
        return _image
    
    def to_dict(self, state:Literal["all","best"]="best")->dict[pDist,dict[pDist,list[pDist]]]:
        """予測分布のみを辞書にまとめる"""
        if self.screen is not None:
            screen, screen_prob = self.screen.pred, self.screen.prob
        else:
            screen, screen_prob = "", -1

        _d:dict[pDist,list[pDist]] = defaultdict(list)
        for num in self.numbers:
            label,label_prob = num.pred, num.prob
            if state == "all":
                for ocr in num.ocrs:
                    if ocr.value is not None:
                        _d[pDist(label,label_prob)].append(pDist(ocr.value, ocr.prob))
            elif state == "best":
                if len(num.ocrs) != 0:
                    ocr = num.ocrs[0]
                    if ocr.value is not None:
                        _d[pDist(label,label_prob)].append(pDist(ocr.value, ocr.prob))
        return {pDist(screen, screen_prob):dict(_d)}
    
    def to_list(self, state:Literal["all","best"]="best")->list[mpDists]:
        """予測分布をリストにまとめる"""
        if self.screen is not None:
            screen, screen_prob = self.screen.pred, self.screen.prob
        else:
            screen, screen_prob = "", -1

        _l:list[mpDists] = []
        for num in self.numbers:
            label,label_prob = num.pred, num.prob
            if state == "all":
                for ocr in num.ocrs:
                    if ocr.value is not None:
                        _l.append(mpDists(screen, screen_prob, label,label_prob, ocr.value, ocr.prob))
            elif state == "best":
                if len(num.ocrs) != 0:
                    ocr = num.ocrs[0]
                    if ocr.value is not None:
                        _l.append(mpDists(screen, screen_prob, label,label_prob, ocr.value, ocr.prob))
        return _l






# ## 結果の描画
# def get_screen_around_image(
#         screen_numbers: OnScreenNumbers|None, 
#         margin: int = 100,
#         tetragon_color: tuple[int, int, int] = (255, 0, 0)
# ) -> npt.NDArray[uint8] | None:
#     try:
#         if screen_numbers is None:
#             return None
#         screen_image = screen_numbers.draw_screen(tetragon_color=tetragon_color, tetragon_vertex_rate=140)

#         if screen_numbers.screen is None:
#             return None
#         x1, y2, x2, y1 = screen_numbers.screen.bbox.xyxy.astype(int)
        
#         # y1とy2の順序を修正（y1が小さい値になるようにする）
#         if y1 > y2:
#             y1, y2 = y2, y1
            
#         # 幅と高さを計算
#         width = x2 - x1
#         height = y2 - y1
        
#         # 範囲の妥当性チェック
#         if width <= 0 or height <= 0:
#             # print("Invalid width or height")
#             return None
            
#         # マージン付きの切り出し範囲を計算
#         start_y = max(0, y1 - margin)
#         end_y = min(screen_image.shape[0], y2 + margin)
#         start_x = max(0, x1 - margin)
#         end_x = min(screen_image.shape[1], x2 + margin)
        
#         # 切り出し範囲の妥当性チェック
#         if start_y >= end_y or start_x >= end_x:
#             return None
            
#         # 画像の切り出し
#         screen_image = screen_image[start_y:end_y, start_x:end_x]
        
#         return screen_image
        
#     except Exception as e:
#         print(f"Error in get_screen_around_image: {e}")
#         return None
    

# def show_result(
#         org_image:npt.NDArray[uint8],
#         screen_numbers:OnScreenNumbers|None,
#         ocr_screen:OnScreenNumbersOCR|None,
#         margin:int = 200,
#         to_rgb:bool=True,
#         resize:float=1.
# )->npt.NDArray[uint8]:

#     org_image = org_image.copy()
#     height , width = org_image.shape[:2]

#     process1 = get_screen_around_image(screen_numbers, tetragon_color=(0,0,255), margin=margin)

#     try:
#         process2 = ocr_screen.draw_screen()

#     except:
#         process2 = None

#     result_image:npt.NDArray[uint8]
#     if height > width:
#         target_size = (height//2, width)

#         process1 = plting.letterbox(process1, target_size)
#         process2 = plting.letterbox(process2, target_size)
#         p_images = (process1, process2)
#         result_image = np.hstack((org_image,np.vstack(p_images)))
#     else:
#         target_size = (height//2, width)
#         process1 = plting.letterbox(process1, target_size)
#         process2 = plting.letterbox(process2, target_size)
#         p_images = (process1, process2)
        
#         result_image = np.vstack((org_image, np.hstack(p_images)))

#     if to_rgb:
#         result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

#     if resize != 1.:
#         result_image = cv2.resize(result_image,None,None,fx=resize,fy=resize)

#     return result_image
# Copyright (C) 2024 Arisaka Naoya
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Generic, TypeVar, Literal, Any
from functools import cached_property
from dataclasses import dataclass, asdict

from numpy import typing as npt
from numpy import uint8, int32

from monitrix.core.plane_geometry import Polygon2D, Tetragon2D, Rectangle2D, Point2D
import monitrix.utils.plotting as plting


P = TypeVar("P", bound=Polygon2D, contravariant=False)
M = TypeVar("M", bound="Mask[Any]", contravariant=False)
R = TypeVar("R", bound="Masks[Any]", contravariant=False)

@dataclass(frozen=True, slots=False)
class Mask(Generic[P]):
    """検出領域"""
    polygon:P
    prob:float
    pred:float|int|str

    def __new_obj__(self:M, **kwargs)->M:
        _param = asdict(self)
        _param.update(**kwargs)
        return self.__class__(**_param)

    def transform_points(self:M, matrix)->M:
        """射影変換"""
        return self.__new_obj__(polygon=self.polygon.transform_points(matrix))

    @cached_property
    def approx_tetragon(self:M)->M:
        """凸四角形"""
        return self.__new_obj__(polygon=self.polygon.approx_tetragon)
    
    @cached_property
    def bounding_rectangle(self:M)->M:
        """傾いていない外接する矩形領域"""
        return self.__new_obj__(polygon=self.bbox)

    @cached_property
    def bbox(self)->Rectangle2D:
        """傾いていない外接する矩形領域の座標"""
        if isinstance(self.polygon, Polygon2D):
            return self.polygon.approx_tetragon.bounding_rectangle
        elif isinstance(self.polygon, Tetragon2D):
            return self.polygon.bounding_rectangle
        else:
            return self.polygon

    def shift(self:M, origin:Point2D)->M:
        """座標の平行移動"""
        return self.__new_obj__(polygon=self.polygon.shift(origin))


    def slice_image(self,image:npt.NDArray[uint8])->npt.NDArray[uint8]:
        """画像の切り出し"""
        x1, y2, x2, y1 = self.bbox.xyxy.astype(int32)
        return image[y1:y2, x1:x2]

    def draw_box(self,
             image:npt.NDArray[uint8],
             label:str|None,
             line_width:int|None = None,
             color:tuple[int,int,int]|plting.Color = (0,0,255),
             loc:Literal["top", "right", "righttop"] = "top"
    )->npt.NDArray[uint8]:
        """
        BBoxの描画
        color: 描画色 (B,G,R)
        """
        try:
            x1, y2, x2, y1 = self.bbox.xyxy.astype(int)
            return plting.box_label(image, x1,y1,x2,y2, text=label, line_width=line_width, color=color,loc=loc)
        except:
            return image

    def draw_polygon(self,
                     image: npt.NDArray[uint8], 
                     color: tuple[int,int,int]|plting.Color = (0,0,255), 
                     alpha: float = 0.3, 
                     thickness: int = 2, 
                     vertex_size: int = 8
    )->npt.NDArray[uint8]:
        """
        ポリゴンの描画
        color: 描画色 (B,G,R)
        """
        try:
            return plting.draw_polygon(image,self.polygon,color,alpha,thickness,vertex_size)
        except:
            return image
    
    def draw_approx_tetragon(self,
                     image: npt.NDArray[uint8], 
                     color: tuple[int,int,int]|plting.Color = (0,0,255), 
                     alpha: float = 0.3, 
                     thickness: int = 2, 
                     vertex_size: int|None = None,
                     vertex_rate:int = 120,
    )->npt.NDArray[uint8]:
        """
        凸四角形の描画
        color: 描画色 (B,G,R)
        """
        try:
            _vertex_size:int = vertex_size if vertex_size else int(max(image.shape) / vertex_rate) 
            return plting.draw_polygon(image,self.approx_tetragon.polygon,color,alpha,thickness,_vertex_size)
        except:
            return image




class Masks(Generic[M], list[M]):
    def transform_points(self:R, matrix)->R:
        """射影変換"""
        return self.__class__(m.transform_points(matrix) for m in self)
    @property
    def approx_tetragon(self:R)->R:
        """凸四角形"""
        return self.__class__(m.approx_tetragon for m in self)
    @property
    def bounding_rectangle(self:R)->R:
        """傾いていない外接する矩形領域"""
        return self.__class__(m.bounding_rectangle for m in self)

    def shift(self:R, origin:Point2D)->R:
        """座標の平行移動"""
        return self.__class__(m.shift(origin) for m in self)
    
    def draw_box(self,
            image:npt.NDArray[uint8], 
            label:str|None = None,
            line_width:int|None = None,
            color:tuple[int,int,int]|plting.Color = (0,0,255),
            loc:Literal["top", "right", 'righttop'] = "top"
    )->npt.NDArray[uint8]:
        """BBox描画"""
        for elm in self:
            try:
                elm.draw_box(image,label,line_width,color,loc)
            except:
                continue
        return image




@dataclass(frozen=True, slots=False)
class BBox(Mask[Rectangle2D]):
    """BBox検出領域"""
    polygon:Rectangle2D # BBox
    prob:float # 予測確率
    pred:float|int|str

    def __str__(self)->str:
        return f"{self.pred} p{self.prob:.2g}"

    def draw_box(self,
            image:npt.NDArray[uint8], 
            label:str|None = None,
            line_width:int|None = None,
            color:tuple[int,int,int]|plting.Color = (0,0,255),
            loc:Literal["top", "right", 'righttop'] = "top"
            )->npt.NDArray[uint8]:
        """
        bboxの描画
        color: 描画色 (B,G,R)
        """
        try:
            x1, y2, x2, y1 = self.bbox.xyxy.astype(int)
            return plting.box_label(
                image, 
                #self.bbox, 
                x1,y1,x2,y2,
                text=str(self) if label is None else label, 
                line_width=line_width, 
                color=color,
                loc=loc)
        except:
            return image

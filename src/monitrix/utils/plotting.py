# Copyright (C) 2024 Arisaka Naoya
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import Literal
import math

import cv2
import numpy.typing as npt
from numpy import uint8
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap, LinearSegmentedColormap
from matplotlib.colors import to_rgb

from monitrix.utils.logger import get_logger

logger = get_logger("monitrix.utils")

class Color:
    """色を順番に生成するクラス"""
    def __init__(self, cmap_name:str|Colormap='tab10', n_colors=None):
        """
        Parameters:
        -----------
        cmap_name : str or Colormap
            使用するカラーマップの名前またはColormapオブジェクト
        n_colors : int or None
            生成する色の数。Noneの場合は256色
        """
        if isinstance(cmap_name, str):
            self.cmap = plt.get_cmap(cmap_name)
        elif isinstance(cmap_name, Colormap):
            self.cmap = cmap_name
        else:
            error = "cmap_name must be string or Colormap object"
            logger.error(error)
            raise ValueError(error)
        
        self.n_colors = n_colors or self.cmap.N #256
        self._color_index = 0
        
        # 色のリストを事前に生成
        self.colors = [self.cmap(i / (self.n_colors - 1)) for i in range(self.n_colors)]
    
    def get_next_color(self)->tuple[int,int,int]:
        """次の色を取得
    
        Returns:
            color: (B,G,R)
        """
        if self._color_index >= self.n_colors:
            self._color_index = 0
        color = self.colors[self._color_index]
        self._color_index += 1

        rgb_c:tuple[int,int,int] = tuple(int(s*256) for s in to_rgb(color))
        return (rgb_c[2], rgb_c[1], rgb_c[0])
    
    def get_color_at(self, index):
        """指定されたインデックスの色を取得"""
        return self.colors[index % self.n_colors]
    
    def get_color_for_value(self, value, vmin=0, vmax=1)->tuple[int,int,int]:
        """
        指定された値に対応する色を取得
        value: 値
        vmin, vmax: 値の範囲

        Returns:
            color: (B,G,R)
        """
        normalized_value = (value - vmin) / (vmax - vmin)
        rgb_c:tuple[int,int,int] = tuple(int(s*256) for s in to_rgb(self.cmap(normalized_value)))
        return (rgb_c[2], rgb_c[1], rgb_c[0])

    def reset(self):
        """インデックスをリセット"""
        self._color_index = 0


    # カスタムカラーマップの作成例
    @classmethod
    def create_custom_colormap(cls,
                               colors:list[tuple[int,int,int]|str],
                               n_colors:int=10,
                               n_bins:int = 200  # 色の補間数
    ) -> "Color":
        # カスタムカラーマップの定義
        custom_cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
        
        # カスタムカラーマップを使用したジェネレータ
        return Color(custom_cmap, n_colors=n_colors)

def box_label(
            image:npt.NDArray[uint8], 
            # bbox:Rectangle2D, 
            x1:int,y1:int,x2:int,y2:int,
            text:str|None,
            line_width:int|None = None,
            color:tuple[int,int,int]|Color = (255,0,0),
            loc:Literal["top", "right", "righttop"] = "top"
)->npt.NDArray[uint8]:
    """バウンディングボックスとラベルの描画"""

    # スクリーン座標系
    #x1, y2, x2, y1 = bbox.xyxy.astype(int)
    p1, p2 = (x1,y1), (x2, y2)

    lw:int = line_width or max(round(sum(image.shape) / 2 * 0.003), 2)

    if not image.data.contiguous:
        error = "Image not contiguous. Apply np.ascontiguousarray(im) to Annotator input images."
        logger.error(error)
        raise ValueError(error)
    
    image = image if image.flags.writeable else image.copy()
    tf:int = max(lw - 1, 1)  # font thickness
    sf:float = lw / 3  # font scale


    c = color.get_next_color() if isinstance(color, Color) else color
    cv2.rectangle(image, p1, p2, c, thickness=lw, lineType=cv2.LINE_AA)

    if text:
        w, h = cv2.getTextSize(text, 0, fontScale=sf, thickness=tf)[0]  # text width, height
        h += 3  # add pixels to pad text
        outside = p1[1] >= h  # label fits outside box
        if p1[0] > image.shape[1] - w:  # shape is (h, w), check if label extend beyond right side of image
                p1 = image.shape[1] - w, p1[1]
        p2 = p1[0] + w, p1[1] - h if outside else p1[1] + h
        
        if loc == "right":
            offset = (x2-x1,y2-y1)
        elif loc == "righttop":
            offset = (x2-x1,0)
        else:
            offset = (0,0)
        cv2.rectangle(image, tuple(p+o for p,o in zip(p1,offset)), tuple(p+o for p,o in zip(p2,offset)), c, -1, cv2.LINE_AA)  # filled
        cv2.putText(image, text, (p1[0]+offset[0], (p1[1] - 2 if outside else p1[1] + h - 1)+offset[1]), 0, sf, (255,255,255), thickness=tf, lineType=cv2.LINE_AA)
    return image



def draw_polygon(
        image:npt.NDArray[uint8], 
        # points:Polygon2D|list[tuple[int,int]], 
        points:list[tuple[int,int]]|npt.NDArray[np.int32], 
        color:tuple[int,int,int]|Color, 
        alpha:float=0.3, 
        thickness:int=2, 
        vertex_size:int=8
)->npt.NDArray[uint8]:
    """
    半透明な多角形を描画する関数
    
    Parameters:
        image: 描画対象の画像
        points: 多角形の頂点座標のリスト [(x1,y1), (x2,y2), ...]
        color: 描画色 (B,G,R)
        alpha: 透明度 (0: 透明 ~ 1: 不透明)
        thickness: 線の太さ（-1の場合は塗りつぶし）
    
    Returns:
        result_image: 描画後の画像
    """
    # 入力画像のコピーを作成
    overlay = image.copy()
    
    # 頂点座標を整形
    pts = np.array(points, np.int32)
    pts = pts.reshape((-1, 1, 2))
    
    # 色
    _color = color.get_next_color() if isinstance(color, Color) else color
    
    # 塗りつぶし用の多角形を描画
    result_image:npt.NDArray[uint8]
    if alpha > 0.:
        cv2.fillPoly(overlay, [pts], _color)
        # アルファブレンディング
        result_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    else:
        result_image = overlay

    # 輪郭線を描画（thickness > 0の場合）
    if thickness > 0:
        cv2.polylines(result_image, [pts], True, _color, thickness)
    
    # 各頂点を描画
    for point in points:
        x, y = point
        # 頂点の形状に応じて描画
        cv2.circle(img=result_image, center=(int(x), int(y)), radius=vertex_size, color=_color, thickness=-1)

    return result_image





def letterbox(
        img: npt.NDArray[uint8]|None,
        target_size: tuple[int, int]
    ) -> npt.NDArray[uint8]:
    """画像をリサイズし、アスペクト比を維持しながら黒いパディングを追加
    (画像をアスペクト比を維持したままリサイズし、余白をletterboxとして追加する関数)

    Args:
        img: 入力画像（None可）
        target_size: 目標サイズ (height, width)

    Returns:
        リサイズされた画像
    """
    if img is None:
        # Noneが渡された場合は黒い画像を返す
        return np.zeros((*target_size, 3), dtype=np.uint8)

    target_h, target_w = target_size
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid target size: {target_size}")

    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid image size: ({h}, {w})")

    # アスペクト比を計算
    aspect = w / h
    target_aspect = target_w / target_h

    if aspect > target_aspect:
        # 画像が目標よりも横長の場合
        new_w = target_w
        new_h = int(target_w / aspect)
        if new_h <= 0:
            new_h = 1
    else:
        # 画像が目標よりも縦長の場合
        new_h = target_h
        new_w = int(target_h * aspect)
        if new_w <= 0:
            new_w = 1

    # リサイズ
    resized = cv2.resize(img, (new_w, new_h))

    # 黒い画像を作成
    result = np.zeros((target_h, target_w, 3), dtype=np.uint8)

    # センタリングして配置
    y_offset = (target_h - new_h) // 2
    x_offset = (target_w - new_w) // 2

    # はみ出し防止
    y_end = min(y_offset + new_h, target_h)
    x_end = min(x_offset + new_w, target_w)
    h_slice = slice(max(0, y_offset), y_end)
    w_slice = slice(max(0, x_offset), x_end)

    result[h_slice, w_slice] = resized[:y_end-y_offset, :x_end-x_offset]

    return result




def create_image_grid(
    images: list[np.ndarray], 
    padding: int = 10,
    max_size: int|None = None,
    background_color: tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    複数の画像を格子状に配置し、なるべく正方形に近い一枚の画像にまとめる
    
    Parameters:
    -----------
    images : List[np.ndarray]
        配置する画像のリスト
    padding : int
        画像間の余白のピクセル数
    max_size : Optional[int]
        出力画像の長辺の最大サイズ（ピクセル）。
        Noneの場合は元のサイズを維持
    background_color : Tuple[int, int, int]
        背景色（BGR形式）
        
    Returns:
    --------
    np.ndarray
        まとめられた画像
    """
    
    def get_optimal_grid(n: int) -> tuple[int, int]:
        """最適な行数と列数を計算"""
        rows = round(math.sqrt(n))
        cols = math.ceil(n / rows)
        return rows, cols
    
    n_images = len(images)
    if n_images == 0:
        raise ValueError()
    
    # 最適なグリッドサイズを計算
    rows, cols = get_optimal_grid(n_images)
    
    # 各セルの基本サイズを計算
    max_height = max(img.shape[0] for img in images)
    # max_width = max(img.shape[1] for img in images)
    aspect_ratios = [img.shape[1] / img.shape[0] for img in images]
    avg_aspect = sum(aspect_ratios) / len(aspect_ratios)
    
    cell_height = max_height
    cell_width = int(cell_height * avg_aspect)
    
    # 最終的な画像サイズを計算
    total_width = cols * cell_width + (cols - 1) * padding
    total_height = rows * cell_height + (rows - 1) * padding
    
    # 結果用の画像を作成
    result = np.full((total_height, total_width, 3), background_color, dtype=np.uint8)
    
    # 画像を配置
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n_images:
                break
                
            y = i * (cell_height + padding)
            x = j * (cell_width + padding)
            
            # 画像をリサイズしてレターボックス処理
            resized = letterbox(images[idx], (cell_height,cell_width))
            result[y:y+cell_height, x:x+cell_width] = resized
            
            idx += 1
    
    # 最終的なサイズ調整
    if max_size is not None and (total_width > max_size or total_height > max_size):
        # アスペクト比を維持しながら長辺をmax_sizeに合わせる
        aspect = total_width / total_height
        if total_width > total_height:
            new_width = max_size
            new_height = int(max_size / aspect)
        else:
            new_height = max_size
            new_width = int(max_size * aspect)
        
        result = cv2.resize(result, (new_width, new_height))
    
    return result



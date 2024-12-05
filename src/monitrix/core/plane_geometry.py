# Copyright (C) 2024 Arisaka Naoya
# SPDX-License-Identifier: AGPL-3.0-or-later
from typing import TypeVar, Literal
from functools import cached_property

import numpy as np
import numpy.typing as npt
import cv2


from monitrix.utils.logger import get_logger

logger = get_logger("monitrix.core")


T = TypeVar("T", bound="Polygon2D")

class LimitShapeArray(np.ndarray):
    """指定シェイプに制限されるNDArray配列"""
    def __new__(cls, input_array, shape_pattern:tuple[int|None,...]|None=None):
        obj = np.asarray(input_array).view(cls)
        obj.shape_pattern = shape_pattern

        # 初期化時にシェイプをチェック
        if shape_pattern is not None:
            obj._check_shape()

        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.shape_pattern = getattr(obj, 'shape_pattern', None)

    def _check_shape(self):
        if self.shape_pattern is not None:
            if len(self.shape) != len(self.shape_pattern):
                error_message = f"Array has {len(self.shape)} dimensions, but shape pattern specifies {len(self.shape_pattern)}"
                logger.error(error_message)
                raise ValueError(error_message)
            
            for actual, pattern in zip(self.shape, self.shape_pattern):
                if pattern is not None and actual != pattern:
                    error_message = f"Shape {self.shape} does not match pattern {self.shape_pattern}"
                    # logger.error(error_message)
                    raise ValueError(error_message)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._check_shape()

    @classmethod
    def sort_clockwise(cls, array2d):
        """時計回りに並び替え"""
        # 重心を計算
        center = np.mean(array2d, axis=0)
        
        # 各点と重心を結ぶベクトルの角度を計算
        angles = np.arctan2(array2d[:, 1] - center[1], array2d[:, 0] - center[0])
        
        # 角度でソート
        sorted_indices = np.argsort(angles)
        
        # ソートされた順序で点を並び替え
        sorted_points = array2d[sorted_indices]
        
        return sorted_points

    @classmethod
    def sort_clockwise_vectorized(cls, array2d: npt.NDArray) -> npt.NDArray:
        """
        ベクトル化バージョン - メモリ効率を改善
        """
        # ブロードキャストを使用して中心点を計算
        center = np.mean(array2d, axis=0, keepdims=True)
        
        # 差分計算を一度の操作で実行
        diff = array2d - center
        
        # インデックスでソートして返す
        return array2d[np.argsort(np.arctan2(diff[:, 1], diff[:, 0]))]
    
    def __hash__(self):
        return hash(tuple(self))
    
    def __eq__(self, other):
        if not isinstance(other, LimitShapeArray):
            return NotImplemented
        return np.array_equal(self, other)


class Point2D(LimitShapeArray):
    """二次元平面上の点"""
    def __new__(cls, input_array):
        return super().__new__(cls, input_array,(2,))
    @cached_property
    def x(self)->float:
        return self[0]
    @cached_property
    def y(self)->float:
        return self[1]



class Polygon2D(LimitShapeArray):
    """
    二次元平面上の点の集合
    - カーブ、輪郭、点群、ポリゴン（多角形）
    - Curve, Contour, Points, polygon

    左下が原点の数学座標系
    """

    def __new__(cls, input_array, point_nums:int|None = None):

        obj = super().__new__(cls, input_array,(point_nums,2))

        # cv2はnp.float32を対象としている
        return cls.sort_clockwise_vectorized(obj.astype(np.float32)).view(cls)

    @cached_property
    def area(self)->float:
        """輪郭の面積"""
        return cv2.contourArea(self, oriented=False)

    @cached_property
    def length(self):
        """輪郭の周囲の長さ"""
        return cv2.arcLength(self, closed=True)

    @cached_property
    def convex_hull(self)->"Polygon2D":
        """
        輪郭の凸包を計算
        """
        return Polygon2D(cv2.convexHull(self).reshape(-1,2))
    
    # def ramer_douglas_peucker(self, ratio:float|None=0.01)->"Polygon2D":
    #     """
    #     Ramer–Douglas–Peucker アルゴリズムを使用した ROI の点の密度の削減

    #     - 輪郭をより少ない点で近似
    #     """
    #     # 輪郭の周囲の長さ
    #     arclen = self.length
    #     # 凸包
    #     convex_hull = self.convex_hull
    #     # Ramer-Douglas-Peucker アルゴリズム
    #     _ratio = 1/len(convex_hull)  if ratio is None else ratio
    #     return Polygon2D(cv2.approxPolyDP(convex_hull, epsilon=_ratio * arclen, closed=True).reshape(-1,2))
    
    def plot(self, close:bool=True)->npt.NDArray:
        """描画用データの作成"""
        return (np.vstack([self, self[0]]) if close else self).T

    @cached_property
    def approx_tetragon(self)->"Tetragon2D":
        """
        凸四角形の算出
        """
        if len(self)<4:
            raise ValueError(f"To estimate a quadrangle, it must be a polygon with four or more coordinates. : {len(self)}")

        return douglas_peucker_tetragon_approximation(self)
    

    def transform_points(self:T, matrix)->T:
        """透視変換行列matrixを適応"""
        # 点群を同次座標に変換 (x, y, 1)
        ones = np.ones(shape=(len(self), 1))
        points_homogeneous = np.hstack((self, ones))

        # 変換行列を適用
        transformed_points = matrix.dot(points_homogeneous.T).T

        # 同次座標から通常の座標に戻す
        transformed_points = transformed_points[:, :2] / transformed_points[:, 2:]

        return self.__class__(transformed_points)

    def shift(self:T, origin:Point2D)->T:
        """平行移動"""
        return self.__class__(self - origin)
    






class Tetragon2D(Polygon2D):
    """
    二次元平面上の（凸）四角形
    """
    def __new__(cls, input_array):
        return super().__new__(cls, input_array, point_nums=4)
    
    @cached_property
    def leftbottom(self)->Point2D:
        """左下の点"""
        return Point2D(self[0])
    @cached_property
    def rightbottom(self)->Point2D:
        """右下の点"""
        return Point2D(self[1])
    @cached_property
    def righttop(self)->Point2D:
        """右上の点"""
        return Point2D(self[2])
    @cached_property
    def lefttop(self)->Point2D:
        """左上の点"""
        return Point2D(self[3])
    @cached_property
    def bounding_rectangle(self)->"Rectangle2D":
        """傾いていない外接する矩形領域"""
        return Rectangle2D.from_xywh(*cv2.boundingRect(self))
    

    def get_perspective_transform(self, other_tetragon:"Tetragon2D"):
        """透視変換行列matrixを取得"""
        return cv2.getPerspectiveTransform(self,other_tetragon)
    
    def __rshift__(self, other_tetragon:"Tetragon2D")->"Tetragon2D":
        """
        透視変換行列matrixを取得して変換
        z = a >> b
        """
        if isinstance(other_tetragon, Tetragon2D):
            return self.transform_points(self.get_perspective_transform(other_tetragon))
        else:
            error_message = f"other_tetragonがTetragon2Dではない:{type(other_tetragon)}"
            logger.error(error_message)
            raise ValueError(error_message)


class Rectangle2D(Tetragon2D):
    """
    二次元平面上の（回転なし）長方形
    """
    def __new__(cls, input_array):
        obj = super().__new__(cls, input_array)
        if cls.is_rectangle(obj):
            return obj
        else:
            error_message = f"Does not satisfy the condition of a rectangle without rotation"
            # logger.error(error_message)
            raise ValueError(error_message)

    @classmethod
    def is_rectangle(cls, tetragon:Tetragon2D)->bool:
        return all((
            tetragon.leftbottom.x == tetragon.lefttop.x,
            tetragon.leftbottom.y == tetragon.rightbottom.y,
            tetragon.rightbottom.x == tetragon.righttop.x,
            tetragon.lefttop.y == tetragon.righttop.y,
        ))
    
    @cached_property
    def aspect_ratio(self)->float:
        """アスペクト比"""
        _,_, width, height = self.xywh
        return abs(width / height)
    
    @cached_property
    def center(self) -> Point2D:
        """中心座標"""
        x1, y1, x2, y2 = self.xyxy
        return Point2D(
            ((x1 + x2) / 2, (y1 + y2) / 2)
        )

    @cached_property
    def xyxy(self)->npt.NDArray:
        """ to (x1, y1, x2, y2)
         - (x1, y1): 左上隅の座標
         - (x2, y2): 右下隅の座標
        """
        return np.array([*self.lefttop, *self.rightbottom])

    @cached_property
    def xywh(self)->npt.NDArray:
        """ to (x, y, width, height)
         - x, y: 左上隅の座標
         - width: 幅
         - height: 高さ
        """
        x1, y1, x2, y2 = self.xyxy
        return np.array((x1, y1, x2 - x1, y2 - y1))
    
    @cached_property
    def height(self)->float:
        _, _, _, height = self.xywh
        return abs(height)
    
    @cached_property
    def width(self)->float:
        _, _, width, _ = self.xywh
        return abs(width)

    @classmethod
    def from_xyxy(cls, x1:float|int, y1:float|int, x2:float|int, y2:float|int)->"Rectangle2D":
        """ (x1, y1, x2, y2)
         - (x1, y1): 左上隅の座標
         - (x2, y2): 右下隅の座標
        """
        return Rectangle2D([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])
    
    @classmethod
    def from_xywh(cls, x:float|int, y:float|int, width:float|int, height:float|int)->"Rectangle2D":
        """ (x, y, width, height)
         - x, y: 左上隅の座標
         - width: 幅
         - height: 高さ
        """
        return Rectangle2D([(x, y), (x+width, y), (x+width, y+height), (x, y+height)])


    def adjust_bbox_aspect_ratio(
            self,
            target_ratio: float,
            mode: Literal["expand","shrink","auto"] = 'auto'
    ) -> "Rectangle2D":
        """
        矩形のアスペクト比を変更する
        * バグがありそう modeはautoにする
        
        Args:
            bbox: 元の矩形
            target_ratio: 目標のアスペクト比 (width / height)
            mode: 調整モード
                - 'expand': 小さい方の辺を広げてアスペクト比を調整
                - 'shrink': 大きい方の辺を縮めてアスペクト比を調整
                - 'auto': 元の面積により近くなる方を選択
        
        Returns:
            新しいアスペクト比の矩形
        """
        current_ratio = self.aspect_ratio
        center_x, center_y = self.center

        if abs(current_ratio - target_ratio) < 1e-6:
            return self
        
        _,_, width, height = self.xywh
        if mode == 'auto':
            # 拡大と縮小それぞれの場合の面積を計算
            current_area = width * height
            
            # expand時の面積
            if current_ratio < target_ratio:
                expand_width = height * target_ratio
                expand_area = expand_width * height
            else:
                expand_height = width / target_ratio
                expand_area = width * expand_height
                
            # shrink時の面積
            if current_ratio < target_ratio:
                shrink_height = width / target_ratio
                shrink_area = width * shrink_height
            else:
                shrink_width = height * target_ratio
                shrink_area = shrink_width * height
            
            # 元の面積により近い方を選択
            mode = 'expand' if abs(expand_area - current_area) < abs(shrink_area - current_area) else 'shrink'
        
        if mode == 'expand':
            if current_ratio < target_ratio:
                # 幅を広げる
                new_width = height * target_ratio
                new_height = height
            else:
                # 高さを広げる
                new_width = width
                new_height = width / target_ratio
        else:  # mode == 'shrink'
            if current_ratio < target_ratio:
                # 高さを縮める
                new_width = width
                new_height = width / target_ratio
            else:
                # 幅を縮める
                new_width = height * target_ratio
                new_height = height
        
        # 中心を維持しながら新しい矩形を作成
        return Rectangle2D.from_xyxy(
            x1=center_x - new_width / 2,
            y1=center_y - new_height / 2,
            x2=center_x + new_width / 2,
            y2=center_y + new_height / 2
        )



# 凸四角形の近似法
# - Douglas-Peucker（ダグラス・ポイカー）アルゴリズム
# - Minimum Area Enclosing Quadrilateral (MAEQ)アルゴリズム
# - Principal Component Analysis (PCA)
# - 最小二乗法による近似
# - モーメントベースの方法
# - エネルギー最小化アプローチ


    
def douglas_peucker_tetragon_approximation(polygon:Polygon2D)->"Tetragon2D":
    """
    Ramer–Douglas–Peucker アルゴリズムを使用した凸四角形の近似
    """

    def ramer_douglas_peucker(polygon:Polygon2D, ratio:float|None=0.01)->"Polygon2D":
        """
        Ramer–Douglas–Peucker アルゴリズムを使用した ROI の点の密度の削減

        - 輪郭をより少ない点で近似
        """
        # 輪郭の周囲の長さ
        arclen = polygon.length
        # 凸包
        convex_hull = polygon.convex_hull
        # Ramer-Douglas-Peucker アルゴリズム
        _ratio = 1/len(convex_hull)  if ratio is None else ratio
        return Polygon2D(cv2.approxPolyDP(convex_hull, epsilon=_ratio * arclen, closed=True).reshape(-1,2))

    _polygon = ramer_douglas_peucker(polygon)
    if len(_polygon) < 4:
        n = len(_polygon.convex_hull)
        i = 1
        while len(_polygon) < 4:
            _polygon = ramer_douglas_peucker(_polygon, 1/(n+i))
            i += 1 
        logger.debug(f"4点未満 it:{i}")
    elif len(_polygon) > 4:
        # 5点以上
        n = len(_polygon.convex_hull)
        for i in range(n-1):
            rate = 1/(n-(i+1))/4
            _polygon = ramer_douglas_peucker(_polygon, rate)
            if len(_polygon) == 4:
                break
        logger.debug(f"5点以上 it:{i}")

    return Tetragon2D(_polygon)

def pca_quadrilateral_tetragon_approximation(polygon:Polygon2D)->"Tetragon2D":
    """
    Optimized version of PCA quadrilateral approximation
    """
    # Precompute mean for centering
    mean = np.mean(polygon, axis=0, keepdims=True)
    centered = polygon - mean
    
    # Compute PCA using SVD instead of eigendecomposition
    # SVD is generally more numerically stable and faster for this case
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    
    # Project points onto principal components
    # Using matrix multiplication optimization
    projected = centered @ Vt.T
    
    # Find extremal points using vectorized operations
    #extremes_idx = []
    directions = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
    
    # Vectorized dot product for all points with all directions
    dots = projected @ directions.T
    #extremes_idx = 
    
    # Convert back to original space
    return Tetragon2D(polygon[np.argmax(dots, axis=0)])




def fast_douglas_peucker_tetragon_approximation(polygon:Polygon2D)->"Tetragon2D":
    """
    最適化されたRamer–Douglas–Peucker アルゴリズムを使用した凸四角形の近似
    
    Args:
        polygon: 入力ポリゴン (np.ndarray)
        
    Returns:
        近似された4点の凸四角形 (np.ndarray)
    """
    @np.vectorize
    def calculate_point_line_distance(px, py, x1, y1, x2, y2):
        """ベクトル化された点と線分の距離計算"""
        numerator = abs((x2 - x1) * (y1 - py) - (x1 - px) * (y2 - y1))
        denominator = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return numerator / denominator

    def fast_ramer_douglas_peucker(points, epsilon):
        """高速化されたRamer-Douglas-Peucker実装"""
        if len(points) <= 2:
            return points
            
        # 始点と終点の座標を取得
        start, end = points[0], points[-1]
        
        # NumPyのベクトル化された距離計算
        distances = calculate_point_line_distance(
            points[:, 0], points[:, 1],
            start[0], start[1],
            end[0], end[1]
        )
        
        # 最大距離のインデックスを取得
        max_dist_index = np.argmax(distances)
        max_distance = distances[max_dist_index]
        
        if max_distance > epsilon:
            # 再帰的に処理（スライスを使用して配列のコピーを最小限に）
            left = fast_ramer_douglas_peucker(points[:max_dist_index + 1], epsilon)
            right = fast_ramer_douglas_peucker(points[max_dist_index:], epsilon)
            return np.concatenate([left[:-1], right])
        
        return np.array([start, end])

    def optimize_epsilon(points:Polygon2D, target_points=4):
        """
        二分探索を使用した効率的なepsilon値の探索
        """
        arclen = cv2.arcLength(points.astype(np.float32), True)
        epsilon_min, epsilon_max = 0.001 * arclen, 0.1 * arclen
        best_approx = None
        best_diff = float('inf')
        
        # 初期のConvex Hullを計算
        convex_points = points.convex_hull
        
        for _ in range(10):  # 反復回数を制限
            epsilon = (epsilon_min + epsilon_max) / 2
            approx = fast_ramer_douglas_peucker(convex_points, epsilon)
            
            point_count = len(approx)
            current_diff = abs(point_count - target_points)
            
            if current_diff < best_diff:
                best_diff = current_diff
                best_approx = approx
            
            if point_count == target_points:
                return approx
            
            if point_count > target_points:
                epsilon_min = epsilon
            else:
                epsilon_max = epsilon
        
        return best_approx

    def fast_adjust_points(points):
        """
        高速な4点調整
        """
        if len(points) == 4:
            return points
            
        points = points.astype(np.float32)
        if len(points) < 4:
            # 最も長い辺を見つけて分割
            while len(points) < 4:
                # ベクトル化された距離計算
                edges = np.roll(points, -1, axis=0) - points
                distances = np.sqrt(np.sum(edges**2, axis=1))
                max_idx = np.argmax(distances)
                
                # 中点を挿入
                mid_point = (points[max_idx] + points[(max_idx + 1) % len(points)]) / 2
                points = np.insert(points, max_idx + 1, mid_point, axis=0)
        
        elif len(points) > 4:
            # 最も近い点のペアを統合
            while len(points) > 4:
                edges = np.roll(points, -1, axis=0) - points
                distances = np.sqrt(np.sum(edges**2, axis=1))
                min_idx = np.argmin(distances)
                
                mid_point = (points[min_idx] + points[(min_idx + 1) % len(points)]) / 2
                points = np.delete(points, [min_idx, (min_idx + 1) % len(points)], axis=0)
                points = np.insert(points, min_idx, mid_point, axis=0)
        
        return points

    # メイン処理の最適化
    approx = optimize_epsilon(polygon)
    result = fast_adjust_points(approx)
    return Tetragon2D(result)



# import matplotlib.pyplot as plt

# n = 53
# points = Polygon2D(np.random.rand(n,2)*100)

# plt.plot(*points.T, ".--")
# print(points.dtype)
# print(points.area())
# plt.plot(*points.convex_hull().plot(), ".--")
# plt.plot(*points.ramer_douglas_peucker().plot(), ".--")
# plt.plot(*points.approx_tetragon().plot(), ".--")
# plt.plot(*points.approx_tetragon().bounding_rectangle().plot(), "+--",lw=1)




# 透視投影
# plt.plot(*points.T, ".--")
# plt.plot(*points.approx_tetragon().plot(), "--")
# _mx = points.approx_tetragon().get_perspective_transform(points.approx_tetragon().bounding_rectangle())
# plt.plot(*points.transform_points(_mx).T, ".--")

# new_p = points.approx_tetragon() >> points.approx_tetragon().bounding_rectangle()
# plt.plot(*new_p.plot(), ".--",lw=0.5, color="yellow")


# x1, y1, x2, y2 = (1,8,3,4)
# bbox = Rectangle2D.from_xyxy(*(x1, y1, x2, y2))
# bbox2 = Rectangle2D.from_xywh(*bbox.xywh)
# plt.plot(*bbox.plot(), ".--")
# plt.plot(*bbox2.plot(), ".--")




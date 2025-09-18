import numpy as np
import cv2

TARGET_POINTS = 4
MAX_ITERATIONS = 100
DEFAULT_RATIO = 0.01
EPSILON_TOLERANCE = 1e-10


def douglas_peucker(polygon: np.ndarray) -> np.ndarray:
    """
    Ramer–Douglas–Peucker アルゴリズムを使用した凸四角形の近似

    Args:
        polygon: 入力ポリゴン

    Returns:
        近似された4点の凸四角形
    """
    # 入力検証
    if len(polygon) < 3:
        raise ValueError("Polygon must have at least 3 points")

    # 型とメモリレイアウトの最適化
    if polygon.dtype != np.float32:
        polygon = polygon.astype(np.float32)

    # 最初の近似
    _polygon = _ramer_douglas_peucker(polygon)

    # 既に4点の場合は早期リターン
    if len(_polygon) == TARGET_POINTS:
        return _polygon

    # 点数が4点未満の場合の処理
    if len(_polygon) < TARGET_POINTS:
        # 凸包の点数を基準とした適応的近似
        convex_hull = _convex_hull(_polygon)
        n = len(convex_hull)

        # 効率的な反復処理
        for i in range(1, min(MAX_ITERATIONS, n * 2)):
            ratio = 1.0 / (n + i)
            _polygon = _ramer_douglas_peucker(_polygon, ratio)

            if len(_polygon) >= TARGET_POINTS:
                break

        # まだ4点未満の場合、中点分割で補完
        if len(_polygon) < TARGET_POINTS:
            _polygon = _add_midpoints_vectorized(_polygon, TARGET_POINTS)

    # 点数が4点を超える場合の処理
    elif len(_polygon) > TARGET_POINTS:
        # 凸包ベースの適応的近似
        convex_hull = _convex_hull(_polygon)
        n = len(convex_hull)

        # より効率的な粗い近似の試行
        ratios = [1.0 / (n - i) / 4 for i in range(1, min(n, 10))]

        for ratio in ratios:
            _polygon = _ramer_douglas_peucker(_polygon, ratio)

            if len(_polygon) == TARGET_POINTS:
                break
            elif len(_polygon) < TARGET_POINTS:
                # 不足分を補完
                _polygon = _add_midpoints_vectorized(_polygon, TARGET_POINTS)
                break

        # まだ4点より多い場合、点削減処理
        if len(_polygon) > TARGET_POINTS:
            _polygon = _reduce_points_vectorized(_polygon, TARGET_POINTS)

    # 最終検証と調整
    if len(_polygon) != TARGET_POINTS:
        if len(_polygon) < TARGET_POINTS:
            _polygon = _add_midpoints_vectorized(_polygon, TARGET_POINTS)
        else:
            _polygon = _reduce_points_vectorized(_polygon, TARGET_POINTS)

    return _polygon


def _ramer_douglas_peucker(
    poly: np.ndarray, ratio: float | None = DEFAULT_RATIO
) -> np.ndarray:
    """
    Ramer–Douglas–Peucker アルゴリズム実装

    Args:
        poly: 近似する多角形（連続メモリ配列を推奨）
        ratio: 近似の比率（デフォルト: 0.01）

    Returns:
        近似された多角形
    """
    # 入力検証と前処理
    if len(poly) < 3:
        return poly.copy()

    # メモリ連続性の保証
    if not poly.flags.c_contiguous:
        poly = np.ascontiguousarray(poly, dtype=np.float32)
    elif poly.dtype != np.float32:
        poly = poly.astype(np.float32)

    # 周囲長の計算
    arclen = cv2.arcLength(poly, closed=True)

    # ゼロ周囲長の処理
    if arclen < EPSILON_TOLERANCE:
        return poly[:TARGET_POINTS] if len(poly) >= TARGET_POINTS else poly

    # 凸包の計算
    convex_hull = _convex_hull(poly)

    # 適切な比率の設定
    if ratio is None:
        _ratio = 1.0 / len(convex_hull)
    else:
        _ratio = ratio

    # OpenCVの高速な近似関数を使用
    epsilon = _ratio * arclen
    approximated = cv2.approxPolyDP(convex_hull, epsilon=epsilon, closed=True)

    return approximated.reshape(-1, 2)


def _convex_hull(poly: np.ndarray) -> np.ndarray:
    """凸包計算"""
    # 入力データの連続性を保証
    if not poly.flags.c_contiguous:
        poly = np.ascontiguousarray(poly)

    # OpenCVの凸包計算
    hull = cv2.convexHull(poly)
    return hull.reshape(-1, 2)


def _add_midpoints_vectorized(points: np.ndarray, target_count: int) -> np.ndarray:
    """ベクトル化された中点追加処理"""
    current_points = points.copy()

    while len(current_points) < target_count:
        # 全ての辺の長さを一度に計算
        edges = np.roll(current_points, -1, axis=0) - current_points
        distances = np.linalg.norm(edges, axis=1)

        # 最も長い辺のインデックス
        max_idx = np.argmax(distances)

        # 中点の計算
        next_idx = (max_idx + 1) % len(current_points)
        mid_point = (current_points[max_idx] + current_points[next_idx]) * 0.5

        # 中点を挿入
        current_points = np.insert(current_points, max_idx + 1, mid_point, axis=0)

    return current_points


def _reduce_points_vectorized(points: np.ndarray, target_count: int) -> np.ndarray:
    """ベクトル化された点削減処理"""
    current_points = points.copy()

    while len(current_points) > target_count:
        # 隣接点間の距離をベクトル計算
        edges = np.roll(current_points, -1, axis=0) - current_points
        distances = np.linalg.norm(edges, axis=1)

        # 最も近い点のペアを見つける
        min_idx = np.argmin(distances)
        next_idx = (min_idx + 1) % len(current_points)

        # 中点を計算
        mid_point = (current_points[min_idx] + current_points[next_idx]) * 0.5

        # 2点を削除して中点を挿入
        # インデックスの順序に注意して削除
        if next_idx == 0:
            # 循環の場合の特別処理
            current_points = np.delete(current_points, [min_idx, 0], axis=0)
            current_points = np.append(current_points, [mid_point], axis=0)
        else:
            current_points = np.delete(current_points, [min_idx, next_idx], axis=0)
            current_points = np.insert(current_points, min_idx, mid_point, axis=0)

    return current_points

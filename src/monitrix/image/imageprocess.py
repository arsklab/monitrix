from typing import TypeAlias, TypeVar, Callable


import numpy as np

from ultralytics.data.augment import LetterBox

Img: TypeAlias = np.ndarray
I = TypeVar("I", Img, list[Img])

UniformImageProcessor: TypeAlias = Callable[[list[Img]], tuple[tuple[int, int], I]]


def crop(
    image: Img,
    start_point: np.ndarray,
    end_point: np.ndarray,
) -> Img:
    """
    指定した開始点と終了点に基づいて画像を切り抜く

    Parameters:
        image: 切り抜く元の画像（NumPy配列）
        start_point: 切り抜き開始位置の (x, y) 座標
        end_point: 切り抜き終了位置の (x, y) 座標

    Returns:
        切り抜かれた画像（NumPy配列）
    """
    h, w = image.shape[:2]
    return image[
        max(0, start_point[1]) : min(h, end_point[1]),
        max(0, start_point[0]) : min(w, end_point[0]),
    ]


def letterbox(
    source: I, imgsz: tuple[int, int] | None = None, **kwargs
) -> tuple[tuple[int, int], I]:
    """レターボックス変換（サイズの異なる画像をレターボックス変換で同じサイズにする）"""
    if isinstance(source, list):
        if imgsz:
            target_size = imgsz
            _letterbox = LetterBox(new_shape=target_size, **kwargs)
        else:
            target_size = np.max([img.shape[:2] for img in source], axis=0)
            _letterbox = LetterBox(new_shape=target_size, **kwargs)
        return target_size, [_letterbox(image=img) for img in source]
    else:
        if imgsz:
            return imgsz, LetterBox(new_shape=imgsz, **kwargs)(image=source)
        else:
            return source.shape[:2], source

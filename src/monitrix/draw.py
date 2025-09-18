from typing import Literal

import cv2
import torch
import numpy.typing as npt
import numpy as np
from numpy import uint8
from matplotlib.colors import to_rgb

from monitrix.results_protocol import ResultsProtocol


def box_label(
    image: npt.NDArray[uint8],
    # bbox:Rectangle2D,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    text: str,
    line_width: int | None = None,
    fill_color: tuple[int, int, int] = (255, 0, 0),
    text_color: tuple[int, int, int] = (255, 0, 0),
    loc: Literal["top", "right", "righttop"] = "top",
) -> npt.NDArray[uint8]:
    """ラベルの描画"""

    # スクリーン座標系
    p1, p2 = (x1, y1), (x2, y2)

    lw: int = line_width or max(round(sum(image.shape) / 2 * 0.003), 2)

    if not image.data.contiguous:
        error = "Image not contiguous. Apply np.ascontiguousarray(im) to Annotator input images."
        raise ValueError(error)

    image = image if image.flags.writeable else image.copy()
    tf: int = max(lw - 1, 1)  # font thickness
    sf: float = lw / 3  # font scale

    w, h = cv2.getTextSize(text, 0, fontScale=sf, thickness=tf)[0]  # text width, height
    h += 3  # add pixels to pad text
    outside = p1[1] >= h  # label fits outside box
    if (
        p1[0] > image.shape[1] - w
    ):  # shape is (h, w), check if label extend beyond right side of image
        p1 = image.shape[1] - w, p1[1]
    p2 = p1[0] + w, p1[1] - h if outside else p1[1] + h

    if loc == "right":
        offset = (x2 - x1, y2 - y1)
    elif loc == "righttop":
        offset = (x2 - x1, 0)
    else:
        offset = (0, 0)
    cv2.rectangle(
        image,
        tuple(p + o for p, o in zip(p1, offset)),
        tuple(p + o for p, o in zip(p2, offset)),
        fill_color,
        -1,
        cv2.LINE_AA,
    )  # filled
    cv2.putText(
        image,
        text,
        (p1[0] + offset[0], (p1[1] - 2 if outside else p1[1] + h - 1) + offset[1]),
        0,
        sf,
        text_color,
        thickness=tf,
        lineType=cv2.LINE_AA,
    )
    return image


def box_label_with_classname(
    image: npt.NDArray[uint8],
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    text: str,
    class_name: str | None = None,
    class_text_size_rate: float = 0.6,
    line_width: int | None = None,
    fill_color: tuple[int, int, int] = (255, 0, 0),
    text_color: tuple[int, int, int] = (255, 0, 0),
    loc: Literal["top", "right", "righttop"] = "top",
) -> npt.NDArray[uint8]:
    """クラス名対応ラベルの描画

    Args:
        image: 描画対象の画像
        x1, y1, x2, y2: バウンディングボックスの座標
        text: メインテキスト（数値など）
        classname: クラス名（小さいフォントで上部に表示）
        line_width: 線の太さ
        fill_color: 背景色
        text_color: テキスト色
        loc: ラベルの位置
    """

    # スクリーン座標系
    p1, p2 = (x1, y1), (x2, y2)

    lw: int = line_width or max(round(sum(image.shape) / 2 * 0.003), 2)

    if not image.data.contiguous:
        error = "Image not contiguous. Apply np.ascontiguousarray(im) to Annotator input images."
        raise ValueError(error)

    image = image if image.flags.writeable else image.copy()
    tf: int = max(lw - 1, 1)  # font thickness
    sf: float = lw / 3  # font scale

    # メインテキストのサイズを計算
    w_main, h_main = cv2.getTextSize(text, 0, fontScale=sf, thickness=tf)[0]
    h_main += 3  # add pixels to pad text

    # クラス名のサイズを計算（小さいフォント）
    w_class, h_class = 0, 0
    sf_class = sf * class_text_size_rate  # クラス名は60%のサイズ
    tf_class = max(1, tf - 1)  # クラス名の線の太さ

    if class_name:
        w_class, h_class = cv2.getTextSize(
            class_name, 0, fontScale=sf_class, thickness=tf_class
        )[0]
        h_class += 2  # パディング

    # 全体のサイズを計算
    w = max(w_main, w_class)
    h = h_main + (h_class if class_name else 0)

    outside = p1[1] >= h  # label fits outside box
    if (
        p1[0] > image.shape[1] - w
    ):  # shape is (h, w), check if label extend beyond right side of image
        p1 = image.shape[1] - w, p1[1]
    p2 = p1[0] + w, p1[1] - h if outside else p1[1] + h

    # 位置オフセットの計算
    if loc == "right":
        offset = (x2 - x1, y2 - y1)
    elif loc == "righttop":
        offset = (x2 - x1, 0)
    else:
        offset = (0, 0)

    # 背景矩形を描画
    cv2.rectangle(
        image,
        tuple(p + o for p, o in zip(p1, offset)),
        tuple(p + o for p, o in zip(p2, offset)),
        fill_color,
        -1,
        cv2.LINE_AA,
    )  # filled

    # テキストの描画位置を計算
    if outside:
        # 上側に表示する場合
        if class_name:
            # クラス名を上部に描画
            cv2.putText(
                image,
                class_name,
                (p1[0] + offset[0], p1[1] - h + h_class - 2 + offset[1]),
                0,
                sf_class,
                text_color,
                thickness=tf_class,
                lineType=cv2.LINE_AA,
            )
            # メインテキストをその下に描画
            cv2.putText(
                image,
                text,
                (p1[0] + offset[0], p1[1] - 2 + offset[1]),
                0,
                sf,
                text_color,
                thickness=tf,
                lineType=cv2.LINE_AA,
            )
        else:
            # クラス名がない場合は通常通り
            cv2.putText(
                image,
                text,
                (p1[0] + offset[0], p1[1] - 2 + offset[1]),
                0,
                sf,
                text_color,
                thickness=tf,
                lineType=cv2.LINE_AA,
            )
    else:
        # 下側に表示する場合
        if class_name:
            # クラス名を上部に描画
            cv2.putText(
                image,
                class_name,
                (p1[0] + offset[0], p1[1] + h_class - 1 + offset[1]),
                0,
                sf_class,
                text_color,
                thickness=tf_class,
                lineType=cv2.LINE_AA,
            )
            # メインテキストをその下に描画
            cv2.putText(
                image,
                text,
                (p1[0] + offset[0], p1[1] + h - 1 + offset[1]),
                0,
                sf,
                text_color,
                thickness=tf,
                lineType=cv2.LINE_AA,
            )
        else:
            # クラス名がない場合は通常通り
            cv2.putText(
                image,
                text,
                (p1[0] + offset[0], p1[1] + h_main - 1 + offset[1]),
                0,
                sf,
                text_color,
                thickness=tf,
                lineType=cv2.LINE_AA,
            )

    return image


def polygon_area(
    image: npt.NDArray[uint8],
    vertex: list[tuple[int, int]] | npt.NDArray[np.int32],
    color: tuple[int, int, int],
    alpha: float = 0.3,
    thickness: int = 2,
) -> npt.NDArray[uint8]:
    """
    半透明な多角形を描画する関数

    Parameters:
        image: 描画対象の画像
        vertex: 多角形の頂点座標のリスト [(x1,y1), (x2,y2), ...]
        color: 描画色 (B,G,R)
        alpha: 透明度 (0: 透明 ~ 1: 不透明)
        thickness: 線の太さ（-1の場合は塗りつぶし）

    Returns:
        result_image: 描画後の画像
    """
    # 入力画像のコピーを作成
    overlay = image.copy()

    # 頂点座標を整形
    pts = np.array(vertex, np.int32)
    pts = pts.reshape((-1, 1, 2))

    # 塗りつぶし用の多角形を描画
    result_image: npt.NDArray[uint8]
    if alpha > 0.0:
        cv2.fillPoly(overlay, [pts], color)
        # アルファブレンディング
        result_image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    else:
        result_image = overlay

    # 輪郭線を描画（thickness > 0の場合）
    if thickness > 0:
        cv2.polylines(result_image, [pts], True, color, thickness)

    return result_image


def points(
    image: npt.NDArray[uint8],
    vertex: list[tuple[int, int]] | npt.NDArray[np.int32],
    color: tuple[int, int, int],
    vertex_size: int = 8,
):

    # 各頂点を描画
    for point in vertex:
        x, y = point
        # 頂点の形状に応じて描画
        cv2.circle(
            img=image,
            center=(int(x), int(y)),
            radius=vertex_size,
            color=color,
            thickness=-1,
        )
    return image


def plot(
    results: ResultsProtocol,
    class_names: dict[int, str] | None = None,  # key=cls
    colors: (
        dict[int, tuple[tuple[int, int, int] | str, tuple[int, int, int] | str]] | None
    ) = None,  # key=cls
    trues: dict[int, int | float] | None = None,  # key=id
    thickness: int = 1,
    marker_size: int = 5,
    label_size: int = 2,
    area_alpha: float = 0.1,
    default_fill_color: tuple[int, int, int] | str = "#F85525",  # bgr
    default_text_color: tuple[int, int, int] | str = "#01204E",  # bgr
    class_text_size_rate: float = 0.6,
    cvt_rgb: bool = False,
) -> npt.NDArray[uint8]:

    if isinstance(results.orig_img, torch.Tensor):
        image = results.orig_img[0].detach().to(torch.uint8).cpu().numpy()
    else:
        image = results.orig_img.copy()

    if boxes := results.boxes:
        if masks := results.masks:
            if ocrs := results.ocrs:
                if boxes.id is not None:
                    miss_flag = False
                    for flag, box, mask, number, _id, cls in zip(
                        boxes.is_base_box,
                        boxes.xyxy,
                        masks.xy,
                        ocrs.number,
                        boxes.id,
                        boxes.cls,
                    ):
                        class_name = (
                            class_names if class_names is not None else {}
                        ).get(int(cls), f"cls={int(cls)}")
                        fill_color, text_color = (
                            colors if colors is not None else {}
                        ).get(int(cls), (default_fill_color, default_text_color))

                        if flag == 1:  # モニタ
                            text = f"id={int(_id)}"
                            loc = "top"
                        else:
                            text = f"{number}"
                            loc = "right"

                            true_value = (trues if trues is not None else {}).get(
                                int(_id), number
                            )
                            if true_value != number:
                                miss_flag = True
                                text = "x " + text + f"({true_value})"
                                fill_color = "#FF0000"
                                text_color = "#FFFFFF"
                                thickness = thickness * 5

                        if isinstance(fill_color, str):
                            fill_color = bgr(fill_color)
                        if isinstance(text_color, str):
                            text_color = bgr(text_color)
                        image = polygon_area(
                            image,
                            mask,
                            fill_color,
                            alpha=area_alpha,
                            thickness=thickness,
                        )
                        image = points(
                            image,
                            to_points(box.numpy()),
                            fill_color,
                            vertex_size=marker_size,
                        )
                        image = box_label_with_classname(
                            image,
                            *box.numpy().astype(int),
                            text=text,
                            class_name=class_name,
                            fill_color=fill_color,
                            text_color=text_color,
                            loc=loc,
                            line_width=label_size,
                            class_text_size_rate=class_text_size_rate,
                        )

                    if miss_flag:
                        h, w = image.shape[:2]
                        fill_color = bgr("#FF0000")
                        image = cv2.rectangle(
                            image,
                            (0, 0),
                            (w, h),
                            fill_color,
                            30,
                            cv2.LINE_AA,
                        )

    # フレーム番号
    cv2.putText(
        image,
        f"{results.frame_no}",
        (50, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        2,
    )

    if cvt_rgb:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def to_points(xyxy: npt.NDArray) -> npt.NDArray:
    """
    xyxy形式のbboxをpoints形式に変換
    Args:
        bbox: [x1, y1, x2, y2] 形式のbounding box
              x1, y1: 左上の座標
              x2, y2: 右下の座標
    Returns:
        points: 矩形の4つの角の座標
    """
    x1, y1, x2, y2 = xyxy

    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def rgb(color: str) -> tuple[int, int, int]:
    return tuple(int(c * 255) for c in to_rgb(color))


def bgr(color: str) -> tuple[int, int, int]:
    return rgb(color)[::-1]

# visualize.py
# Utilities for drawing heatmaps, boxes, points, readable text, and gaze arrows.

import cv2
import numpy as np
from typing import List, Tuple, Optional

from config import HEATMAP_ALPHA, DRAW_ARGMAX_DOT, DOT_RADIUS

def overlay_heatmap(
    image_bgr: np.ndarray,
    heatmap: np.ndarray,
    alpha: Optional[float] = None,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    if alpha is None:
        alpha = HEATMAP_ALPHA
    hm = heatmap.astype(np.float32)
    hm -= hm.min()
    maxv = hm.max()
    if maxv > 0:
        hm /= maxv
    hm_u8 = np.clip(hm * 255.0, 0, 255).astype(np.uint8)
    H, W = image_bgr.shape[:2]
    hm_u8 = cv2.resize(hm_u8, (W, H), interpolation=cv2.INTER_CUBIC)
    hm_color = cv2.applyColorMap(hm_u8, colormap)
    out = cv2.addWeighted(image_bgr, 1.0, hm_color, float(alpha), 0.0)
    return out

def heatmap_argmax_xy(heatmap: np.ndarray, W: int, H: int) -> Tuple[int, int]:
    if heatmap.ndim != 2:
        raise ValueError("heatmap must be 2D")
    iy, ix = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    x = int(round((ix / max(heatmap.shape[1] - 1, 1)) * (W - 1)))
    y = int(round((iy / max(heatmap.shape[0] - 1, 1)) * (H - 1)))
    return x, y

def enlarge_box(px_box: List[int], W: int, H: int, fx: float = 0.15, fy: float = 0.20) -> List[int]:
    x1, y1, x2, y2 = px_box
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    nw = w * (1.0 + 2.0 * fx)
    nh = h * (1.0 + 2.0 * fy)
    nx1 = int(max(round(cx - nw * 0.5), 0))
    ny1 = int(max(round(cy - nh * 0.5), 0))
    nx2 = int(min(round(cx + nw * 0.5), W - 1))
    ny2 = int(min(round(cy + nh * 0.5), H - 1))
    if nx2 <= nx1: nx2 = min(nx1 + 1, W - 1)
    if ny2 <= ny1: ny2 = min(ny1 + 1, H - 1)
    return [nx1, ny1, nx2, ny2]

def draw_boxes_and_points(
    img_bgr: np.ndarray,
    boxes: List[List[int]],
    points: Optional[List[Optional[Tuple[int, int]]]] = None,
    box_color: Tuple[int, int, int] = (255, 255, 255),
    point_color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    vis = img_bgr.copy()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, thickness)
        if points and i < len(points) and points[i] is not None and DRAW_ARGMAX_DOT:
            x, y = points[i]
            cv2.circle(vis, (x, y), DOT_RADIUS, point_color, -1)
    return vis

def draw_text(
    img_bgr: np.ndarray,
    text: str,
    org: Tuple[int, int],
    font_scale: float = 0.8,
    color: Tuple[int, int, int] = (255, 255, 255),
    thickness: int = 2,
    bg_color: Tuple[int, int, int] = (0, 0, 0),
    bg_alpha: float = 0.45,
) -> np.ndarray:
    vis = img_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    pad = 6
    x1, y1 = x - pad, y - th - pad
    x2, y2 = x + tw + pad, y + baseline + pad
    H, W = vis.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W - 1, x2), min(H - 1, y2)
    overlay = vis.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), bg_color, -1)
    vis = cv2.addWeighted(overlay, bg_alpha, vis, 1 - bg_alpha, 0)
    ty = max(y, y1 + th + 1)
    cv2.putText(vis, text, (x, ty), font, font_scale, color, thickness, cv2.LINE_AA)
    return vis

def _box_center(box: List[int]) -> Tuple[int, int]:
    x1, y1, x2, y2 = box
    return int((x1 + x2) * 0.5), int((y1 + y2) * 0.5)

def draw_gaze_arrows(
    img_bgr: np.ndarray,
    src_points: List[Tuple[int, int]],
    dst_points: List[Optional[Tuple[int, int]]],
    color: Tuple[int, int, int] = (0, 255, 255),  # yellow
    thickness: int = 2,
    tip_length: float = 0.25,
) -> np.ndarray:
    """Draw arrow from src -> dst for each pair; skips None."""
    vis = img_bgr.copy()
    for s, d in zip(src_points, dst_points):
        if s is None or d is None:
            continue
        cv2.arrowedLine(vis, s, d, color, thickness, tipLength=tip_length)
    return vis

def centers_from_boxes(boxes: List[List[int]]) -> List[Tuple[int, int]]:
    return [_box_center(b) for b in boxes]

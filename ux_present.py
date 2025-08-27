# ux_present.py
# Presentation-quality overlay: stable IDs, soft labels, colored boxes and gaze beams.
from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np
from collections import defaultdict

# ---------- Style knobs ----------
ARROW_TIP = 0.12
ARROW_THICKNESS = 2
ARROW_HIDE_THRESH = 0.20   # hide arrow if in-frame < this  (was 0.5)
BOX_THICKNESS = 2
PILL_ALPHA = 0.65
PILL_PAD_X = 6
PILL_PAD_Y = 4
SMALL_FONT = 0.4           # nameplate font
TINY_FONT = 0.25           # tiny line under face
TEXT_THICK = 1

PALETTE = [
    (80, 180, 255),   # light orange (BGR)
    (120, 220, 120),  # light green
    (255, 180, 80),   # light blue
    (200, 140, 255),  # pink
    (140, 220, 220),  # teal
    (220, 220, 140),  # sand
    (180, 180, 255),  # lavender
    (180, 255, 180),  # mint
    (80, 255, 180),
    (140, 220, 220),
    (140, 200, 220)
]

def _color_for_id(pid: int) -> Tuple[int,int,int]:
    return PALETTE[pid % len(PALETTE)]

def _center(box: List[int]) -> Tuple[int,int]:
    x1,y1,x2,y2 = box
    return int((x1+x2)/2), int((y1+y2)/2)

def _draw_pill(img, text, org, bg_bgr, txt_bgr=(255,255,255),
               font=cv2.FONT_HERSHEY_SIMPLEX, scale=SMALL_FONT,
               thick=TEXT_THICK, alpha=PILL_ALPHA):
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    x, y = int(org[0]), int(org[1])
    x1 = x - PILL_PAD_X
    y1 = y - th - PILL_PAD_Y
    x2 = x + tw + PILL_PAD_X
    y2 = y + bl + PILL_PAD_Y

    H, W = img.shape[:2]
    x1, y1 = max(0,x1), max(0,y1)
    x2, y2 = min(W-1,x2), min(H-1,y2)

    overlay = img.copy()
    cv2.rectangle(overlay, (x1,y1), (x2,y2), bg_bgr, -1)
    out = cv2.addWeighted(overlay, alpha, img, 1-alpha, 0)

    ty = max(y, y1 + th + 1)   # baseline inside the pill
    cv2.putText(out, text, (x, ty), font, scale, txt_bgr, thick, cv2.LINE_AA)
    return out

# ---------- Simple IoU tracker for stable IDs ----------
def _iou(a, b):
    ax1,ay1,ax2,ay2 = a; bx1,by1,bx2,by2 = b
    ix1,iy1 = max(ax1,bx1), max(ay1,by1)
    ix2,iy2 = min(ax2,bx2), min(ay2,by2)
    iw,ih = max(0,ix2-ix1), max(0,iy2-iy1)
    inter = iw*ih
    areaA = max(0,ax2-ax1)*max(0,ay2-ay1)
    areaB = max(0,bx2-bx1)*max(0,by2-by1)
    return inter / (areaA + areaB - inter + 1e-6)

class SimpleIOUTracker:
    def __init__(self, iou_thresh=0.3, max_ghost=15):
        self.iou_thresh = iou_thresh
        self.max_ghost = max_ghost
        self.next_id = 0
        self.box_by_id: Dict[int, List[int]] = {}
        self.ghost: Dict[int, int] = {}

    def update(self, boxes: List[List[int]]) -> List[int]:
        ids = [-1]*len(boxes)
        used_prev = set()
        items = list(self.box_by_id.items())
        # match to existing by best IoU
        for i, b in enumerate(boxes):
            best_id, best_iou = -1, 0.0
            for pid, pb in items:
                if pid in used_prev: continue
                j = _iou(b, pb)
                if j > best_iou:
                    best_id, best_iou = pid, j
            if best_id != -1 and best_iou >= self.iou_thresh:
                ids[i] = best_id
                used_prev.add(best_id)
                self.box_by_id[best_id] = b
                self.ghost[best_id] = 0
        # new ids
        for i in range(len(boxes)):
            if ids[i] == -1:
                pid = self.next_id; self.next_id += 1
                ids[i] = pid
                self.box_by_id[pid] = boxes[i]
                self.ghost[pid] = 0
        # age & delete ghosts
        for pid in list(self.box_by_id.keys()):
            if pid not in used_prev and pid not in ids:
                self.ghost[pid] = self.ghost.get(pid,0)+1
                if self.ghost[pid] > self.max_ghost:
                    self.box_by_id.pop(pid, None)
                    self.ghost.pop(pid, None)
            else:
                self.ghost[pid] = 0
        return ids

# ---------- EMA smoothing for gaze points ----------
class PointSmoother:
    def __init__(self, beta=0.7):
        self.beta = beta
        self.prev: Dict[int, Tuple[float,float]] = {}

    def push(self, pid: int, pt: Optional[Tuple[int,int]]) -> Optional[Tuple[int,int]]:
        if pt is None:
            return None
        if pid not in self.prev:
            self.prev[pid] = (float(pt[0]), float(pt[1]))
        else:
            px, py = self.prev[pid]
            nx = self.beta*px + (1-self.beta)*pt[0]
            ny = self.beta*py + (1-self.beta)*pt[1]
            self.prev[pid] = (nx, ny)
        sx, sy = self.prev[pid]
        return (int(round(sx)), int(round(sy)))

# ---------- High-level renderer ----------
def render_present_frame(
    img_bgr: np.ndarray,
    face_boxes: List[List[int]],
    head_boxes: List[List[int]],
    points: Optional[List[Optional[Tuple[int,int]]]],
    inout: Optional[List[float]],
    angles_deg: Optional[List[float]],
    eye_flags: Optional[List[bool]],
    ids: List[int],
    names: Optional[Dict[int,str]] = None,
    *,
    # NEW: probability-based eye contact (preferred)
    eye_probs: Optional[List[float]] = None,
    prob_thr: float = 0.5,
) -> np.ndarray:
    """Draw everything: pills, boxes, arrows, tiny eye text.

    - If `eye_probs` is provided, shows "eye: YES/NO (p=…)".
    - Else if `angles_deg/eye_flags` provided, shows "eye: YES/NO (ang=…)".
    - Arrows are drawn only when in-frame >= ARROW_HIDE_THRESH.
    """
    vis = img_bgr.copy()
    H, W = vis.shape[:2]

    N = len(face_boxes)
    # normalize list lengths
    if points is None:
        points = [None]*N

    # build arrow destinations (respect in-frame)
    arrow_points: List[Optional[Tuple[int,int]]] = []
    for i in range(N):
        pr = inout[i] if (inout is not None and i < len(inout)) else 1.0
        pt = points[i] if (points is not None and i < len(points)) else None
        arrow_points.append(pt if (pt is not None and pr >= ARROW_HIDE_THRESH) else None)

    # draw arrows first (under boxes/text)
    for i in range(N):
        pid = ids[i] if i < len(ids) else i
        dst = arrow_points[i]
        if dst is None: continue
        src = _center(head_boxes[i] if i < len(head_boxes) else face_boxes[i])
        col = _color_for_id(pid)
        cv2.arrowedLine(vis, src, dst, col, ARROW_THICKNESS, tipLength=ARROW_TIP)

    # draw boxes and labels
    for i in range(N):
        pid = ids[i] if i < len(ids) else i
        col = _color_for_id(pid)
        x1,y1,x2,y2 = face_boxes[i]

        # face box
        cv2.rectangle(vis, (x1,y1), (x2,y2), col, BOX_THICKNESS, cv2.LINE_AA)

        # nameplate pill above the face
        name = names.get(pid, f"Guest {pid}") if names else f"Guest {pid}"
        inprob = inout[i] if (inout is not None and i < len(inout)) else None
        pill_text = f"{name} · in={inprob:.2f}" if inprob is not None else name
        vis = _draw_pill(vis, pill_text, (x1, max(0,y1-8)), bg_bgr=col, txt_bgr=(0,0,0), scale=SMALL_FONT)

        # tiny eye text below
        if eye_probs is not None and i < len(eye_probs):
            p = float(eye_probs[i])
            ok = p >= prob_thr
            eye_txt = f"eye: {'YES' if ok else 'NO '} (p={p:.2f})"
            vis = _draw_pill(vis, eye_txt, (x1, y2 + 18), bg_bgr=(30,30,30),
                             txt_bgr=(0,255,0) if ok else (0,0,255), scale=TINY_FONT)
        elif angles_deg is not None and eye_flags is not None and i < len(angles_deg) and i < len(eye_flags):
            ang = float(angles_deg[i]); ok = bool(eye_flags[i])
            eye_txt = (f"eye: {'YES' if ok else 'NO '} (ang={ang:.1f} deg)") if ang < 179.9 else "eye: N/A"
            vis = _draw_pill(vis, eye_txt, (x1, y2 + 18), bg_bgr=(30,30,30),
                             txt_bgr=(0,255,0) if ok and ang<179.9 else ((0,0,255) if ang<179.9 else (200,200,200)),
                             scale=TINY_FONT)
        else:
            vis = _draw_pill(vis, "eye: N/A", (x1, y2 + 18), bg_bgr=(30,30,30),
                             txt_bgr=(200,200,200), scale=TINY_FONT)
    return vis

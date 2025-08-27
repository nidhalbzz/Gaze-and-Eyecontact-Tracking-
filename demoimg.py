# demoimg.py
# Single-image presentation overlay (NO heatmap). Uses ux_present for visuals.

import argparse
from pathlib import Path
import cv2
from PIL import Image

from config import (
    GAZELLE_CKPT, USE_YOLO, YOLO_FACE_WEIGHTS,
    ENLARGE_BOX_X, ENLARGE_BOX_Y
)
from detectors import MediaPipeFaceDetector, YOLOFaceDetector
from gazelle_runner import GazelleGazeModel
from visualize import heatmap_argmax_xy, enlarge_box
from eye_contact_rt import EyeContactRT
from ux_present import render_present_frame

def to_norm_bbox(px_box, W, H):
    x1, y1, x2, y2 = px_box
    return (x1 / W, y1 / H, x2 / W, y2 / H)

def resolve_eye_weights(arg_path: str) -> Path:
    cand = Path(arg_path) if arg_path else None
    if cand is None or str(cand).strip() in ("", ".", "./") or (cand.exists() and cand.is_dir()) or not cand.exists():
        for p in (Path("models/contact_columbia.pt"), Path("models/contact_columbia_clean.pt")):
            if p.is_file(): return p
        raise FileNotFoundError("Place weights at models/contact_columbia.pt (or pass --eye-weights)")
    return cand

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--save", default="out.jpg")
    ap.add_argument("--eye-weights", default="")
    ap.add_argument("--eye-thr", type=float, default=0.50)
    ap.add_argument("--arrow-min-conf", type=float, default=0.20)  # kept for config parity; handled in ux_present
    args = ap.parse_args()

    frame_bgr = cv2.imread(str(args.image))
    if frame_bgr is None:
        raise RuntimeError(f"Failed to read image: {args.image}")
    H, W = frame_bgr.shape[:2]

    # detector (your detectors/)
    detector = YOLOFaceDetector(YOLO_FACE_WEIGHTS) if USE_YOLO else MediaPipeFaceDetector()
    face_boxes = detector(frame_bgr)  # [(x1,y1,x2,y2), ...]
    if not face_boxes:
        print("No face detected.")
        cv2.imwrite(args.save, frame_bgr); print(f"Saved to {args.save}")
        return

    # head boxes for arrows
    head_boxes = [enlarge_box(b, W, H, ENLARGE_BOX_X, ENLARGE_BOX_Y) for b in face_boxes]

    # eye-contact probs
    eye_rt = EyeContactRT(str(resolve_eye_weights(args.eye_weights)), thresh=args.eye_thr)
    eye_probs = eye_rt.predict_probs(frame_bgr, face_boxes)

    # gazelle arrows (no heatmap overlay)
    points, inout = None, None
    try:
        model = GazelleGazeModel(GAZELLE_CKPT)
        pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).convert("RGB")
        norm_boxes = [to_norm_bbox(b, W, H) for b in head_boxes]
        heatmap, inout_t = model.infer(pil, norm_boxes)
        points = []
        for i in range(heatmap.shape[0]):
            hm = heatmap[i].detach().cpu().numpy()
            x, y = heatmap_argmax_xy(hm, W, H)
            points.append((x, y))
        inout = [float(inout_t[i].item()) for i in range(heatmap.shape[0])] if inout_t is not None else None
    except Exception as e:
        print(f"[gazelle] disabled: {e}")

    # IDs for colors (static for image)
    ids = list(range(len(face_boxes)))

    vis = render_present_frame(
        frame_bgr, face_boxes, head_boxes,
        points, inout,
        angles_deg=None, eye_flags=None, ids=ids, names=None,
        eye_probs=eye_probs, prob_thr=eye_rt.thresh,
    )
    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.save, vis)
    print(f"Saved to {args.save}")

if __name__ == "__main__":
    main()

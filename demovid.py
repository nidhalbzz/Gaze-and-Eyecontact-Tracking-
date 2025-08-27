# demovid.py
# Video overlay using ux_present visuals. Stable IDs + smoothed gaze points.

import argparse, time
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
from ux_present import render_present_frame, SimpleIOUTracker, PointSmoother

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
    ap.add_argument("--source", default="0", help="camera index or video path")
    ap.add_argument("--save", default="")
    ap.add_argument("--max_fps", type=float, default=30.0)
    ap.add_argument("--eye-weights", default="")
    ap.add_argument("--eye-thr", type=float, default=0.50)
    args = ap.parse_args()

    src = 0 if (args.source.isdigit() and len(args.source) < 4) else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"Cannot open source: {args.source}"); return
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    fps_in = cap.get(cv2.CAP_PROP_FPS) or args.max_fps
    print(f"[cap] {W}x{H} @ {fps_in:.1f} fps")

    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(args.save, fourcc, min(args.max_fps, max(1.0, fps_in)), (W, H))
        print("[save]", args.save)

    # detectors
    detector = YOLOFaceDetector(YOLO_FACE_WEIGHTS) if USE_YOLO else MediaPipeFaceDetector()

    # eye-contact model
    eye_rt = EyeContactRT(str(resolve_eye_weights(args.eye_weights)), thresh=args.eye_thr)

    # gazelle
    gazelle = None
    try:
        gazelle = GazelleGazeModel(GAZELLE_CKPT)
        print("[gazelle] loaded; arrows enabled")
    except Exception as e:
        print(f"[gazelle] disabled: {e}")

    # tracking + smoothing
    tracker = SimpleIOUTracker(iou_thresh=0.35, max_ghost=20)
    smoother = PointSmoother(beta=0.7)

    # fps limiter
    min_dt = 1.0 / max(1.0, args.max_fps)
    t_prev = 0.0

    while True:
        ok, frame = cap.read()
        if not ok: break

        now = time.time()
        if now - t_prev < min_dt:
            time.sleep(max(0.0, min_dt - (now - t_prev)))
        t_prev = time.time()

        faces = detector(frame)  # [(x1,y1,x2,y2), ...]
        vis = frame.copy()

        if faces:
            # stable IDs
            ids = tracker.update(faces)

            # eye-contact probs (index-aligned with faces list)
            probs = eye_rt.predict_probs(frame, faces)

            # arrows via Gazelle -> smoothed per id
            points, inout = None, None
            if gazelle is not None:
                heads = [enlarge_box(b, W, H, ENLARGE_BOX_X, ENLARGE_BOX_Y) for b in faces]
                try:
                    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGB")
                    nboxes = [to_norm_bbox(b, W, H) for b in heads]
                    heatmap, inout_t = gazelle.infer(pil, nboxes)

                    raw_pts = []
                    for i in range(heatmap.shape[0]):
                        hm = heatmap[i].detach().cpu().numpy()
                        x, y = heatmap_argmax_xy(hm, W, H)
                        raw_pts.append((x, y))
                    # smooth by persistent id
                    smoothed = []
                    for i, pid in enumerate(ids):
                        pt = raw_pts[i] if i < len(raw_pts) else None
                        smoothed.append(smoother.push(pid, pt))
                    points = smoothed
                    inout = [float(inout_t[i].item()) for i in range(heatmap.shape[0])] if inout_t is not None else None
                except Exception:
                    points, inout = None, None

            # head boxes for beams
            heads = [enlarge_box(b, W, H, ENLARGE_BOX_X, ENLARGE_BOX_Y) for b in faces]

            vis = render_present_frame(
                vis, faces, heads, points, inout,
                angles_deg=None, eye_flags=None, ids=ids, names=None,
                eye_probs=probs, prob_thr=eye_rt.thresh,
            )

        cv2.imshow("gaze_target_demo (no heatmap)", vis)
        if writer: writer.write(vis)
        if (cv2.waitKey(1) & 0xFF) in (27, ord('q')): break

    cap.release()
    if writer: writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

from pathlib import Path

# ====== Paths to models ======
# Gaze-LLE checkpoint (make sure the file exists in checkpoints/)
GAZELLE_CKPT = Path("checkpoints/gazelle_dinov2_vitl14_inout.pt")

# Face detector: start with MediaPipe (no extra weights needed).
# If you want YOLO later, download yolov8n-face.pt into checkpoints/ and set USE_YOLO=True
USE_YOLO = True
YOLO_FACE_WEIGHTS = Path("checkpoints/yolov8n-face-lindevs.pt")

# ====== Runtime settings ======
DEVICE = "cuda"   # "cuda" if torch.cuda.is_available(), else "cpu"
IMG_SIZE = 448
INOUT_THRESHOLD = 0.5

# Visualization tweaks
HEATMAP_ALPHA = 0.55
DRAW_ARGMAX_DOT = True
DOT_RADIUS = 6

# Detection tweak: expand face box to approximate head box
ENLARGE_BOX_X = 0.15
ENLARGE_BOX_Y = 0.20



# --- looking-into-camera heuristic ---
LOOK_CAM_DISTANCE_FRAC = 0.25   # how close the target must be to the eye anchor (fraction of head-box diagonal)
LOOK_CAM_INFRAME_MAX   = 0.5    # if inout head is present: treat as "out-of-frame" when in-frame prob < 0.5
LABEL_TEXT             = "LOOKING INTO THE CAMERA"
LABEL_FONT_SCALE       = 0.8
LABEL_THICKNESS        = 2
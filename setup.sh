#!/usr/bin/env bash
set -e

# ---- Python env (conda or venv) is recommended, but optional here ----
python3 -m venv .venv || python -m venv .venv || py -3 -m venv .venv
source .venv/Scripts/activate  # Windows venv uses Scripts/, not bin/
python -m pip install -U pip

# ---- Base deps ----
pip install -r requirements.txt

# ---- Clone & install Gaze-LLE (a.k.a. gazelle) ----
if [ ! -d "gazelle" ]; then
  git clone https://github.com/fkryan/gazelle.git
fi
pip install -e ./gazelle

echo "Setup complete.

Next:
1) Download a Gaze-LLE checkpoint (e.g., gazelle_dinov2_vitl14_inout.pt) and update config.py.
2) If using YOLO face, provide a face model (e.g., yolov8n-face.pt) and update config.py.
3) Run:
   python demo_image.py --image path/to/img.jpg
   # or
   python demo_video.py --source 0
"

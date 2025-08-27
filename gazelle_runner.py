from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T

from config import IMG_SIZE, DEVICE

# Try to use gazelle's official transform if available; else fallback to ImageNet/DINOv2 norm.
def _build_transform():
    try:
        # If the package exposes a transform builder, prefer it.
        from gazelle.utils import build_transform  # may exist in future versions
        return build_transform(size=IMG_SIZE)
    except Exception:
        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]
        return T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

class GazelleGazeModel:
    """
    Thin wrapper around the Gaze-LLE model:
      Inputs:
        - image PIL
        - list of bboxes (normalized [xmin,ymin,xmax,ymax]) for one image
      Outputs:
        - heatmap: [P, 64, 64] float (P = num persons)
        - inout: [P] float (or None if model has no in/out head)
    """

    def __init__(self, ckpt_path: Path, model_name="gazelle_dinov2_vitl14_inout"):
        from gazelle.model import get_gazelle_model

        self.model, _ = get_gazelle_model(model_name)  # returns (model, transform?) in repo
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        # Some repos save strict/non-strict; try both.
        try:
            self.model.load_gazelle_state_dict(state)
        except Exception:
            self.model.load_state_dict(state, strict=False)

        self.model.eval().to(DEVICE)
        self.transform = _build_transform()

    @torch.no_grad()
    def infer(self, image_pil: Image.Image, norm_bboxes):
        """
        image_pil: PIL RGB
        norm_bboxes: list of (xmin, ymin, xmax, ymax) in [0,1] (head boxes)
        """
        if len(norm_bboxes) == 0:
            return None, None

        images = self.transform(image_pil).unsqueeze(0).to(DEVICE)     # [1, 3, 448, 448]
        inp = {"images": images, "bboxes": [norm_bboxes]}              # one image, P persons

        out = self.model(inp)
        heatmap = out["heatmap"][0]                                    # [P, 64, 64]
        inout = out.get("inout")
        if inout is not None:
            inout = inout[0]                                           # [P]
        return heatmap, inout

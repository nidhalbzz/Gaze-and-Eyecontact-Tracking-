# eye_contact_rt.py
import cv2, torch, numpy as np
import torchvision.transforms as T
from torchvision import models

IMG_SIZE = 224

class EyeContactRT:
    def __init__(self, weights_path="models/contact_columbia.pt", device=None, thresh=0.5):
        self.device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        in_feats = m.classifier[0].in_features
        m.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_feats,256), torch.nn.Hardswish(), torch.nn.Dropout(0.2), torch.nn.Linear(256,1)
        )
        ckpt = torch.load(weights_path, map_location="cpu")
        sd = ckpt["model"]
        # allow both 'clean' and compiled checkpoints
        if any(k.startswith("_orig_mod.") for k in sd.keys()):
            sd = {k.replace("_orig_mod.",""): v for k,v in sd.items()}
        m.load_state_dict(sd, strict=True)
        m.eval().to(self.device)
        self.model = m
        self.thresh = float(thresh)
        self.tf = T.Compose([
            T.ToPILImage(), T.Resize(IMG_SIZE), T.CenterCrop(IMG_SIZE),
            T.ToTensor(), T.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ])

    def _crop_rgb224(self, bgr, box_xyxy):
        x1,y1,x2,y2 = [int(round(v)) for v in box_xyxy]
        x1,y1 = max(0,x1), max(0,y1)
        x2,y2 = min(bgr.shape[1]-1,x2), min(bgr.shape[0]-1,y2)
        if x2<=x1 or y2<=y1:
            return np.zeros((IMG_SIZE,IMG_SIZE,3), np.uint8)
        crop = cv2.resize(bgr[y1:y2, x1:x2], (IMG_SIZE,IMG_SIZE), interpolation=cv2.INTER_AREA)
        return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)

    @torch.no_grad()
    def predict_probs(self, frame_bgr, face_boxes_xyxy):
        if not face_boxes_xyxy:
            return []
        xs = [self.tf(self._crop_rgb224(frame_bgr, b)) for b in face_boxes_xyxy]
        x = torch.stack(xs, 0).to(self.device)
        p = torch.sigmoid(self.model(x)).view(-1).cpu().numpy().tolist()
        return p

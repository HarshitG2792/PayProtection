"""python predict.py path/to/screenshot.jpg  [--ckpt checkpoints/best.pt]"""
import argparse
import torch
from PIL import Image
from ela import to_tensor_array, compute_ela
from model import TamperNet

_model = None


def load_model(ckpt="checkpoints/best.pt"):
    global _model
    if _model is None:
        _model = TamperNet()
        _model.load_state_dict(torch.load(ckpt, map_location="cpu"))
        _model.eval()
    return _model


def predict_image(img: Image.Image, ckpt="checkpoints/best.pt") -> float:
    """Returns probability (0..1) that the screenshot has been edited."""
    m = load_model(ckpt)
    x = torch.from_numpy(to_tensor_array(img)).unsqueeze(0)
    with torch.no_grad():
        return torch.sigmoid(m(x)).item()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("image"); ap.add_argument("--ckpt", default="checkpoints/best.pt")
    ap.add_argument("--ela_out", default="ela_result.png")
    a = ap.parse_args()
    img = Image.open(a.image)
    p = predict_image(img, a.ckpt)
    compute_ela(img).save(a.ela_out)
    print(f"P(tampered) = {p:.3f} -> {'EDITED / SUSPICIOUS' if p > 0.5 else 'looks genuine'}   (ELA map: {a.ela_out})")

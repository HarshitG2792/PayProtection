"""Error Level Analysis (ELA) + image preprocessing.

Idea: re-save the image as JPEG at a fixed quality and look at the difference.
Regions that were pasted / edited after the original compression respond
differently, so they "light up" in the ELA map. The CNN sees RGB + ELA together.
"""
import io
import numpy as np
from PIL import Image, ImageChops

IMG_H, IMG_W = 448, 256          # fixed network input size (portrait screenshots)
ELA_QUALITY = 90


def compute_ela(img: Image.Image, quality: int = ELA_QUALITY) -> Image.Image:
    img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    resaved = Image.open(buf).convert("RGB")
    diff = ImageChops.difference(img, resaved)
    arr = np.asarray(diff).astype(np.float32)
    mx = max(arr.max(), 1.0)
    arr = arr * (255.0 / mx)                 # normalise so faint edits are visible
    return Image.fromarray(arr.clip(0, 255).astype(np.uint8))


def to_tensor_array(img: Image.Image) -> np.ndarray:
    """PIL image -> float32 array (6, H, W): RGB + ELA, each scaled to [0,1]."""
    img = img.convert("RGB")
    ela = compute_ela(img)                    # ELA at ORIGINAL resolution (important)
    img = img.resize((IMG_W, IMG_H), Image.BILINEAR)
    ela = ela.resize((IMG_W, IMG_H), Image.BILINEAR)
    x = np.concatenate([np.asarray(img), np.asarray(ela)], axis=2).astype(np.float32) / 255.0
    return x.transpose(2, 0, 1)

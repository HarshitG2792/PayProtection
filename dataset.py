import os, random
import torch
from torch.utils.data import Dataset
from PIL import Image
from ela import to_tensor_array

EXT = (".jpg", ".jpeg", ".png")


class ScreenshotDataset(Dataset):
    """Folder layout: root/genuine/*.jpg (label 0), root/tampered/*.jpg (label 1)."""

    def __init__(self, root):
        self.items = []
        for label, cls in enumerate(["genuine", "tampered"]):
            d = os.path.join(root, cls)
            for f in sorted(os.listdir(d)):
                if f.lower().endswith(EXT):
                    self.items.append((os.path.join(d, f), label))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        x = torch.from_numpy(to_tensor_array(Image.open(path)))
        return x, torch.tensor(float(y))

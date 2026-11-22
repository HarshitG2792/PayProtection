import torch
import torch.nn as nn


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class TamperNet(nn.Module):
    """Small CNN on 6 channels (RGB + ELA). Output: 1 logit, >0 means 'tampered'."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(block(6, 32), block(32, 64), block(64, 128), block(128, 192), block(192, 256))
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)   # max-pool helps: an edit is a SMALL local region
        self.head = nn.Sequential(nn.Dropout(0.4), nn.Linear(512, 128), nn.ReLU(inplace=True), nn.Linear(128, 1))

    def forward(self, x):
        f = self.features(x)
        z = torch.cat([self.avg(f).flatten(1), self.max(f).flatten(1)], 1)
        return self.head(z).squeeze(1)

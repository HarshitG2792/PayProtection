"""python train.py --data data --epochs 15"""
import argparse, os
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tqdm import tqdm
from dataset import ScreenshotDataset
from model import TamperNet


def evaluate(model, loader, dev):
    model.eval(); ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps += torch.sigmoid(model(x.to(dev))).cpu().tolist(); ys += y.tolist()
    pred = [int(p > 0.5) for p in ps]
    return dict(acc=accuracy_score(ys, pred), prec=precision_score(ys, pred, zero_division=0),
                rec=recall_score(ys, pred, zero_division=0), f1=f1_score(ys, pred, zero_division=0),
                auc=roc_auc_score(ys, ps) if len(set(ys)) > 1 else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data"); ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--out", default="checkpoints")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", dev)
    tr = DataLoader(ScreenshotDataset(f"{a.data}/train"), a.bs, shuffle=True, num_workers=a.workers)
    va = DataLoader(ScreenshotDataset(f"{a.data}/val"), a.bs, num_workers=a.workers)
    te = DataLoader(ScreenshotDataset(f"{a.data}/test"), a.bs, num_workers=a.workers)
    model = TamperNet().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    lossf = nn.BCEWithLogitsLoss()
    os.makedirs(a.out, exist_ok=True); best = -1
    for ep in range(1, a.epochs + 1):
        model.train(); tot = 0
        for x, y in tqdm(tr, desc=f"epoch {ep}/{a.epochs}"):
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(); loss = lossf(model(x), y); loss.backward(); opt.step(); tot += loss.item() * len(y)
        sched.step()
        m = evaluate(model, va, dev)
        print(f"  train_loss={tot/len(tr.dataset):.4f}  val: " + "  ".join(f"{k}={v:.3f}" for k, v in m.items()))
        if m["f1"] > best:
            best = m["f1"]; torch.save(model.state_dict(), f"{a.out}/best.pt"); print("  saved best.pt")
    model.load_state_dict(torch.load(f"{a.out}/best.pt", map_location=dev))
    print("TEST:", {k: round(v, 3) for k, v in evaluate(model, te, dev).items()})


if __name__ == "__main__":
    main()

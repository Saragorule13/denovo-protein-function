"""Phase 4c: 1D-CNN on frozen ESM sequences (from repo root: python src/training/train_cnn.py).

Conv(480->256)x2 + mean||max pool + MLP -> 65 ECs. BCE + pos_weight, early-stop on
val macro-F1. Out: checkpoints/cnn_baseline.pt, results row + scores_cnn_test.npy
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.metrics import full_report, tune_threshold

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SeqDS(Dataset):
    def __init__(self, ids, embs, y_of):
        self.ids = ids
        self.embs = embs  # dict acc -> fp16 tensor
        self.y_of = y_of

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        a = self.ids[i]
        return self.embs[a].float(), torch.tensor(self.y_of[a]).float()


def collate(b):
    L = max(e.size(0) for e, _ in b)
    X = torch.zeros(len(b), L, 480)
    M = torch.zeros(len(b), L, dtype=torch.bool)
    Y = torch.stack([y for _, y in b])
    for i, (e, _) in enumerate(b):
        X[i, :e.size(0)] = e
        M[i, :e.size(0)] = True
    return X, M, Y


class CNN(nn.Module):
    def __init__(self, n_cls=65):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(480, 256, 7, padding=3), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(256, 256, 7, padding=3), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.2))
        self.head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(),
                                  nn.Dropout(0.3), nn.Linear(256, n_cls))

    def forward(self, x, mask):
        h = self.conv(x.transpose(1, 2))  # [B,256,L]
        m = mask.unsqueeze(1).float()
        mean = (h * m).sum(2) / m.sum(2).clamp(min=1)
        mx = (h + (1 - m) * -1e4).amax(2)
        return self.head(torch.cat([mean, mx], 1))


@torch.no_grad()
def predict(model, loader):
    model.eval()
    S, Y = [], []
    for X, M, y in loader:
        S.append(model(X.to(DEVICE), M.to(DEVICE)).cpu())
        Y.append(y)
    return torch.cat(S).numpy(), torch.cat(Y).numpy()


def main(epochs=50, patience=8, batch=32, lr=3e-4):
    print(f"device={DEVICE}", flush=True)
    torch.manual_seed(42)
    clean = pd.read_csv(PROC / "clean.csv")
    lmap = json.load(open(PROC / "label_map.json"))
    splits = json.load(open(PROC / "splits.json"))
    y_of = {}
    for r in clean.itertuples():
        v = np.zeros(len(lmap))
        for e in str(r.ec_top).split(";"):
            v[lmap[e]] = 1
        y_of[r.accession] = v
    print("preloading embeddings (fp16)...", flush=True)
    embs = {a: torch.load(PROC / "embeddings" / f"{a}.pt", weights_only=True)
            for ids in splits.values() for a in ids}
    tr = DataLoader(SeqDS(splits["train"], embs, y_of), batch, True, collate_fn=collate)
    va = DataLoader(SeqDS(splits["val"], embs, y_of), 64, False, collate_fn=collate)
    te = DataLoader(SeqDS(splits["test"], embs, y_of), 64, False, collate_fn=collate)
    Ytr = np.stack([y_of[a] for a in splits["train"]])
    pw = torch.from_numpy(((len(Ytr) - Ytr.sum(0)) / Ytr.sum(0).clip(min=1)).astype(np.float32))
    model = CNN(len(lmap)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw.to(DEVICE))
    best, wait, ckpt = -1, 0, ROOT / "checkpoints" / "cnn_baseline.pt"
    ckpt.parent.mkdir(exist_ok=True)
    for ep in range(epochs):
        model.train()
        tot, t0 = 0, time.time()
        for X, M, y in tr:
            opt.zero_grad()
            loss = loss_fn(model(X.to(DEVICE), M.to(DEVICE)), y.to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(X)
        Sv, Yv = predict(model, va)
        f1 = full_report(Yv, 1 / (1 + np.exp(-Sv)), 0.5)["macro-F1"]
        print(f"ep{ep + 1}: loss={tot / len(tr.dataset):.4f} val-macroF1={f1:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if f1 > best:
            best, wait = f1, 0
            torch.save(model.state_dict(), ckpt)
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    Sv, Yv = predict(model, va)
    Pv = 1 / (1 + np.exp(-Sv))
    t = tune_threshold(Yv, Pv)
    Ste, Yte = predict(model, te)
    Pte = 1 / (1 + np.exp(-Ste))
    np.save(ROOT / "results" / "scores_cnn_test.npy", Pte)
    rep = {"thr": t["thr"], **full_report(Yte, Pte, t["thr"])}
    print({k: round(v, 4) for k, v in rep.items()})
    mp = ROOT / "results" / "metrics_baselines.csv"
    old = pd.read_csv(mp)
    row = pd.DataFrame([{"model": "cnn-ESMseq",
                         **{k: round(v, 4) for k, v in rep.items()}}])
    pd.concat([old, row], ignore_index=True).drop_duplicates("model", keep="last") \
      .to_csv(mp, index=False)
    print(f"best val-macroF1={best:.4f} -> {mp}")


if __name__ == "__main__":
    main()
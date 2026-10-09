"""Phase 5: train GCN / GAT / ProteinGT (from repo root).

  python src/training/train_gnn.py --model gt --overfit     # M4: 32 graphs -> F1~1
  python src/training/train_gnn.py --model {gcn,gat,gt} [--seed 42]

BCE + train pos_weight, AdamW, AMP, clip 1.0, early-stop patience 12 on val macro-F1.
"""
import argparse
import json
import warnings
warnings.filterwarnings("ignore", message=".*torch-scatter.*")
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.graph_dataset import GraphDS, make_loaders
from src.evaluation.metrics import full_report, tune_threshold
from src.models.gnn import make

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_all(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def pos_weight(train_ids):
    clean = pd.read_csv(PROC / "clean.csv")
    lmap = json.load(open(PROC / "label_map.json"))
    Y = np.zeros((len(train_ids), len(lmap)))
    rmap = {r.accession: r for r in clean.itertuples()}
    for i, a in enumerate(train_ids):
        for e in str(rmap[a].ec_top).split(";"):
            Y[i, lmap[e]] = 1
    return torch.from_numpy(((len(Y) - Y.sum(0)) / Y.sum(0).clip(min=1)).astype(np.float32)), Y


@torch.no_grad()
def predict(model, loader, name):
    model.eval()
    S, Y, ids = [], [], []
    for b in loader:
        b = b.to(DEVICE)
        kw = {"edge_attr": b.edge_attr} if name != "gcn" else {"edge_weight": b.edge_weight}
        S.append(model(b.x, b.edge_index, b.batch, **kw).cpu())
        Y.append(b.y.cpu())
        ids += list(b.acc)
    return torch.cat(S).numpy(), torch.cat(Y).numpy(), ids


def main(model, seed, epochs, patience, batch, lr, overfit, resume=False):
    seed_all(seed)
    print(f"model={model} seed={seed} device={DEVICE}", flush=True)
    lds, _ = make_loaders(batch)
    if overfit:
        ids = lds["train"].dataset.ids[:32]
        sc = lds["train"].dataset.scaler
        lds = {"train": DataLoader(GraphDS(ids, sc), batch_size=8, shuffle=True)}
        epochs, patience = 50, 100
    pw, Ytr = pos_weight(lds["train"].dataset.ids)
    net = make(model, n_cls=65).to(DEVICE)
    print(f"params: {sum(p.numel() for p in net.parameters())}", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw.to(DEVICE))
    scaler = torch.amp.GradScaler("cuda") if DEVICE.type == "cuda" else None
    ckpt = ROOT / "checkpoints" / f"{model}_s{seed}.pt"
    best, wait, start = -1, 0, 0
    if resume and ckpt.exists():
        d = torch.load(ckpt, map_location="cpu")
        net.load_state_dict(d["state"])
        best, start = d.get("best", -1), d.get("epoch", 0) + 1
        print(f"resumed {ckpt} at ep{start} best={best:.4f}", flush=True)
    for ep in range(start, epochs):
        net.train()
        tot, t0 = 0, time.time()
        for b in lds["train"]:
            b = b.to(DEVICE)
            kw = {"edge_attr": b.edge_attr} if model != "gcn" else \
                 {"edge_weight": b.edge_weight}
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=scaler is not None):
                loss = loss_fn(net(b.x, b.edge_index, b.batch, **kw), b.y)
            (scaler.scale(loss).backward() if scaler else loss.backward())
            if scaler:
                scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            (scaler.step(opt) if scaler else opt.step())
            if scaler:
                scaler.update()
            tot += loss.item() * b.num_graphs
        if overfit:
            S, Y, _ = predict(net, lds["train"], model)
            r = full_report(Y, torch.sigmoid(torch.from_numpy(S)).numpy(), 0.5)
            f1 = r["micro-F1"]
            print(f"ep{ep + 1}: loss={tot / len(lds['train'].dataset):.4f} "
                  f"train-microF1={f1:.4f} macro={r['macro-F1']:.4f} ({time.time() - t0:.0f}s)", flush=True)
            best = f1
            continue
        Sv, Yv, _ = predict(net, lds["val"], model)
        f1 = full_report(Yv, torch.sigmoid(torch.from_numpy(Sv)).numpy(), 0.5)["macro-F1"]
        print(f"ep{ep + 1}: loss={tot / len(lds['train'].dataset):.4f} "
              f"val-macroF1={f1:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if f1 > best:
            best, wait = f1, 0
            torch.save({"name": model, "seed": seed, "best": best,
                        "epoch": ep, "state": net.state_dict()}, ckpt)
        else:
            wait += 1
            if wait >= patience:
                break
    if overfit:
        print(f"M4 overfit-32 train-microF1={best:.4f} "
              f"({'PASS' if best > 0.9 else 'FAIL: investigate capacity/flow'})")
        return
    net.load_state_dict(torch.load(ckpt, map_location=DEVICE)["state"])
    Sv, Yv, _ = predict(net, lds["val"], model)
    Pv = torch.sigmoid(torch.from_numpy(Sv)).numpy()
    t = tune_threshold(Yv, Pv)
    Ste, Yte, ids = predict(net, lds["test"], model)
    Pte = torch.sigmoid(torch.from_numpy(Ste)).numpy()
    np.save(ROOT / "results" / f"scores_{model}_s{seed}_test.npy", Pte)
    pd.DataFrame({"acc": ids, "split": "test"}).to_csv(
        ROOT / "results" / f"ids_{model}_s{seed}_test.csv", index=False)
    rep = {"thr": t["thr"], **full_report(Yte, Pte, t["thr"])}
    print({k: round(v, 4) for k, v in rep.items()})
    mp = ROOT / "results" / "metrics_baselines.csv"
    old = pd.read_csv(mp)
    row = pd.DataFrame([{**{"model": f"{model}-s{seed}"},
                         **{k: round(v, 4) for k, v in rep.items()}}])
    pd.concat([old, row], ignore_index=True).drop_duplicates("model", keep="last") \
      .to_csv(mp, index=False)
    print(f"best val-macroF1={best:.4f} -> {mp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gt", choices=["gcn", "gat", "gt"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--overfit", action="store_true")
    ap.add_argument("--resume", action="store_true")
    main(**vars(ap.parse_args()))
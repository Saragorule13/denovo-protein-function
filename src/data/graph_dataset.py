"""PyG dataset over cached graphs + train-fitted standardization of raw columns.

Raw columns (physchem 20:27, conf 508) are standardized with TRAIN-only stats
(saved to data/processed/scaler.pt). Everything else (one-hot, ESM, relpos) untouched.
"""
import json
from pathlib import Path

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RAW_COLS = list(range(20, 27)) + [508]


class GraphDS(Dataset):
    def __init__(self, ids, scaler=None):
        super().__init__()
        self.ids = [a for a in ids if (PROC / "graphs" / f"{a}.pt").exists()]
        self.scaler = scaler

    def len(self):
        return len(self.ids)

    def get(self, i):
        g = torch.load(PROC / "graphs" / f"{self.ids[i]}.pt", weights_only=False)
        x = g["x"].float()
        if self.scaler is not None:
            mu, sd = self.scaler["mean"], self.scaler["std"]
            x[:, RAW_COLS] = (x[:, RAW_COLS] - mu) / sd
        ew = 1.0 / (1.0 + g["edge_attr"][:, 0])  # 1/(1+d) for GCN
        return Data(x=x, edge_index=g["edge_index"], edge_attr=g["edge_attr"].float(),
                    edge_weight=ew, y=g["y"].float().unsqueeze(0), acc=g["accession"])


def fit_scaler(train_ids):
    xs = []
    for a in train_ids:
        p = PROC / "graphs" / f"{a}.pt"
        if p.exists():
            xs.append(torch.load(p, weights_only=False)["x"][:, RAW_COLS].float())
    X = torch.cat(xs)
    sc = {"mean": X.mean(0), "std": X.std(0).clamp(min=1e-6), "cols": RAW_COLS}
    torch.save(sc, PROC / "scaler.pt")
    return sc


def make_loaders(batch=16):
    splits = json.load(open(PROC / "splits.json"))
    sc_path = PROC / "scaler.pt"
    sc = torch.load(sc_path, weights_only=False) if sc_path.exists() else None
    if sc is None:
        sc = fit_scaler(splits["train"])
        print(f"fitted scaler on train "
              f"({len(splits['train'])} ids) -> {sc_path}")
    lds = {}
    for k, b in (("train", batch), ("val", 32), ("test", 32)):
        ds = GraphDS(splits[k], sc)
        lds[k] = DataLoader(ds, batch_size=b, shuffle=(k == "train"))
        print(f"{k}: {len(ds)} graphs", flush=True)
    return lds, sc
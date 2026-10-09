"""Phase 4a: prior / LR / MLP baselines on mean-ESM (from repo root).

  python src/training/run_sklearn.py

In : clean.csv, splits.json, label_map.json, embeddings/
Out : results/metrics_baselines.csv (rows appended), results/scores_*_test.npy
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.metrics import full_report, tune_threshold

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"


def load_arrays():
    cache = PROC / "mean_esm.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        return {k: d[k] for k in d.files}
    clean = pd.read_csv(PROC / "clean.csv")
    lmap = json.load(open(PROC / "label_map.json"))
    splits = json.load(open(PROC / "splits.json"))
    row_of = {r.accession: r for r in clean.itertuples()}
    out = {}
    for sp, ids in splits.items():
        X = np.stack([torch.load(PROC / "embeddings" / f"{a}.pt",
                                 weights_only=True).float().numpy().mean(axis=0)
                      for a in ids])
        Y = np.zeros((len(ids), len(lmap)), dtype=int)
        for i, a in enumerate(ids):
            for e in str(row_of[a].ec_top).split(";"):
                if e in lmap:
                    Y[i, lmap[e]] = 1
        out[f"X_{sp}"], out[f"Y_{sp}"], out[f"ids_{sp}"] = X, Y, np.array(ids)
    np.savez_compressed(cache, **out)
    print(f"cached mean-ESM -> {cache}")
    return out


def save_row(rows, name, rep):
    rows.append({"model": name, **{k: round(v, 4) for k, v in rep.items()}})


def main():
    d = load_arrays()
    Xtr, Ytr, Xva, Yva, Xte, Yte = (d["X_train"], d["Y_train"], d["X_val"],
                                    d["Y_val"], d["X_test"], d["Y_test"])
    print(f"train={Xtr.shape} val={Xva.shape} test={Xte.shape} "
          f"pos_rate={Ytr.mean():.4f}")
    rows = []
    # 1. prior (train frequencies as scores)
    prior = np.tile(Ytr.mean(axis=0), (len(Yte), 1))
    t = tune_threshold(Yva, np.tile(Ytr.mean(axis=0), (len(Yva), 1)))
    save_row(rows, "prior", {"thr": t["thr"],
                             **full_report(Yte, prior, t["thr"])})
    # 2. logistic regression (OvR, balanced)
    lr = OneVsRestClassifier(LogisticRegression(max_iter=2000, n_jobs=-1), n_jobs=-1)
    lr.fit(Xtr, Ytr)
    Ste = np.stack([e.decision_function(Xte) for e in lr.estimators_], axis=1)
    Sva = np.stack([e.decision_function(Xva) for e in lr.estimators_], axis=1)
    t = tune_threshold(Yva, Sva)
    np.save(ROOT / "results" / "scores_lr_test.npy", Ste)
    save_row(rows, "lr-meanESM", {"thr": t["thr"], **full_report(Yte, Ste, t["thr"])})
    # 3. MLP
    mlp = MLPClassifier(hidden_layer_sizes=(256,), early_stopping=True,
                        validation_fraction=0.1, n_iter_no_change=10,
                        max_iter=500, random_state=42, verbose=False)
    mlp.fit(Xtr, Ytr)
    Pte, Pva = mlp.predict_proba(Xte), mlp.predict_proba(Xva)
    t = tune_threshold(Yva, Pva)
    np.save(ROOT / "results" / "scores_mlp_test.npy", Pte)
    save_row(rows, "mlp-meanESM", {"thr": t["thr"], **full_report(Yte, Pte, t["thr"])})
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    mp = ROOT / "results" / "metrics_baselines.csv"
    old = pd.read_csv(mp) if mp.exists() else pd.DataFrame()
    pd.concat([old, res], ignore_index=True).drop_duplicates("model", keep="last") \
      .to_csv(mp, index=False)
    print(f"-> {mp}")


if __name__ == "__main__":
    main()
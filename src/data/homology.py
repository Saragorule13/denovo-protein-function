"""Phase 4b: homology baseline - transfer ECs from best MMseqs2 hit in train.

  python src/data/homology.py

Exports split FASTAs (LF endings), runs easy-search test->train in WSL,
transfers top-hit EC set, scores with src/evaluation/metrics.py.
In : clean.csv, splits.json | Out: results/homology_pred.csv + metrics row
"""
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.evaluation.metrics import full_report

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
HOM = PROC / "hom"
HOM.mkdir(parents=True, exist_ok=True)


def wsl(path):
    return "/mnt/c/" + str(path.resolve()).replace("C:\\", "").replace("\\", "/")


def write_fasta(ids, seq_of, name):
    p = HOM / name
    with open(p, "w", newline="") as f:  # LF only: MMseqs names must not carry \r
        for a in ids:
            f.write(f">{a}\n{seq_of[a]}\n")
    return p


def main():
    clean = pd.read_csv(PROC / "clean.csv")
    splits = json.load(open(PROC / "splits.json"))
    seq_of = dict(zip(clean["accession"], clean["sequence"]))
    ec_of = {r.accession: str(r.ec_top).split(";")
             for r in clean.itertuples()}
    lmap = json.load(open(PROC / "label_map.json"))
    q = write_fasta(splits["test"], seq_of, "test.faa")
    t = write_fasta(splits["train"], seq_of, "train.faa")
    out = HOM / "hits.m8"
    cmd = (f"~/tools/mmseqs/bin/mmseqs easy-search {wsl(q)} {wsl(t)} "
           f"{wsl(out)} {wsl(HOM)}/tmp --max-seqs 5 "
           f"--format-output 'query,target,evalue,bits' --threads 20")
    print("running:", cmd[:100], "...", flush=True)
    subprocess.run(["wsl", "-d", "Ubuntu", "--", "bash", "-lc", cmd], check=True)
    hits = pd.read_csv(out, sep="\t", header=None,
                       names=["query", "target", "evalue", "bits"])
    hits["query"] = hits["query"].str.strip()
    hits["target"] = hits["target"].str.strip()
    best = hits.sort_values("bits", ascending=False).drop_duplicates("query")
    bmap = dict(zip(best["query"], best["target"]))
    nC = len(lmap)
    Yte = np.zeros((len(splits["test"]), nC), int)
    Ste = np.zeros((len(splits["test"]), nC))
    for i, a in enumerate(splits["test"]):
        for e in ec_of[a]:
            Yte[i, lmap[e]] = 1
        h = bmap.get(a)
        if h:
            for e in ec_of.get(h, []):
                Ste[i, lmap[e]] = 1.0
    cov = sum(a in bmap for a in splits["test"]) / len(splits["test"])
    rep = full_report(Yte, Ste, 0.5)
    print(f"top-1 hit coverage={cov:.1%} P@1={rep['P@1']:.4f} "
          f"macro-F1={rep['macro-F1']:.4f} micro-F1={rep['micro-F1']:.4f}")
    pd.DataFrame({"model": ["homology-top1"],
                  **{k: [round(v, 4)] for k, v in
                     ({"thr": 0.5} | rep).items()}}).to_csv(
        ROOT / "results" / "metrics_homology.csv", index=False)
    pd.DataFrame(Ste, index=splits["test"]).to_csv(
        ROOT / "results" / "homology_pred.csv")


if __name__ == "__main__":
    main()
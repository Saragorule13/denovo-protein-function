"""Phase 2b: cluster splits + leakage audit (from repo root: python src/data/splits.py).

In : data/processed/clean.csv + data/processed/mm_cluster.tsv
Out : data/processed/splits.json, random_split.json, split_stats.csv
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data" / "processed"
RNG = np.random.default_rng(42)

clean = pd.read_csv(PROC / "clean.csv")
raw = (PROC / "mm_cluster.tsv").read_bytes().replace(b"\r", b"").decode()
pairs = [(p[0].strip(), p[1].strip()) for l in raw.split("\n")
         for p in [l.split("\t")] if len(p) >= 2]
clu = pd.DataFrame(pairs, columns=["rep", "member"])
clusters = clu.groupby("rep")["member"].apply(list).tolist()
sizes = sorted([len(c) for c in clusters], reverse=True)
print(f"proteins={len(clean)} clusters={len(clusters)} "
      f"singletons={sum(s == 1 for s in sizes)} largest={sizes[0]}")
plt.figure(); plt.hist(sizes, bins=50)
plt.xlabel("cluster size"); plt.ylabel("# clusters")
plt.title("MMseqs2 clusters (40% id, cov 80%)")
plt.savefig(ROOT / "results" / "phase2_clusters.png", dpi=120)

# stage 1: greedy cluster-to-split assignment targeting 70/15/15 by count
targets = {"train": 0.70, "val": 0.15, "test": 0.15}
N = len(clean)
goal = {k: v * N for k, v in targets.items()}
count = {k: 0 for k in targets}
assign = {}
ordered = sorted(clusters, key=len, reverse=True)
for c in ordered:
    k = min(targets, key=lambda k: (count[k] + len(c)) / goal[k])
    for m in c:
        assign[m] = k
    count[k] += len(c)

# stage 2: hill-climb whole-cluster moves to also balance EC level-1 mix
l1_of = {r.accession: str(r.ec_level1).split(";")
         for r in clean.itertuples()}
all_l1 = [l for v in l1_of.values() for l in v]
classes = sorted(set(all_l1))
glob = {c: all_l1.count(c) / len(all_l1) for c in classes}
clu_of = {}
for c in ordered:
    for m in c:
        clu_of[m] = tuple(c)


def total_cost(assign):
    n = {k: 0 for k in targets}
    f = {k: {c: 0 for c in classes} for k in targets}
    for m, k in assign.items():
        n[k] += 1
        for x in l1_of.get(m, []):
            f[k][x] += 1
    bal = sum((n[k] / N - targets[k]) ** 2 for k in targets)
    comp = sum(((f[k][x] / max(n[k], 1)) - glob[x]) ** 2
               for k in targets for x in classes)
    return bal + comp


cur = total_cost(assign)
for _ in range(5):
    moved = False
    for c in ordered:
        src = assign[c[0]]
        for dst in targets:
            if dst == src:
                continue
            for m in c:
                assign[m] = dst
            new = total_cost(assign)
            if new < cur - 1e-9:
                cur = new
                src = dst
                moved = True
            else:
                for m in c:
                    assign[m] = src
    if not moved:
        break
count = {k: sum(1 for v in assign.values() if v == k) for k in targets}
print(f"refined cost={cur:.5f}")
splits = {k: sorted([m for m, s in assign.items() if s == k]) for k in targets}
print({k: len(v) for k, v in splits.items()})

# audit 1: every clean accession assigned exactly once, no overlap
assert sorted(sum(splits.values(), [])) == sorted(clean["accession"].tolist()), \
    "assignment incomplete/overlapping!"
# audit 2: no cluster spans two splits
for c in clusters:
    assert len({assign[m] for m in c}) == 1, f"cluster spans splits: {c[:3]}"
print("audit passed: full coverage, zero overlap, zero cluster-spanning")

# audit 3: EC level-1 distribution per split (stratification check)
clean["split"] = clean["accession"].map(assign)
rows = (clean.assign(l1=clean["ec_level1"].str.split(";")).explode("l1")
        .groupby(["split", "l1"]).size().unstack(fill_value=0))
rows.to_csv(PROC / "split_stats.csv")
print(rows.to_string())
rows.div(rows.sum(axis=1), axis=0).plot(kind="bar", stacked=True, figsize=(8, 4))
plt.title("EC level-1 share per split"); plt.tight_layout()
plt.savefig(ROOT / "results" / "phase2_split_ec.png", dpi=120)

json.dump(splits, open(PROC / "splits.json", "w"))
# random-split twin for the appendix leakage-gap experiment (same ratios, seed 42)
ids = clean["accession"].sample(frac=1.0, random_state=42).tolist()
n1, n2 = int(0.70 * N), int(0.85 * N)
json.dump({"train": sorted(ids[:n1]), "val": sorted(ids[n1:n2]),
           "test": sorted(ids[n2:])}, open(PROC / "random_split.json", "w"))
print("wrote splits.json, random_split.json, split_stats.csv")
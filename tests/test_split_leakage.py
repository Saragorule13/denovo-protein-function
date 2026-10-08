"""Leakage audit (run any time: python tests/test_split_leakage.py). Exits 1 on failure."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"

sp = json.load(open(PROC / "splits.json"))
tr, va, te = set(sp["train"]), set(sp["val"]), set(sp["test"])
assert not (tr & va) and not (tr & te) and not (va & te), "overlap between splits!"
print(f"no overlap: train={len(tr)} val={len(va)} test={len(te)}")

# no MMseqs2 cluster spans two splits
raw = (PROC / "mm_cluster.tsv").read_bytes().replace(b"\r", b"").decode()
owner = {}
for line in raw.split("\n"):
    p = line.split("\t")
    if len(p) >= 2:
        owner.setdefault(p[0].strip(), set()).add(p[1].strip())
which = {}
for k, ids in sp.items():
    for i in ids:
        which[i] = k
bad = [r for r, ms in owner.items() if len({which.get(m) for m in ms}) > 1]
assert not bad, f"{len(bad)} clusters span splits, e.g. {bad[:3]}"
print(f"no cluster spanning across {len(owner)} clusters")

import pandas as pd
clean = pd.read_csv(PROC / "clean.csv")
assert set(clean["accession"]) == tr | va | te, "split != clean set!"
print("splits cover exactly the clean set. ALL CHECKS PASSED")
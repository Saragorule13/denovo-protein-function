"""Phase 2a: cleaning + dedup + label map (from repo root: python src/data/preprocess.py).

In : data/raw/labels_raw.csv + data/raw/acquisition_log.csv
Out : data/processed/clean.csv, label_map.json, class_stats.csv, seqs.fasta
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

STD_AA = set("ACDEFGHIKLMNPQRSTVWY")

lab = pd.read_csv(RAW / "labels_raw.csv")
log = pd.read_csv(RAW / "acquisition_log.csv")
print(f"input labels: {len(lab)}")

# 1. keep only proteins with a usable structure
have = set(log.loc[log["status"].str.startswith("ok"), "accession"])
lab = lab[lab["accession"].isin(have)].copy()
print(f"after structure filter: {len(lab)} (dropped missing-structure)")

# 2. ambiguous-residue filter (>5% non-standard AA -> drop)
def ambig_frac(s):
    s = str(s).upper()
    return sum(1 for c in s if c not in STD_AA) / max(len(s), 1)

lab["ambig"] = lab["sequence"].apply(ambig_frac)
lab = lab[lab["ambig"] <= 0.05].copy()
print(f"after ambiguous-residue filter: {len(lab)}")

# 3. drop partial EC codes (contain '-'), then drop ECs with <50 examples
lab["ec_top"] = lab["ec_top"].apply(
    lambda s: [e for e in str(s).split(";") if "-" not in e])
lab = lab[lab["ec_top"].map(len) > 0].copy()
print(f"after partial-EC removal: {len(lab)}")
cnt = Counter(e for l in lab["ec_top"] for e in l)
print(f"ECs with >=50 examples after partial drop: {sum(v >= 50 for v in cnt.values())}")
keep_ec = {e for e, v in cnt.items() if v >= 50}
lab["ec_top"] = lab["ec_top"].apply(lambda l: [e for e in l if e in keep_ec])
lab = lab[lab["ec_top"].map(len) > 0].copy()
print(f"kept ECs: {len(keep_ec)}, proteins: {len(lab)}")

# 4. exact-sequence dedup (keep first accession alphabetically)
lab = lab.sort_values("accession").drop_duplicates("sequence", keep="first")
lab = lab.reset_index(drop=True)
print(f"after exact dedup: {len(lab)}")

# 5. label map (sorted EC order) + pos_weight for BCE
ec_sorted = sorted(keep_ec)
label_map = {ec: i for i, ec in enumerate(ec_sorted)}
json.dump(label_map, open(PROC / "label_map.json", "w"), indent=1)
n = len(lab)
rows = []
for ec in ec_sorted:
    pos = sum(ec in l for l in lab["ec_top"])
    rows.append({"ec": ec, "pos": pos, "neg": n - pos,
                 "pos_weight": round((n - pos) / max(pos, 1), 3)})
pd.DataFrame(rows).to_csv(PROC / "class_stats.csv", index=False)
print(f"classes: {len(ec_sorted)}, most rare: {min(r['pos'] for r in rows)}, "
      f"most common: {max(r['pos'] for r in rows)}")

lab["ec_level1"] = lab["ec_top"].apply(
    lambda l: ";".join(sorted({e.split(".")[0] for e in l})))
lab["ec_top"] = lab["ec_top"].apply(";".join)
lab[["accession", "sequence", "length", "organism", "ec_top", "ec_level1"]].to_csv(
    PROC / "clean.csv", index=False)

with open(PROC / "seqs.fasta", "w") as f:
    for t in lab.itertuples():
        f.write(f">{t.accession}\n{t.sequence}\n")
print(f"wrote clean.csv ({len(lab)}), label_map.json, class_stats.csv, seqs.fasta")
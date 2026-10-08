"""Phase 1: download UniProtKB Swiss-Prot enzymes (+ non-enzyme sample).

Usage (from repo root):
  python src/data/download_uniprot.py --pilot        # 500 rows, quick validation
  python src/data/download_uniprot.py --full         # paginated up to max_full_records
  python src/data/download_uniprot.py --process      # build labels_raw.csv (Top-N EC + caps)

Steps are separate so the pilot can be inspected before the bulk download.
"""
import argparse
import io
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[2]
CFG = yaml.safe_load(open(ROOT / "configs" / "data.yaml"))["uniprot"]
PATHS = yaml.safe_load(open(ROOT / "configs" / "data.yaml"))["paths"]
LCFG = yaml.safe_load(open(ROOT / "configs" / "data.yaml"))["labels"]


def fetch_pages(query, fields, page_size, limit, tag):
    url = CFG["base"]
    params = {"query": query, "fields": fields, "format": "tsv",
              "size": page_size}
    frames, total = [], 0
    sess = requests.Session()
    while True:
        for attempt in range(4):
            r = sess.get(url, params=params, timeout=120)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1))
                continue
            r.raise_for_status()
            break
        df = pd.read_csv(io.StringIO(r.text), sep="\t")
        if df.empty:
            break
        frames.append(df)
        total += len(df)
        print(f"[{tag}] pages: got {len(df)} rows (total {total})", flush=True)
        if total >= limit:
            break
        nxt = r.links.get("next", {}).get("url")
        if not nxt:
            break
        url, params = nxt, None  # next URL already contains query+cursor
        time.sleep(0.5)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out.head(limit)


def cmd_pilot():
    df = fetch_pages(CFG["query"], CFG["fields"], CFG["page_size"],
                     CFG["pilot_size"], "pilot")
    dest = ROOT / "data" / "raw" / "uniprot_pilot.tsv"
    df.to_csv(dest, sep="\t", index=False)
    print(f"pilot rows={len(df)} cols={list(df.columns)} -> {dest}")
    print(df["Length"].describe().to_string())
    print("EC level-1 counts:")
    lvl1 = df["EC number"].dropna().str.split(";").explode().str.strip()
    lvl1 = lvl1[lvl1 != ""].str.split(".").str[0]
    print(lvl1.value_counts().to_string())


def cmd_full():
    df = fetch_pages(CFG["query"], CFG["fields"], CFG["page_size"],
                     CFG["max_full_records"], "full")
    dest = ROOT / PATHS["raw_tsv"]
    df.to_csv(dest, sep="\t", index=False)
    print(f"full rows={len(df)} -> {dest}")


def cmd_process():
    src = ROOT / PATHS["raw_tsv"]
    if not src.exists():  # fall back to pilot for a dry run
        src = ROOT / "data" / "raw" / "uniprot_pilot.tsv"
        print(f"NOTE: {PATHS['raw_tsv']} missing, processing pilot file {src}")
    df = pd.read_csv(src, sep="\t")
    df = df.rename(columns={"Entry": "accession", "Sequence": "sequence",
                            "Length": "length", "EC number": "ec_raw",
                            "Organism": "organism"})
    # explode EC list per protein
    df["ec_list"] = (df["ec_raw"].fillna("")
                     .str.split(";").apply(lambda l: [e.strip() for e in l if e.strip()]))
    df = df[df["ec_list"].map(len) > 0].copy()
    counts = (df.explode("ec_list").groupby("ec_list").size()
              .sort_values(ascending=False))
    top = counts[counts >= LCFG["min_examples_per_ec"]].head(LCFG["top_n_ec"])
    print(f"ECs with >={LCFG['min_examples_per_ec']} examples: {(counts >= LCFG['min_examples_per_ec']).sum()}")
    print("Top-10 ECs:"); print(top.head(10).to_string())
    keep_ecs = set(top.index)
    df["ec_top"] = df["ec_list"].apply(lambda l: [e for e in l if e in keep_ecs])
    df = df[df["ec_top"].map(len) > 0].copy()
    # cap per-EC for laptop scale (stratified-ish: sample within each EC's member rows)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    seen = {e: 0 for e in keep_ecs}
    keep_idx = []
    for i, row in df.iterrows():
        if all(seen[e] >= LCFG["max_per_ec"] for e in row["ec_top"]):
            continue
        keep_idx.append(i)
        for e in row["ec_top"]:
            seen[e] += 1
    df = df.loc[keep_idx].reset_index(drop=True)
    df["ec_level1"] = df["ec_top"].apply(lambda l: sorted({e.split(".")[0] for e in l}))
    out = df[["accession", "sequence", "length", "organism",
              "ec_raw", "ec_top", "ec_level1"]].copy()
    out["ec_top"] = out["ec_top"].apply(lambda l: ";".join(l))
    out["ec_level1"] = out["ec_level1"].apply(lambda l: ";".join(l))
    dest = ROOT / PATHS["labels_csv"]
    out.to_csv(dest, index=False)
    print(f"labels rows={len(out)} kept ECs={len(keep_ecs)} -> {dest}")
    print(out["length"].describe().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--process", action="store_true")
    a = ap.parse_args(sys.argv[1:])
    if a.pilot:
        cmd_pilot()
    elif a.full:
        cmd_full()
    elif a.process:
        cmd_process()
    else:
        ap.print_help()
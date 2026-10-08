"""Phase 1: fetch AlphaFold structures for accessions in labels (or pilot).

Usage (from repo root):
  python src/data/fetch_structures.py --pilot        # accessions from uniprot_pilot.tsv
  python src/data/fetch_structures.py --full         # accessions from labels_raw.csv

Resume-safe: skips accessions already present in acquisition_log.csv with status ok.
PDB fallback: for AF misses with an xref_pdb entry, downloads first PDB from RCSB.
"""
import argparse
import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
import yaml

ROOT = Path(__file__).resolve().parents[2]
CFG_ALL = yaml.safe_load(open(ROOT / "configs" / "data.yaml"))
ACFG = CFG_ALL["alphafold"]
PATHS = CFG_ALL["paths"]


def load_targets(pilot):
    if pilot:
        df = pd.read_csv(ROOT / "data" / "raw" / "uniprot_pilot.tsv", sep="\t")
        df = df.rename(columns={"Entry": "accession", "PDB": "xref_pdb"})
    else:
        df = pd.read_csv(ROOT / PATHS["labels_csv"])
        raw = pd.read_csv(ROOT / PATHS["raw_tsv"], sep="\t", usecols=["Entry", "PDB"])
        pdb_map = dict(zip(raw["Entry"], raw["PDB"].fillna("")))
        df["xref_pdb"] = df["accession"].map(pdb_map).fillna("")
    return df[["accession", "xref_pdb"]].drop_duplicates("accession")


def load_done():
    log = ROOT / PATHS["acquisition_log"]
    if log.exists():
        d = pd.read_csv(log)
        return set(d.loc[d["status"].str.startswith("ok"), "accession"])
    return set()


def fetch_one(sess, acc, xref_pdb):
    errs = ""
    for attempt in range(ACFG["retry"]):
        try:
            r = sess.get(f"{ACFG['api']}/{acc}", timeout=ACFG["timeout_s"])
            if r.status_code == 404:
                errs = "af_404"; break
            r.raise_for_status()
            preds = r.json()
            if not preds:
                errs = "af_empty"; break
            best = max(preds, key=lambda p: p.get("globalMetricValue", -1))
            pdb_url = best.get("pdbUrl", "")
            plddt = float(best.get("globalMetricValue", -1))
            if not pdb_url:
                errs = "af_no_url"; break
            dl = sess.get(pdb_url, timeout=ACFG["timeout_s"])
            dl.raise_for_status()
            dest = ROOT / PATHS["structures_dir"] / f"AF-{acc}-F1.pdb"
            dest.write_bytes(dl.content)
            return {"accession": acc, "source": "AF",
                    "mean_plddt": round(plddt, 2),
                    "af_version": pdb_url.split("model_v")[-1].split(".")[0]
                    if "model_v" in pdb_url else "",
                    "status": "ok" if plddt >= ACFG["plddt_flag_below"]
                    else "ok_low_plddt", "bytes": len(dl.content), "note": ""}
        except Exception as e:  # noqa: BLE001 - log and retry
            errs = f"af_err:{type(e).__name__}"
            time.sleep(2 * (attempt + 1))
    # PDB fallback
    if isinstance(xref_pdb, str) and xref_pdb.strip():
        pdb_id = xref_pdb.split(";")[0].strip().split()[0][:4]
        try:
            dl = sess.get(f"https://files.rcsb.org/download/{pdb_id}.pdb",
                          timeout=ACFG["timeout_s"])
            dl.raise_for_status()
            dest = ROOT / PATHS["structures_dir"] / f"PDB-{acc}-{pdb_id}.pdb"
            dest.write_bytes(dl.content)
            return {"accession": acc, "source": f"PDB:{pdb_id}",
                    "mean_plddt": "", "af_version": "", "status": "ok_pdb_fallback",
                    "bytes": len(dl.content), "note": errs}
        except Exception as e:  # noqa: BLE001
            errs += f"|pdb_err:{type(e).__name__}"
    return {"accession": acc, "source": "", "mean_plddt": "",
            "af_version": "", "status": "missing", "bytes": 0, "note": errs}


def main(pilot):
    targets = load_targets(pilot)
    done = load_done()
    todo = [t for t in targets.itertuples() if t.accession not in done]
    print(f"targets={len(targets)} already_ok={len(done)} todo={len(todo)}")
    log = ROOT / PATHS["acquisition_log"]
    write_header = not log.exists()
    sess = requests.Session()
    n_ok = 0
    with open(log, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["accession", "source", "mean_plddt",
                                          "af_version", "status", "bytes", "note"])
        if write_header:
            w.writeheader()
        with ThreadPoolExecutor(max_workers=ACFG["workers"]) as ex:
            futs = {ex.submit(fetch_one, sess, t.accession, str(t.xref_pdb)): t.accession
                    for t in todo}
            for i, fut in enumerate(as_completed(futs), 1):
                row = fut.result()
                w.writerow(row); f.flush()
                if row["status"].startswith("ok"):
                    n_ok += 1
                if i % 25 == 0:
                    print(f"  ...{i}/{len(todo)} ok={n_ok}", flush=True)
                time.sleep(0.05)
    print(f"done. newly ok={n_ok}/{len(todo)} log->{log}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    main(pilot=not a.full)
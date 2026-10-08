"""Phase 3b: residue graphs with ESM + structural features (from repo root).

  python src/data/build_graphs.py [--n N] [--workers 6]

In : clean.csv, splits.json, label_map.json, structures/, embeddings/
Out : data/processed/graphs/{acc}.pt + graphs_manifest.csv
Graph: nodes=residues [509-d], edges=kNN(10)+seq, edge_attr [19-d], y [65] multi-hot.
"""
import argparse
import io
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio.Align import PairwiseAligner
from Bio.PDB import PDBParser
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
GDIR = PROC / "graphs"
GDIR.mkdir(parents=True, exist_ok=True)

AA3 = {"ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
       "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
       "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
       "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y"}
AA20 = "ACDEFGHIKLMNPQRSTVWY"
A2I = {a: i for i, a in enumerate(AA20)}
# physicochemical table: [hydrophobicity(KD), charge, polarity, mass, volume, flexibility, hbond]
PHYS = {
 "A": [-1.8, 0, 0.0, 89.1, 88.6, 0.36, 0], "C": [-2.5, 0, 1.5, 121.2, 108.5, 0.35, 1],
 "D": [3.5, -1, 49.7, 133.1, 111.1, 0.51, 3], "E": [3.5, -1, 49.9, 147.1, 138.4, 0.50, 3],
 "F": [2.8, 0, 0.3, 165.2, 189.9, 0.31, 0], "G": [0.4, 0, 0.0, 75.1, 60.1, 0.54, 0],
 "H": [3.2, 1, 51.6, 155.2, 153.2, 0.32, 2], "I": [-4.5, 0, 0.2, 131.2, 166.7, 0.46, 0],
 "K": [3.9, 1, 49.5, 146.2, 168.6, 0.47, 2], "L": [-3.8, 0, 0.2, 131.2, 166.7, 0.37, 0],
 "M": [-1.9, 0, 1.4, 149.2, 162.9, 0.32, 0], "N": [3.5, 0, 3.4, 132.1, 114.1, 0.46, 3],
 "P": [1.6, 0, 1.6, 115.1, 122.7, 0.51, 0], "Q": [3.5, 0, 3.5, 146.1, 143.9, 0.49, 3],
 "R": [4.5, 1, 52.0, 174.2, 173.4, 0.53, 4], "S": [0.8, 0, 1.7, 105.1, 89.0, 0.51, 2],
 "T": [0.7, 0, 1.7, 119.1, 116.1, 0.44, 2], "V": [-4.2, 0, 0.1, 117.1, 140.0, 0.47, 0],
 "W": [0.9, 0, 2.1, 204.2, 227.8, 0.31, 1], "Y": [1.3, 0, 1.6, 181.2, 193.6, 0.32, 2]}
RBF_C = np.linspace(0, 20, 16).astype(np.float32)
RBF_S = 1.0
KNN_K = 10
MIN_IDENT = 0.85  # min fraction of struct residues identical to mapped seq positions

_ALN = None


def _aligner():
    global _ALN
    if _ALN is None:
        a = PairwiseAligner()
        a.mode = "local"
        a.match_score = 2.0
        a.mismatch_score = -1.0
        a.open_insertion_score = -2.0
        a.open_deletion_score = -2.0
        a.extend_insertion_score = -0.5
        a.extend_deletion_score = -0.5
        _ALN = a
    return _ALN


def find_struct(acc):
    p = RAW / "structures" / f"AF-{acc}-F1.pdb"
    if p.exists():
        return p, "AF"
    cands = sorted((RAW / "structures").glob(f"PDB-{acc}-*.pdb"))
    return (cands[0], "PDB") if cands else (None, "")


def parse_chains(path):
    """All chains with >=10 CA as (aas, coords, conf). Handles bad encodings."""
    txt = path.read_bytes().decode("utf-8", errors="replace")
    struct = PDBParser(QUIET=True).get_structure("x", io.StringIO(txt))
    chains = []
    for model in struct:
        for chain in model:
            recs = []
            for res in chain:
                if "CA" not in res:
                    continue
                a = AA3.get(res.get_resname().strip().upper(), "X")
                ca = res["CA"]
                recs.append((a, ca.get_coord(), ca.get_bfactor()))
            if len(recs) >= 10:
                aas = [r[0] for r in recs]
                chains.append((aas, np.array([r[1] for r in recs], np.float32),
                               np.array([r[2] for r in recs], np.float32)))
        break  # first model only
    return sorted(chains, key=lambda c: -len(c[0]))


def align_chain(struct_aas, seq):
    """Return (seq_idx array, ident) or (None, ident). Tags/mutations tolerated."""
    s = "".join(struct_aas)
    if len(s) == len(seq) and s[:5] == seq[:5] and s[-5:] == seq[-5:]:
        return np.arange(len(s)), 1.0
    i = seq.find(s)
    if i >= 0:
        return np.arange(i, i + len(s)), 1.0
    best = _aligner().align(seq, s)[0]
    m = np.full(len(s), -1, dtype=np.int64)
    nmatch = 0
    for (a0, a1), (b0, b1) in zip(*best.aligned):
        for a, b in zip(range(a0, a1), range(b0, b1)):
            m[b] = a
            nmatch += (seq[a] == s[b])
    ident = nmatch / max(len(s), 1)
    if ident < MIN_IDENT or (m < 0).all():
        return None, round(float(ident), 3)
    return m, round(float(ident), 3)


def build(row, label_map, split_of):
    acc, seq, ec_top = row
    path, src = find_struct(acc)
    if path is None:
        return {"accession": acc, "status": "no_file", "src": src}
    try:
        chains = parse_chains(path)
    except Exception as e:  # noqa: BLE001
        return {"accession": acc, "status": f"parse_err:{type(e).__name__}", "src": src}
    if not chains:
        return {"accession": acc, "status": "too_few_ca", "src": src}
    cand = []
    for aas, ca, conf in chains[:3]:
        m, ident = align_chain(aas, seq)
        if m is not None:
            cand.append((ident, aas, ca, conf, m))
    if not cand:
        return {"accession": acc, "status": "align_fail", "src": src,
                "n_ca": len(chains[0][0]), "len": len(seq)}
    ident, aas, ca, conf, m = max(cand, key=lambda c: (c[0], len(c[1])))
    keep = m >= 0  # drop tag/mutated residues with no sequence counterpart
    aas = [a for a, k in zip(aas, keep) if k]
    ca, conf, m = ca[keep], conf[keep], m[keep]
    L = len(aas)
    if L < 10:
        return {"accession": acc, "status": "too_few_aligned", "src": src}
    emb = torch.load(PROC / "embeddings" / f"{acc}.pt",
                     weights_only=True).float().numpy()  # [Ls,480]
    if emb.shape[0] != len(seq):
        return {"accession": acc, "status": "emb_len_mismatch", "src": src}
    oh = np.zeros((L, 20), np.float32)
    ph = np.zeros((L, 7), np.float32)
    for i, a in enumerate(aas):
        if a in A2I:
            oh[i, A2I[a]] = 1.0
            ph[i] = PHYS[a]
    x = np.concatenate([oh, ph, emb[m],
                        (np.arange(L) / max(L - 1, 1)).reshape(-1, 1).astype(np.float32),
                        conf.reshape(-1, 1)], axis=1)
    k = min(KNN_K, L - 1)
    d, idx = NearestNeighbors(n_neighbors=k + 1).fit(ca).kneighbors(ca)
    es = set()
    for i in range(L):
        for j, dist in zip(idx[i][1:], d[i][1:]):
            es.add((min(i, int(j)), max(i, int(j)), float(dist)))
        for j in (i - 1, i + 1):
            if 0 <= j < L:
                es.add((min(i, j), max(i, j), float(np.linalg.norm(ca[i] - ca[j]))))
    ei, ea = [], []
    for i, j, dist in sorted(es):
        rbf = np.exp(-((dist - RBF_C) ** 2) / (2 * RBF_S ** 2))
        ss = abs(i - j)
        ea_row = np.concatenate([[dist], rbf, [np.log1p(ss)],
                                 [1.0 if dist < 8.0 else 0.0]]).astype(np.float32)
        ei += [[i, j], [j, i]]
        ea += [ea_row, ea_row]
    y = np.zeros(len(label_map), np.float32)
    for e in str(ec_top).split(";"):
        if e in label_map:
            y[label_map[e]] = 1.0
    g = {"x": torch.from_numpy(x), "edge_index": torch.tensor(ei).t().long(),
         "edge_attr": torch.from_numpy(np.stack(ea)),
         "y": torch.from_numpy(y), "accession": acc, "split": split_of.get(acc, "?"),
         "struct_source": src, "seq_idx": torch.from_numpy(m.astype(np.int64)),
         "align_ident": ident}
    torch.save(g, GDIR / f"{acc}.pt")
    return {"accession": acc, "status": "ok", "nodes": L,
            "edges": len(ei), "avg_deg": round(2 * len(es) / L, 2),
            "split": g["split"], "src": src, "ident": ident}


def run(n, workers):
    label_map = json.load(open(PROC / "label_map.json"))
    splits = json.load(open(PROC / "splits.json"))
    split_of = {a: k for k, vs in splits.items() for a in vs}
    clean = pd.read_csv(PROC / "clean.csv")
    if n:
        clean = clean.head(n)
    have = {p.stem for p in GDIR.glob("*.pt")}
    # rebuild previously-failed entries too (new alignment logic)
    man_path = PROC / "graphs_manifest.csv"
    prev_fail = set()
    if man_path.exists():
        prev = pd.read_csv(man_path)
        prev_fail = set(prev.loc[prev["status"] != "ok", "accession"])
    todo = [(r.accession, r.sequence, r.ec_top) for r in clean.itertuples()
            if r.accession not in have or r.accession in prev_fail]
    for acc in list(prev_fail):
        (GDIR / f"{acc}.pt").unlink(missing_ok=True)
    print(f"total={len(clean)} todo={len(todo)}", flush=True)
    rows = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            for i, res in enumerate(ex.map(build, todo,
                                           [label_map] * len(todo),
                                           [split_of] * len(todo), chunksize=20)):
                rows.append(res)
                if (i + 1) % 500 == 0:
                    print(f"  ...{i + 1}/{len(todo)}", flush=True)
    else:
        for i, r in enumerate(todo):
            rows.append(build(r, label_map, split_of))
            if (i + 1) % 200 == 0:
                print(f"  ...{i + 1}/{len(todo)}", flush=True)
    man = pd.DataFrame(rows)
    old = pd.read_csv(man_path) if man_path.exists() else pd.DataFrame()
    full = pd.concat([old, man], ignore_index=True)
    full = full.sort_values("accession").drop_duplicates("accession", keep="last")
    full.to_csv(man_path, index=False)
    print(full["status"].value_counts().to_string())
    print(f"manifest rows={len(full)} -> {man_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    run(a.n, a.workers)
"""M2: visualize one protein graph (from repo root: python src/explain/graph_viz.py [--acc XXX]).

Out: results/graph_M2_contactmap.png (seq vs spatial edges) + results/graph_M2_3d.png
     (C-alpha trace coloured by node degree).
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]


def pick():
    man = pd.read_csv(ROOT / "data" / "processed" / "graphs_manifest.csv")
    ok = man[(man["status"] == "ok") & (man["split"] == "train")]
    cand = ok[(ok["nodes"] >= 250) & (ok["nodes"] <= 400)].sort_values("nodes")
    for acc in cand["accession"]:
        g = torch.load(ROOT / "data" / "processed" / "graphs" / f"{acc}.pt",
                       weights_only=False)
        if g["y"].sum().item() == 1.0:
            return acc
    return cand.iloc[0]["accession"]


def main(acc):
    acc = acc or pick()
    g = torch.load(ROOT / "data" / "processed" / "graphs" / f"{acc}.pt",
                   weights_only=False)
    L = g["x"].size(0)
    ei = g["edge_index"].numpy()
    ea_src = "AF" if g["struct_source"] == "AF" else "PDB"
    print(f"{acc}: L={L} E={ei.shape[1]} src={ea_src} "
          f"EC={np.where(g['y'].numpy() > 0)[0].tolist()}")
    is_seq = np.abs(ei[0] - ei[1]) == 1
    # --- contact map: seq edges (blue) vs spatial/kNN edges (red) ---
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ei[1][~is_seq], ei[0][~is_seq], s=1, c="crimson", label="spatial (kNN)",
               alpha=0.6)
    ax.scatter(ei[1][is_seq], ei[0][is_seq], s=1, c="steelblue", label="sequential",
               alpha=0.6)
    ax.set_xlabel("residue j"); ax.set_ylabel("residue i")
    ax.set_title(f"{acc} residue graph (L={L}, E={ei.shape[1]})")
    ax.legend(markerscale=5); ax.set_aspect("equal")
    fig.savefig(ROOT / "results" / "graph_M2_contactmap.png", dpi=130,
                bbox_inches="tight")
    # --- 3D C-alpha trace coloured by degree ---
    from Bio.PDB import PDBParser
    import glob as G
    pdb = (ROOT / "data" / "raw" / "structures" / f"AF-{acc}-F1.pdb")
    if not pdb.exists():
        pdb = Path(sorted(G.glob(str(ROOT / "data" / "raw" / "structures" / f"PDB-{acc}-*.pdb")))[0])
    cas = [a.get_coord() for m in PDBParser(QUIET=True).get_structure("x", str(pdb))
           for c in m for r in c for a in r if a.get_id() == "CA"]
    cas = np.array(cas[:L])
    deg = np.bincount(ei[0], minlength=L)
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(cas[:, 0], cas[:, 1], cas[:, 2], c=deg, s=8, cmap="plasma")
    for i in range(L - 1):
        ax.plot(cas[i:i + 2, 0], cas[i:i + 2, 1], cas[i:i + 2, 2],
                c="gray", lw=0.5, alpha=0.5)
    fig.colorbar(sc, label="node degree")
    ax.set_title(f"{acc} C-alpha trace (colour=node degree)")
    fig.savefig(ROOT / "results" / "graph_M2_3d.png", dpi=130, bbox_inches="tight")
    print("wrote results/graph_M2_contactmap.png + graph_M2_3d.png")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--acc", default="")
    main(ap.parse_args().acc or None)
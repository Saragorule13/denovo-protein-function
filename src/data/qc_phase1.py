"""Phase 1 QC: stats + figures + DATA_CARD.md (from repo root: python src/data/qc_phase1.py)."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

lab = ROOT / "data" / "raw" / "labels_raw.csv"
pilot = ROOT / "data" / "raw" / "uniprot_pilot.tsv"
log = ROOT / "data" / "raw" / "acquisition_log.csv"

print("=== labels ===")
if lab.exists():
    df = pd.read_csv(lab)
    print(f"rows={len(df)}")
    print(df["length"].describe().to_string())
    print("EC level-1 distribution:")
    print(df["ec_level1"].str.split(";").explode().value_counts().to_string())
    plt.figure(); df["length"].hist(bins=50)
    plt.xlabel("length"); plt.ylabel("count"); plt.title("Protein length")
    plt.savefig(ROOT / "results" / "phase1_length.png", dpi=120)
elif pilot.exists():
    df = pd.read_csv(pilot, sep="\t")
    print(f"pilot rows={len(df)} cols={list(df.columns)}")
    print(df["Length"].describe().to_string())
    lvl1 = df["EC number"].dropna().str.split(";").explode().str.strip()
    lvl1 = lvl1[lvl1 != ""].str.split(".").str[0]
    print("EC level-1 distribution:"); print(lvl1.value_counts().to_string())
    plt.figure(); df["Length"].hist(bins=50)
    plt.xlabel("length"); plt.ylabel("count"); plt.title("Pilot protein length")
    plt.savefig(ROOT / "results" / "phase1_length.png", dpi=120)
    plt.figure(); lvl1.value_counts().sort_index().plot(kind="bar")
    plt.xlabel("EC level-1"); plt.ylabel("count"); plt.title("Pilot EC level-1")
    plt.savefig(ROOT / "results" / "phase1_ec1.png", dpi=120)

print("=== structures ===")
n_struct = len(list((ROOT / "data" / "raw" / "structures").glob("*.pdb")))
print(f"pdb files on disk: {n_struct}")
if log.exists():
    lg = pd.read_csv(log)
    print(lg["status"].value_counts().to_string())
    print(f"coverage ok = {(lg['status'].str.startswith('ok')).mean():.1%} of {len(lg)}")
    ok = lg[lg["status"].str.startswith("ok") & lg["mean_plddt"].notna()]
    if len(ok):
        plt.figure(); pd.to_numeric(ok["mean_plddt"]).hist(bins=30)
        plt.xlabel("mean pLDDT"); plt.ylabel("count"); plt.title("AlphaFold mean pLDDT")
        plt.savefig(ROOT / "results" / "phase1_plddt.png", dpi=120)

card = ROOT / "data" / "raw" / "DATA_CARD.md"
card.write_text(f"""# ProteinGT Phase 1 DATA CARD
- Date: 2026-10-08 (re-run updates this file)
- UniProt query: see configs/data.yaml (`uniprot.query`)
- Labels: {lab.name if lab.exists() else pilot.name}
- Structures on disk: {n_struct}
- Acquisition log: {log.name if log.exists() else "not yet run"}
- Figures: results/phase1_*.png
""")
print(f"wrote {card}")
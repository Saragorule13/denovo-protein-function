# ProteinGT - Phase 1: Dataset Acquisition (DONE 2026-10-08)

Custom Swiss-Prot + AlphaFoldDB EC dataset: 9,587 enzymes (Top-100 EC),
2000 non-enzymes, 9,845 structures, 99.4% coverage. See `data/raw/DATA_CARD.md`.

## Reproduce
```
pip install -r requirements-phase1.txt
python src/data/download_uniprot.py --pilot     # validate query (500 rows)
python src/data/download_uniprot.py --full      # 30k rows paginated
python src/data/download_uniprot.py --process   # Top-100 EC, cap 200/EC -> labels_raw.csv
python src/data/fetch_structures.py --pilot     # 500 structures (test)
python src/data/fetch_structures.py --full      # all structures (resume-safe)
python src/data/qc_phase1.py                    # stats + figures + DATA_CARD
```

## Layout
- `configs/data.yaml` - all queries, caps, thresholds
- `src/data/download_uniprot.py` - UniProt REST download + label processing
- `src/data/fetch_structures.py` - AlphaFold API + RCSB PDB fallback
- `src/data/qc_phase1.py` - QC stats + figures
- `data/raw/` - TSVs, labels_raw.csv, structures/, acquisition_log.csv, DATA_CARD.md
- `results/phase1_*.png` - length / EC-1 / pLDDT distributions

## Next: Phase 2 - preprocessing (cleaning, dedup, label map, MMseqs2 splits)
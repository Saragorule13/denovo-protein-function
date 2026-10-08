# ProteinGT Phase 1 DATA CARD

- Date: 2026-10-08
- UniProt query: `(reviewed:true) AND (ec:*) AND (length:[50 TO 1000]) AND ((existence:1) OR (existence:2))`
- Non-enzyme query: same with `(NOT ec:*)`, 2000 sampled (for Stage-A warm-up)
- UniProt download: 30,000 rows paginated `rest.uniprot.org/uniprotkb` (`data/raw/uniprot_ec_raw.tsv`, 17.9 MB)
- Labels: Top-100 EC 4-digit/part-level codes with >=50 examples, capped at 200/EC
  -> 9,587 proteins, length 50-1000 (median 432), file `data/raw/labels_raw.csv`
- EC level-1 split: EC3 4013 / EC2 3547 / EC1 1296 / EC5 447 / EC4 268 / EC6 108 / EC7 75
- Note: ~10 kept EC codes are partial (e.g. `3.1.-.-`, `2.3.1.-`); review in Phase 2
  whether to drop/merge them.
- Structures: AlphaFoldDB API first (`AF-{ACC}-F1.pdb`), RCSB PDB fallback for AF misses
- Acquisition log: `data/raw/acquisition_log.csv` (9,903 rows incl. 316 pilot-only extras)
  ok 9209 / ok_low_plddt (mean pLDDT<70) 585 / ok_pdb_fallback 51 / missing 58
- Coverage: 9,529/9,587 label proteins have a structure = 99.40%
- Structures on disk: 9,845 PDB files, ~2.7 GB total `data/` = ~2.8 GB
- Sample parse check (Biopython, 5 random): C-alpha counts 240/406/243/656/381 - all sane
- Figures: `results/phase1_length.png`, `results/phase1_plddt.png`, `results/phase1_ec1.png`
- Reproduce: `python src/data/download_uniprot.py --full`,
  `python src/data/download_uniprot.py --process`,
  `python src/data/fetch_structures.py --full`, `python src/data/qc_phase1.py`
- Config: `configs/data.yaml`
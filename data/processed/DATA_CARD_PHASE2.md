# ProteinGT DATA CARD (Phase 1 + Phase 2, 2026-10-08/09)

## Phase 1 - acquisition
- UniProt query: `(reviewed:true) AND (ec:*) AND (length:[50 TO 1000]) AND ((existence:1) OR (existence:2))`
- 30,000 rows paginated -> Top-100 EC (>=50 ex, cap 200/EC) -> 9,587 proteins
- Non-enzymes: 2,000 sampled (Stage-A warm-up, `data/raw/uniprot_nonenz.tsv`)
- Structures: AlphaFoldDB API first, RCSB PDB fallback; 9,845 PDB files
- Coverage: 9,529/9,587 = 99.40% (ok 9209 / low-pLDDT 585 / pdb-fallback 51 / missing 58)

## Phase 2 - preprocessing & splits
- Dropped: 58 missing-structure, 0 ambiguous-residue (>5% non-ACDEFGHIKLMNPQRSTVWY),
  partial-EC-only proteins (codes with `-`), rare ECs (<50 ex), 145 exact-duplicate sequences
- SCOPE CHANGE (documented): partial-EC removal collapsed 100 -> 65 full EC classes,
  9,587 -> 5,605 proteins. Rarest class n=49, most common n=232.
- Artefacts: `data/processed/clean.csv` (5605), `label_map.json` (65 ECs, sorted),
  `class_stats.csv` (pos/neg/pos_weight per EC), `seqs.fasta`
- Clustering: MMseqs2 easy-cluster `--min-seq-id 0.4 -c 0.8 --cov-mode 0`
  (static v15 in WSL Ubuntu, 20 threads) -> 1,769 clusters, 994 singletons, largest 76
- Splits (cluster-stratified, seed 42): train 3905 / val 880 / test 820 (69.7/15.7/14.6)
- EC level-1 mix matched across splits within ~1pp (hill-climb refinement, cost 2.3e-4)
- Leakage audit `tests/test_split_leakage.py`: no overlap, no cluster spanning, exact coverage
- Twin `random_split.json` (same ratios, seed 42) kept for appendix leakage-gap experiment
- GOTCHA: `seqs.fasta` written on Windows has CRLF; MMseqs2 kept `\r` in names.
  `splits.py`/`test_split_leakage.py` strip `\r` when reading `mm_cluster.tsv`. Fixed at read time.
- Reproduce Phase 2: `python src/data/preprocess.py` (+ WSL mmseqs command from card),
  `python src/data/splits.py`, `python tests/test_split_leakage.py`
- Params: `configs/data.yaml` (`phase2` section)

## Next: Phase 3 - protein graph construction (C-alpha kNN+seq edges, ESM-35M node features)
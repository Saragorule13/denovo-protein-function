# Phase 3 DATA CARD - protein graphs (2026-10-08)

## Embeddings (`src/features/esm_embed.py`, ckpt `facebook/esm2_t12_35M_UR50D`)
- Frozen ESM-2 35M, per-residue last-hidden-state (BOS/EOS stripped), fp16 `[L,480]`
- 5,605/5,605 proteins in ~8 min on RTX 5050 Laptop 8GB (`batch-tokens 4000`, ~9 prot/s)
- Out: `data/processed/embeddings/{acc}.pt` + `embeddings_manifest.csv`

## Graphs (`src/data/build_graphs.py`, `--workers 6`)
- Parse: Biopython, largest chains, CA-only; encoding-safe via utf-8/replace handle
- Alignment seq<->structure: exact -> substring -> pairwise local (PairwiseAligner,
  accept >=85% identical); per-chain best; tag/mutated residues dropped via seq_idx map
- Nodes [509-d]: one-hot20 | physchem7 RAW | ESM480 | relpos | conf RAW (pLDDT or B-factor)
  -> NOTE for training: standardize physchem+conf with TRAIN-fitted scaler (no leakage)
- Edges: kNN(10) on CA + sequential(i,i+-1), undirected stored both directions
- Edge attr [19-d]: dist | RBF16(0-20A, sigma 1.0) | log1p(seq-sep) | contact<8A
- Labels: y multi-hot [65] from label_map.json
- Result: 5,584 ok / 19 align_fail (AF fragment entries: file covers only part of seq)
  / 2 too_few_ca -> 99.6% coverage; splits: train 3888 / val 879 / test 817
- Stats: L median 447, directed edges/node mean 13.4, align ident mean 0.978 (min 0.86)
- Audit (300 random graphs): shapes exact, indices valid, 0 non-finite, all y>=1 label
- Out: `data/processed/graphs/{acc}.pt` + `graphs_manifest.csv`

## M2 visualization (`src/explain/graph_viz.py`, protein P34345, L=250, EC idx 19)
- `results/graph_M2_contactmap.png`: sequential (blue diagonal) vs spatial kNN (red off-diagonal)
- `results/graph_M2_3d.png`: C-alpha trace coloured by node degree

## Env
- torch 2.14+cu132 (CUDA True, RTX 5050 8GB), transformers 5.16, torch-geometric 2.8,
  biopython, scikit-learn. Full list: requirements.txt
- GOTCHA (kept): MMseqs2 cluster names carry `\r` (see Phase 2 card); stripped at read.

## Next: Phase 4 - baselines (homology, LR/MLP on mean-ESM, 1D-CNN, Stage-A 7-class)
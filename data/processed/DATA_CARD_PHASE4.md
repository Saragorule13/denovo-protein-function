# Phase 4 DATA CARD - sequence baselines (2026-10-09)

## Design deviation (documented)
- Skipped the Stage-A 7-class warm-up: same pipeline-debugging value comes from
  running the baselines directly on the 65-EC multi-label task, with no throwaway
  single-label code. M3 redefined as: full-task baseline table complete.

## Artefacts
- `src/evaluation/metrics.py`: macro/micro/weighted-F1, mAP, P@k/R@k (k=1,3,5),
  val threshold tuning. AP warnings suppressed (rare ECs absent from val/test).
- `src/training/run_sklearn.py` + `data/processed/mean_esm.npz` (mean-pooled ESM cache)
- `src/data/homology.py`: test->train easy-search (WSL mmseqs, top-5, e<0.001),
  `results/homology_pred.csv`
- `src/training/train_cnn.py`: Conv1d(480->256)x2 + mean||max pool + MLP(512->256->65),
  BCE+pos_weight, AdamW 3e-4, early-stop patience 8 on val macro-F1
  -> `checkpoints/cnn_baseline.pt` (best val 0.7717)
- `results/metrics_baselines.csv`, `results/scores_{lr,mlp,cnn}_test.npy`,
  `results/phase4_baselines.png`

## Results (test, 65 EC, 40%-identity cluster split, thresholds tuned on val)
- prior: macro-F1 0.000 (floor, as expected)
- lr-meanESM: 0.448 / micro 0.623
- mlp-meanESM: 0.646 / micro 0.773
- cnn-ESMseq: 0.711 / micro 0.857 / P@1 0.862
- homology-top1: 0.711 / micro 0.857 / P@1 0.827 (coverage 89.5%)

## Reading
- Homology transfer (remote homologs <40% id, e<0.001) ties the CNN: function is
  conserved below the split threshold, so the bar for Phase 5 GNNs is macro-F1 > 0.71.
- MLP < CNN shows local sequence patterns add value over mean pooling.
- ESM-35M mean-pooled MLP alone (0.646) is already strong: graph structure must earn its keep.

## Next: Phase 5 - GCN -> GAT -> ProteinGT (TransformerConv x4) on identical splits/features
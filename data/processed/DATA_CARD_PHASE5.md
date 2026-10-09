# Phase 5 DATA CARD - GNNs single-seed (2026-10-09)

## Artefacts
- `src/data/graph_dataset.py`: PyG Dataset over cached graphs, y unsqueezed [1,65] for
  batching, edge_weight=1/(1+d) for GCN, train-fitted scaler (physchem+conf) -> scaler.pt
- `src/models/gnn.py`: GCNNet(3xGCN, 478k) / GATNet(3xGAT-4h edge-aware, 495k) /
  ProteinGT(4xTransformerConv-4h edge_dim=19, 1.35M); shared mean||max pool + MLP head
- `src/training/train_gnn.py`: BCE+pos_weight, AdamW 3e-4, AMP, clip 1.0, early-stop
  patience 12 on val macro-F1, val threshold tuning, --overfit + --resume support
- Checkpoints: `checkpoints/{gcn,gat,gt}_s42.pt` (+cnn_baseline.pt from Phase 4)

## M4 overfit-32 (ProteinGT): PASS
- Criterion corrected mid-phase: macro-F1@[0.5] is capped by absent classes in a tiny
  subset, so M4 uses micro-F1: 0.977 at ep50 (loss 1.0->0.018). Capacity confirmed.

## Single-seed test results (65 EC, cluster split)
- gcn-s42: macro 0.6999 / micro 0.8438 (best val 0.7421, 58 ep)
- gat-s42: macro 0.7141 / micro 0.8586 (best val 0.7567, 41 ep)
- gt-s42: macro 0.7170 / micro 0.8547 (best val 0.7465, ~55 ep total incl. resumed run)
- Ranking: prior 0.00 < LR 0.448 < MLP 0.646 < GCN 0.700 < CNN 0.711 ~= homology 0.711
  < GAT 0.714 < GT 0.717

## Notes
- GT run was killed by 1h tool timeout at ep21, resumed via new --resume flag
  (ckpt now stores best+epoch). Epoch time ~47s warm (first-run slowness was cold I/O).
- Sigmoid made numerically stable (torch.sigmoid) after overflow warnings in GCN run.
- GT beats homology by +0.006 macro-F1 single-seed; multi-seed needed to confirm gap.

## Next: Phase 6 - full training (GT x3 seeds), eval + curves, ablations (ESM size,
graph construction, edge features, depth, pooling), random-split leakage-gap appendix
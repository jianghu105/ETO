# ETO-Core Design Document

**Version:** 1.0
**Date:** 2026-10-19
**Status:** Implemented, smoke test passed

---

## 1. Scientific Problem

How can we use limited multi-round SELEX data to learn transferable sequence state transition rules, and apply these rules to explore the vast sequence space to discover novel candidates worth experimental validation under limited computational or experimental budget?

## 2. Architecture: One State, One Operator, One Decoder

```
Input: (sequence, history_counts, history_mask, transition_type, delta_t)
                │
                ▼
    ┌───────────────────────┐
    │  SequenceEncoder      │  Multi-kernel CNN [3,5,7] → 128d
    │  (reused from Legacy) │
    └───────────┬───────────┘
                │ seq_emb (128d)
                ▼
    ┌───────────────────────┐
    │  HistoryEncoder       │  MLP: [counts, mask] → 32d
    └───────────┬───────────┘
                │ hist_emb (32d)
                ▼
    ┌───────────────────────┐
    │  ContextEncoder       │  Transition embed + Fourier time → 16d
    └───────────┬───────────┘
                │ ctx_emb (16d)
                ▼
    ┌───────────────────────┐
    │  StateBuilder         │  [seq, hist, ctx] → MLP → 128d state z
    └───────────┬───────────┘
                │ z (128d)
                ▼
    ┌───────────────────────┐
    │  TransitionOperator   │  z_next = z + MLP(z, ctx, delta_t)
    └───────────┬───────────┘
                │ z_next (128d)
                ▼
    ┌───────────────────────┐
    │  ObservationDecoder   │  z_next → predicted next-round log-count
    └───────────────────────┘
```

## 3. Model Variants

| Model | Components | Purpose |
|---|---|---|
| **A: Static** | SequenceEncoder → head | Sequence-only baseline |
| **B: History** | Seq + Hist + Context → head | Tests history value |
| **C: ETO-Core** | Full pipeline with transition | Tests transition value |

All three share:
- Same data splits (family-aware, no leakage)
- Same target (normalized next-round log-count)
- Same loss (masked Huber)
- Same evaluation protocol

## 4. Data Protocol

### 4.1 Standard Fields
Each sample: `(sequence, history_counts, history_mask, transition_type, round_from, round_to, delta_t, target_value, target_mask, family_id)`

### 4.2 Splits
- **Family-aware**: No sequence family appears in multiple splits
- **70/15/15**: Train / Val / Test
- **Temporal OOD**: R9→R10 held out from test for temporal generalization
- **R10→R11 excluded**: No valid target (terminal round)

### 4.3 Target
- Primary: next-round normalized log-count (log1p(count))
- Normalization: fit mean/std from training data only
- All targets valid (R10→R11 excluded at data construction)

### 4.4 Family Assignment
- Greedy centroid k-mer clustering (k=8, 30% overlap threshold)
- No transitive closure (avoids merging unrelated families)

## 5. Training Protocol

- **Loss**: Masked Huber (delta=1.0) on valid targets
- **Optimizer**: AdamW (lr=1e-3, weight_decay=1e-4)
- **Schedule**: CosineAnnealingLR
- **Gradient clipping**: max_norm=1.0
- **AMP**: Mixed precision on GPU
- **Early stopping**: patience=15 on validation loss
- **Checkpointing**: Best validation + last epoch
- **Resume**: From last checkpoint if available

## 6. Evaluation Metrics

### Prediction Layer
- Spearman correlation
- Pearson correlation
- MAE, RMSE
- NDCG@50
- Per-transition breakdown
- Family-out test
- Temporal OOD test

### Comparison Protocol
- Same test set for all models
- Same normalization parameters
- Multiple seeds (planned)
- Report mean ± std

## 7. Key Design Decisions

### Why observable targets (not latent)?
Latent states have no ground truth. Direct MSE on latent predictions is unsupervised and requires careful handling (target networks, stop gradients, collapse prevention). V1 uses observable targets for clean supervision.

### Why residual transition?
`z_next = z + delta_z` ensures the operator learns *changes*, not absolute states. This is more sample-efficient and prevents the model from ignoring the input state.

### Why family-aware splits?
Random splits allow closely related sequences in both train and test, inflating metrics through memorization rather than generalization. Family splits test true generalization.

### Why exclude R10→R11?
R10→R11 is the terminal transition with no future round data. EA/EPS targets are undefined. Including it would require fabricating targets.

### Why greedy centroid clustering (not union-find)?
Union-find k-mer clustering creates transitive chains that merge all sequences into one family for typical SELEX data. Greedy centroid clustering with a strict overlap threshold produces meaningful family structure.

## 8. Files

```
data/
  unified_dataset.py    — Data loading, family assignment, splits, normalization
  eto_dataset.py        — PyTorch Dataset and DataLoader builders
  preprocessing.py      — Legacy preprocessing (preserved)
  dataset.py            — Legacy dataset (preserved)

models/
  eto_core.py           — ETO-Core model + Static/History baselines
  sequence_encoder.py   — Legacy (reused by ETO-Core)
  [other legacy files preserved]

training/
  eto_core_trainer.py   — Training loop, metrics, evaluation
  losses.py             — Legacy losses (preserved)
  trainer.py            — Legacy trainer (preserved)

tests/
  test_data_pipeline.py — 19 tests for data interface
  test_eto_core.py      — 21 tests for model

train_etocore.py        — Main training entry point
ETO_Colab.ipynb         — Colab notebook
```

## 9. Smoke Test Results (5 epochs, CPU)

| Model | Test Spearman | Test NDCG@50 | Params |
|---|---|---|---|
| Static | 0.1194 | 0.0603 | 76,161 |
| History | 0.8125 | 0.9695 | 141,049 |
| ETO-Core | 0.8125 | 0.9791 | 219,769 |

**Key finding**: History provides massive value (+0.69 Spearman over static). ETO-Core shows slight improvement over History. Full training on GPU expected to improve further.

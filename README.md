# ETO v1: Evolutionary Transition Operator

Learning molecular population evolution dynamics from multi-round SELEX NGS data.

## Overview

ETO v1 models SELEX as a conditional dynamical system:

```
Z_(t+1) = Z_t + T_theta(Z_t, E_t)
```

Where:
- `Z_t`: latent evolutionary state at round t
- `E_t`: selection context (normal / counter-selection / recovery)
- `T_theta`: Conditional Evolutionary Transition Operator

The model outputs an **Evolutionary Potential Score (EPS)** for each sequence, identifying latent high-potential aptamers that abundance-based ranking misses.

## Architecture

```
Sequence (36nt) -> [Conv1D + Attention] -> Seq Embedding (128d)
                                                    |
Count + Velocity -> [Concat] -------------> Individual State (130d)
                                                    |
All states -> [Evolutionary Attention Pool] -> Population State Z_t (256d)
                                                    |
Context (round type) -> [Embedding] ------> Context E_t (32d)
                                                    |
                                    [Transition Operator] -> Z_(t+1)
                                                    |
                                    [Potential Decoder] -> EPS Score
```

## Project Structure

```
ETO_v1/
├── data/
│   ├── preprocessing.py    # Data loading, one-hot encoding, feature computation
│   └── dataset.py          # PyTorch Dataset/DataLoader with ablation support
├── models/
│   ├── sequence_encoder.py # Multi-kernel Conv1D + attention pooling
│   ├── evolutionary_state.py  # Individual state builder
│   ├── population_encoder.py  # Attention-based population aggregation
│   ├── context_encoder.py     # Selection context embedding
│   ├── transition_operator.py # Conditional residual transition
│   ├── potential_decoder.py   # EPS score + population reconstruction
│   └── eto_model.py          # Full model assembly
├── training/
│   ├── losses.py           # Transition + Population KL + Ranking loss
│   └── trainer.py          # Training loop with checkpointing
├── evaluation/
│   ├── metrics.py          # Spearman, NDCG, top-k overlap
│   └── visualization.py    # Training curves, phi distributions, ablations
├── configs/
│   └── config.yaml         # All hyperparameters
├── train.py                # Main training script
├── evaluate.py             # Evaluation script
└── results/                # Outputs (checkpoint, metrics, figures, CSV)
```

## Quick Start

### 1. Synthetic Data Test (validates pipeline)

```bash
cd ETO_v1
python train.py --synthetic --ablation --epochs 50 --output_dir results/synthetic
```

### 2. Real MB-SELEX Data

```bash
python train.py --ablation --epochs 100 --output_dir results/real
```

### 3. Evaluate a Checkpoint

```bash
python evaluate.py --checkpoint results/checkpoint.pt --output_dir results/eval
```

## Training Objectives

```
L = L_transition + lambda_1 * L_population + lambda_2 * L_ranking
```

1. **Transition Loss**: MSE between predicted and target population states
2. **Population Reconstruction**: KL divergence on abundance distribution
3. **Ranking Loss**: Pairwise ranking - sequences with higher future counts should have higher EPS

## Anti-Shortcut Experiments

| Experiment | Purpose |
|-----------|---------|
| Count dropout | Force model to learn from sequence, not just counts |
| Trajectory shuffle | Verify model learns temporal order, not static mapping |
| Sequence-only baseline | Upper bound for sequence-only information |
| Abundance-only baseline | Upper bound for count-only information |

## R9 Counter-Selection Handling

R9 is encoded as a distinct context type (counter_selection). The transition loss weight for R5->R9 is reduced to 0.2 (vs 1.0 for other transitions) to account for the unusual selection pressure.

## Output

- `checkpoint.pt`: Model weights + training state
- `metrics.csv`: Evaluation metrics
- `predicted_potential.csv`: Per-sequence EPS scores and rankings
- `training_loss.png`: Training curves
- `phi_vs_count.png`: EPS vs future abundance correlation
- `ablation_comparison.png`: Full model vs baselines
- `round_trajectory.png`: Per-round transition loss

## Requirements

```
torch>=2.0
numpy
pandas
scipy
matplotlib
seaborn
pyyaml
openpyxl
```

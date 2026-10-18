# ETO-Legacy Audit Report

**Date:** 2026-10-19
**Auditor:** Automated code audit + manual review
**Scope:** All code in `ETO/` as of commit before ETO-Core refactoring
**Status:** CRITICAL ISSUES FOUND — retraining required with proper data splits

---

## 1. Architecture Overview (ETO-Legacy)

```
RefactoredETOModel (2,030,621 parameters)
├── SequenceEncoder        (61,441 params,  3.0%)  — multi-kernel CNN [3,5,7], attention pool
├── StructureEncoder       (1,328 params,   0.1%)  — GC/MFE/paired_prob → 16d
├── AbundanceFreeStateBuilder (0 params)           — concatenation logic only
├── AbundanceFreePopulationEncoder (393,233 params, 19.4%) — attention pool + gated obs
├── HybridContextEncoder   (2,512 params,   0.1%)  — context embed + Fourier time
├── RolloutTransitionOperator (1,067,776 params, 52.6%) — 3-layer 512-wide MLP
├── SetPopulationDecoder   (134,154 params,  6.6%) — Z → per-round stats
└── IndividualPotentialDecoder (370,177 params, 18.2%) — Z_future + features → EPS
```

### Data Flow
```
one_hot(B,4,36) → SequenceEncoder → seq_emb(B,128)
structure_features(B,3) → StructureEncoder → struct_emb(B,16)
[seq_emb, velocity, struct_emb] → dynamic_state(B,145)
log_count → observation_state(B,1)
dynamic_state → PopulationEncoder(attention pool) → z_dynamic(B,256)
  + gated observation → Z_t(B,256)
context_id + continuous_time → HybridContextEncoder → E_t(B,32)
[Z_t, E_t] → RolloutTransitionOperator → delta_z → Z_next(B,256)
Z_next → IndividualPotentialDecoder → EPS(B,)
```

### Data Statistics (verified)
- Source: `dataset/20230531AptDMC双端-MPO.xlsx` (30MB)
- Unique sequences: ~227,401
- Round-pair transfer samples: ~909K (4 round pairs × ~227K)
- Sequence length: 36 nt (fixed)
- Round pairs: R3→R5, R5→R9, R9→R10, R10→R11
- R9: 100% zero counts (counter-selection round)
- R10→R11: no valid EA (terminal round, no future data)

---

## 2. Critical Bugs (CRASH)

### 2.1 `train_real.py:174` — undefined `training_cfg`
```python
# Line 174 references training_cfg which is not defined in train() scope
# It was a local variable in build_dataloaders()
# This will raise NameError during post-training evaluation
```
**Impact:** Training completes but evaluation crashes.

### 2.2 `evaluation/baselines.py:256` — undefined `arrays`
```python
def evaluate_baselines(samples, features, config, device):
    n = len(arrays['pair_idx'])  # arrays is NOT defined
```
**Impact:** Baseline evaluation cannot run at all.

### 2.3 `validation/final_scientific_audit.py` — multiple crashes
- Line 33: Config path `configs/config.yaml` does not exist (should be `train_real.yaml`)
- Line 318: `batch['future_growth']` key does not exist in dataset output
- Lines 573-579: Iterates dict as list of sample dicts (type mismatch)

**Impact:** Scientific audit script cannot execute.

### 2.4 `models/eto_model.py:37-42` — legacy ETOModel dimension mismatch
```python
# PopulationEncoder constructed with state_dim=pop_cfg['population_dim'] (=256)
# But receives states of dimension state_dim = seq_emb_dim + 2 = 130
# Would crash at runtime if legacy model were instantiated
```
**Impact:** Legacy model is dead code.

---

## 3. Critical Design Issues

### 3.1 NO TRAIN/VAL/TEST SPLIT
**Severity: CRITICAL**
**Files:** `train_real.py:82,97`, `eval_colab.py:64-68`

```python
# train_real.py
val_loader = None  # Line 82 and 97 — no validation split exists
# Post-training evaluation runs on SAME data as training
```

All reported metrics (Spearman 0.41, NDCG@50 0.94) are **training-set metrics**. They do not measure generalization.

### 3.2 MASSIVE SAMPLE DUPLICATION
**Severity: HIGH**
**File:** `data/preprocessing.py:457-493`

Each sequence is replicated across all 4 round pairs. A dataset of N sequences produces 4N samples where the same sequence appears 4 times with different context. Without family-aware splitting, the same sequence (and its mutants) appear in both "training" and "evaluation."

### 3.3 FOURIER TIME ENCODING MISMATCH
**Severity: HIGH**
**Files:** `data/preprocessing.py:490`, `models/context_encoder.py:92`

```python
# preprocessing.py sets continuous_time to tau_curr from EVOLUTION_TIME_MAP:
#   R3:0, R5:2, R9:6, R10:7, R11:8
# But ContinuousTimeEncoder uses period T=4.0:
#   angles = 2*pi*t/4.0  (designed for 0-3 range)
# Actual input range is 0-8, causing aliasing
```

### 3.4 ROLLOUT TARGETS ALL IDENTICAL
**Severity: HIGH**
**File:** `models/eto_model.py:413`

```python
result['z_rollout_targets'] = [z_t_target] * rollout_steps
# All rollout steps target the SAME single-step Z_target
# Multi-step rollout is semantically incorrect
```

### 3.5 NO EARLY STOPPING / MODEL SELECTION
**Severity: HIGH**
**File:** `training/trainer.py:262`

```python
is_best = current_loss < best_loss  # Based on TRAINING loss, not validation
# val_loader is None, so no validation-based selection
```

### 3.6 LR SCHEDULE MISMATCH
**Severity: MEDIUM**
**File:** `training/trainer.py:41`

```python
total_steps = epochs * 100  # Hardcodes 100 steps/epoch
# Actual steps/epoch depends on dataset_size / batch_size
```

---

## 4. Data Leakage Risks

### 4.1 Train-Eval Contamination
- Same sequences in training and evaluation (no split at all)
- Same sequence appears in multiple round pairs (4× duplication)
- Family-level leakage: closely related sequences in both sets

### 4.2 Target Circularity
- EA for R3→R5 is computed from R9, R10, R11 counts
- This means early-round targets incorporate ALL future round information
- Not input leakage (model only sees prev/curr counts), but creates very strong targets

### 4.3 Velocity Carries Abundance
- velocity = log(count_curr) - log(count_prev) is part of dynamic_state
- EPS DOES change when abundance changes (acknowledged in test 11)
- "Abundance-free" property is partial, not complete

### 4.4 Population Loss Ignores Most Rounds
- `PopulationReconstructionLoss` only uses `pop_pred[:, :2]`
- SetPopulationDecoder outputs 8 values (4 rounds × 2), but only first 2 used
- 75% of decoder output is wasted

---

## 5. Minor Issues

| File | Issue |
|---|---|
| `dataset.py:51-54` | Count dropout uses `np.random.rand()` not PyTorch RNG — non-reproducible |
| `structure_encoder.py:48` | Batch-level zero check: if any sample has structure, all go through encoder |
| `transition_operator.py:93` | `nan_to_num` clipping at ±1.0 is arbitrary |
| `individual_potential_decoder.py:80` | EPS clipping at ±5.0 is arbitrary |
| `population_sampler.py` | Duplicate of `CompactSELEXDataset` in `dataset.py` |
| `configs/train_real.yaml:43` | `dynamic_dim: 145` is dead config, never read |
| `configs/train_real.yaml:99` | `output_dir: experiments/real_v2` but code uses `real_v1` |

---

## 6. Conclusions

### What Works
- Sequence encoder (multi-kernel CNN) is sound and reusable
- Data loading pipeline (compact arrays) is memory-efficient
- EA validity mask (R10→R11 exclusion) is correctly implemented
- Core model architecture has reasonable parameter distribution

### What Must Be Fixed Before Retraining
1. **Implement proper train/val/test splits** with family-aware grouping
2. **Fix Fourier time encoding** mismatch (period vs actual range)
3. **Fix or remove broken scripts** (baselines, audit, legacy model)
4. **Add validation-based early stopping** and model selection
5. **Fix LR schedule** to use actual steps per epoch
6. **Eliminate sample duplication** in evaluation (each sequence evaluated once per its actual round pair)

### What Must Be Fixed for ETO-Core
1. Design unified data interface with explicit masks
2. Implement proper held-out test set (family-level split)
3. Replace population-level architecture with per-sequence state transition
4. Use observable targets (next-round log-abundance) instead of latent targets
5. Implement fair baselines (Static, History) with same data pipeline

### Recommendation
**DO NOT** continue training the current model. The reported metrics are unreliable because there is no held-out evaluation. Proceed with ETO-Core refactoring per the task specification, starting with Phase 2 (unified data interface with proper splits).

---

## 7. Files Preserved

All legacy code is preserved under the `ETO/` directory. No files were deleted or modified during this audit. The legacy model is hereby designated **ETO-Legacy**.

- Models: `ETO/models/` (all files)
- Training: `ETO/train_real.py`, `ETO/training/`
- Data: `ETO/data/`
- Evaluation: `ETO/evaluation/`, `ETO/eval_colab.py`
- Config: `ETO/configs/train_real.yaml`
- Tests: `ETO/tests/test_refactored_architecture.py`
- Validation: `ETO/validation/`
- Checkpoints: `ETO/experiments/real_v1/checkpoint_best.pt` (on Colab/Drive)

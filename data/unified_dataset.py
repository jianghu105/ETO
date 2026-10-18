"""
ETO-Core Unified Data Interface
================================

Standard fields per training sample:
  sequence           : one-hot (4, L)
  history_counts     : (max_rounds,) normalized log-counts up to round_from
  history_mask       : (max_rounds,) bool — which rounds have valid observations
  transition_type    : int — selection condition category (0-3)
  round_from         : int — starting round index
  round_to           : int — target round index
  delta_t            : float — time span
  target_value       : float — next-round normalized log-count (primary target)
  target_mask        : bool — whether target is valid
  family_id          : int — sequence family for group-level splits

Design principles:
  - No sample duplication: each (sequence, round_pair) appears in exactly one split
  - Family-aware splitting: sequences from the same family stay together
  - Targets are real observables (next-round log-count), not latent EA/EPS
  - Normalization parameters fit from training data only
  - R10→R11 excluded (no valid target — terminal round)
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

from .preprocessing import (
    load_real_data, one_hot_encode, compute_structure_features,
    ROUND_NAMES, ROUND_PAIRS,
)

ROUND_TIME = {"R3": 0.0, "R5": 2.0, "R9": 6.0, "R10": 7.0, "R11": 8.0}
ROUND_TO_INDEX = {r: i for i, r in enumerate(ROUND_NAMES)}

VALID_TRANSITIONS = [
    ("R3", "R5"),
    ("R5", "R9"),
    ("R9", "R10"),
]

TRANSITION_TO_ID = {t: i for i, t in enumerate(VALID_TRANSITIONS)}


def _get_kmers(seq: str, k: int) -> set:
    return {seq[j:j + k] for j in range(len(seq) - k + 1)}


def assign_family_ids(sequences: List[str], k: int = 8,
                      min_overlap_frac: float = 0.3
                      ) -> np.ndarray:
    """
    Assign family IDs by greedy centroid k-mer clustering.

    For each unassigned sequence (centroid), assign all unassigned sequences
    that share >= min_overlap_frac of the centroid's k-mers.
    No transitive closure — avoids merging unrelated families.
    """
    n = len(sequences)
    kmer_sets = [_get_kmers(seq, k) for seq in sequences]
    family_ids = np.full(n, -1, dtype=np.int64)
    assigned = np.zeros(n, dtype=bool)
    current_family = 0

    order = np.argsort([-len(ks) for ks in kmer_sets])

    for idx in order:
        if assigned[idx]:
            continue
        centroid_kmers = kmer_sets[idx]
        if len(centroid_kmers) == 0:
            family_ids[idx] = current_family
            assigned[idx] = True
            current_family += 1
            continue

        family_ids[idx] = current_family
        assigned[idx] = True

        min_shared = max(1, int(len(centroid_kmers) * min_overlap_frac))
        for j in order:
            if assigned[j]:
                continue
            shared = len(centroid_kmers & kmer_sets[j])
            if shared >= min_shared:
                family_ids[j] = current_family
                assigned[j] = True

        current_family += 1

    return family_ids


def build_transition_samples(
    df: pd.DataFrame,
    family_ids: np.ndarray,
    seq_length: int = 36,
) -> Dict[str, np.ndarray]:
    """
    Build transition samples from raw data.

    Each sample = (sequence, history up to round_from, target = round_to count).
    Only includes VALID_TRANSITIONS (excludes R10→R11 which has no target).
    """
    n_seqs = len(df)
    counts = {}
    log_counts = {}
    for r in ROUND_NAMES:
        counts[r] = df[r].values.astype(np.float64)
        log_counts[r] = np.log1p(counts[r])

    n_transitions = len(VALID_TRANSITIONS)
    total = n_seqs * n_transitions
    max_history = 5

    one_hot = np.zeros((total, 4, seq_length), dtype=np.float32)
    structure_feat = np.zeros((total, 3), dtype=np.float32)
    history_counts = np.zeros((total, max_history), dtype=np.float32)
    history_mask = np.zeros((total, max_history), dtype=bool)
    transition_type = np.zeros(total, dtype=np.int64)
    round_from = np.zeros(total, dtype=np.int64)
    round_to = np.zeros(total, dtype=np.int64)
    delta_t = np.zeros(total, dtype=np.float32)
    target_value = np.zeros(total, dtype=np.float32)
    target_mask = np.zeros(total, dtype=bool)
    family_id = np.zeros(total, dtype=np.int64)
    seq_idx = np.zeros(total, dtype=np.int64)
    raw_count_target = np.zeros(total, dtype=np.float64)
    raw_count_curr = np.zeros(total, dtype=np.float64)

    one_hot_cache = np.zeros((n_seqs, 4, seq_length), dtype=np.float32)
    struct_cache = np.zeros((n_seqs, 3), dtype=np.float32)
    for i, seq in enumerate(df['sequence'].values):
        one_hot_cache[i] = one_hot_encode(seq, seq_length)
        struct_cache[i] = compute_structure_features(seq)

    for ti, (r_from, r_to) in enumerate(VALID_TRANSITIONS):
        start = ti * n_seqs
        end = (ti + 1) * n_seqs

        one_hot[start:end] = one_hot_cache
        structure_feat[start:end] = struct_cache

        history_rounds = []
        for r in ROUND_NAMES:
            if ROUND_TO_INDEX[r] <= ROUND_TO_INDEX[r_from]:
                history_rounds.append(r)

        for hi, r in enumerate(history_rounds):
            if hi >= max_history:
                break
            history_counts[start:end, hi] = log_counts[r]
            history_mask[start:end, hi] = True

        transition_type[start:end] = ti
        round_from[start:end] = ROUND_TO_INDEX[r_from]
        round_to[start:end] = ROUND_TO_INDEX[r_to]
        delta_t[start:end] = ROUND_TIME[r_to] - ROUND_TIME[r_from]

        target_value[start:end] = log_counts[r_to]
        target_mask[start:end] = True

        family_id[start:end] = family_ids
        seq_idx[start:end] = np.arange(n_seqs)
        raw_count_target[start:end] = counts[r_to]
        raw_count_curr[start:end] = counts[r_to]

    return {
        'one_hot': one_hot,
        'structure_features': structure_feat,
        'history_counts': history_counts,
        'history_mask': history_mask,
        'transition_type': transition_type,
        'round_from': round_from,
        'round_to': round_to,
        'delta_t': delta_t,
        'target_value': target_value,
        'target_mask': target_mask,
        'family_id': family_id,
        'seq_idx': seq_idx,
        'raw_count_target': raw_count_target,
        'raw_count_curr': raw_count_curr,
        'n_sequences': n_seqs,
        'n_transitions': n_transitions,
    }


class FamilySplitter:
    """
    Family-aware train/val/test splitter.

    Ensures no sequence family appears in multiple splits.
    Optionally holds out specific transitions for temporal ood testing.
    """

    def __init__(self, train_frac: float = 0.7, val_frac: float = 0.15,
                 seed: int = 42):
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.seed = seed

    def split(self, family_ids: np.ndarray,
              transition_type: np.ndarray,
              seq_idx: np.ndarray,
              ) -> Dict[str, np.ndarray]:
        """
        Returns dict with 'train', 'val', 'test' keys, each an array of
        sample indices into the full arrays.
        """
        rng = np.random.RandomState(self.seed)
        unique_families = np.unique(family_ids)
        n_families = len(unique_families)

        shuffled = rng.permutation(unique_families)
        n_train = int(n_families * self.train_frac)
        n_val = int(n_families * self.val_frac)

        train_families = set(shuffled[:n_train].tolist())
        val_families = set(shuffled[n_train:n_train + n_val].tolist())
        test_families = set(shuffled[n_train + n_val:].tolist())

        n_samples = len(family_ids)
        train_mask = np.array([family_ids[i] in train_families
                               for i in range(n_samples)])
        val_mask = np.array([family_ids[i] in val_families
                             for i in range(n_samples)])
        test_mask = np.array([family_ids[i] in test_families
                              for i in range(n_samples)])

        train_idx = np.where(train_mask)[0]
        val_idx = np.where(val_mask)[0]
        test_idx = np.where(test_mask)[0]

        return {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx,
            'train_families': sorted(train_families),
            'val_families': sorted(val_families),
            'test_families': sorted(test_families),
        }

    def temporal_split(self, family_ids: np.ndarray,
                       transition_type: np.ndarray,
                       seq_idx: np.ndarray,
                       ood_transition: int = 2,
                       ) -> Dict[str, np.ndarray]:
        """
        Family split + temporal OOD: test set includes held-out transition type.

        ood_transition=2 means R9→R10 is held out for temporal OOD testing.
        """
        base_split = self.split(family_ids, transition_type, seq_idx)

        test_idx = base_split['test']
        in_dist_mask = transition_type[test_idx] != ood_transition
        ood_mask = transition_type[test_idx] == ood_transition

        return {
            'train': base_split['train'],
            'val': base_split['val'],
            'test': test_idx[in_dist_mask],
            'test_ood_temporal': test_idx[ood_mask],
            'train_families': base_split['train_families'],
            'val_families': base_split['val_families'],
            'test_families': base_split['test_families'],
        }


class TargetNormalizer:
    """
    Fit normalization parameters from training data only.

    target_normalized = (target_value - mean) / std
    Parameters stored for inverse transform at evaluation time.
    """

    def __init__(self):
        self.mean: float = 0.0
        self.std: float = 1.0
        self.fitted = False

    def fit(self, target_values: np.ndarray, mask: np.ndarray):
        valid = target_values[mask]
        if len(valid) == 0:
            raise ValueError("No valid targets to fit normalizer")
        self.mean = float(np.mean(valid))
        self.std = float(np.std(valid))
        if self.std < 1e-8:
            self.std = 1.0
        self.fitted = True

    def transform(self, values: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted")
        return (values - self.mean) / self.std

    def inverse_transform(self, normalized: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Normalizer not fitted")
        return normalized * self.std + self.mean

    def state_dict(self) -> dict:
        return {'mean': self.mean, 'std': self.std, 'fitted': self.fitted}

    def load_state_dict(self, d: dict):
        self.mean = d['mean']
        self.std = d['std']
        self.fitted = d['fitted']


def build_etocore_dataset(
    data_path: str,
    seq_length: int = 36,
    min_total_count: int = 10,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
    kmer_k: int = 8,
    kmer_min_overlap_frac: float = 0.3,
) -> Dict:
    """
    Full pipeline: load → family assign → build transitions → split → normalize.

    Returns a dict containing:
      arrays: the full sample arrays
      splits: {'train': idx, 'val': idx, 'test': idx, ...}
      normalizer: fitted TargetNormalizer
      metadata: data statistics
    """
    print(f"Loading data from {data_path}")
    df = load_real_data(data_path, min_total_count=min_total_count,
                        seq_length=seq_length)
    n_seqs = len(df)

    print(f"Assigning family IDs (k={kmer_k}, min_overlap={kmer_min_overlap_frac})")
    family_ids = assign_family_ids(
        df['sequence'].tolist(), k=kmer_k, min_overlap_frac=kmer_min_overlap_frac
    )
    n_families = len(np.unique(family_ids))
    print(f"  {n_families} families from {n_seqs} sequences")

    print("Building transition samples")
    arrays = build_transition_samples(df, family_ids, seq_length=seq_length)
    total = arrays['target_value'].shape[0]
    print(f"  {total} total samples ({n_seqs} seqs × {len(VALID_TRANSITIONS)} transitions)")

    print("Splitting by family")
    splitter = FamilySplitter(train_frac=train_frac, val_frac=val_frac, seed=seed)
    splits = splitter.temporal_split(
        arrays['family_id'], arrays['transition_type'], arrays['seq_idx']
    )

    for split_name in ['train', 'val', 'test', 'test_ood_temporal']:
        if split_name in splits:
            idx = splits[split_name]
            n_split = len(idx)
            n_split_seqs = len(np.unique(arrays['seq_idx'][idx]))
            n_split_fams = len(np.unique(arrays['family_id'][idx]))
            print(f"  {split_name}: {n_split} samples, "
                  f"{n_split_seqs} seqs, {n_split_fams} families")

    print("Fitting target normalizer from training data")
    normalizer = TargetNormalizer()
    train_idx = splits['train']
    normalizer.fit(arrays['target_value'][train_idx],
                   arrays['target_mask'][train_idx])
    print(f"  mean={normalizer.mean:.4f}, std={normalizer.std:.4f}")

    target_norm = np.zeros_like(arrays['target_value'])
    target_norm[arrays['target_mask']] = normalizer.transform(
        arrays['target_value'][arrays['target_mask']]
    )
    arrays['target_normalized'] = target_norm

    train_families = set(splits['train_families'])
    val_families = set(splits['val_families'])
    test_families = set(splits['test_families'])
    overlap_tv = train_families & val_families
    overlap_tt = train_families & test_families
    overlap_vt = val_families & test_families
    assert len(overlap_tv) == 0, f"Train-val family overlap: {overlap_tv}"
    assert len(overlap_tt) == 0, f"Train-test family overlap: {overlap_tt}"
    assert len(overlap_vt) == 0, f"Val-test family overlap: {overlap_vt}"
    print("  Family split verified: zero overlap between splits")

    metadata = {
        'n_sequences': n_seqs,
        'n_families': n_families,
        'n_transitions': len(VALID_TRANSITIONS),
        'total_samples': total,
        'seq_length': seq_length,
        'valid_transitions': VALID_TRANSITIONS,
        'round_time': ROUND_TIME,
        'normalizer': normalizer.state_dict(),
        'split_seed': seed,
        'split_fractions': {'train': train_frac, 'val': val_frac},
    }

    return {
        'arrays': arrays,
        'splits': splits,
        'normalizer': normalizer,
        'metadata': metadata,
    }

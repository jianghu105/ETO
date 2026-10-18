"""
ETO Data Preprocessing Pipeline
================================

ETO learns a continuous evolutionary transition operator that estimates
latent evolutionary advantage under selective environments.

Core variables:
    Population state:          Z(t)
    Continuous transition:     Z(t+dt) = Z(t) + dt * T_theta(Z(t), E(t))
    Individual potential:      Phi_i(t,c) = f_theta(Z(t), x_i, E(t))

Phi represents:
    "the latent potential of a sequence to maintain or gain evolutionary
     advantage under a given selection context"

EPS = Evolutionary Potential Score (a.k.a. Evolutionary Advantage Potential)
"""

import os
import re
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Union


BASE_MAP = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
ROUND_NAMES = ["R3", "R5", "R9", "R10", "R11"]
ROUND_PAIRS = [("R3", "R5"), ("R5", "R9"), ("R9", "R10"), ("R10", "R11")]
CONTEXT_MAP = {
    "R3_R5": 0,
    "R5_R9": 1,
    "R9_R10": 2,
    "R10_R11": 3,
}

ROUND_INDEX_MAP = {
    "R3_R5": 0,
    "R5_R9": 1,
    "R9_R10": 2,
    "R10_R11": 3,
}

CONTINUOUS_TIME_MAP = {
    "R3_R5": 0.0,
    "R5_R9": 1.0,
    "R9_R10": 2.0,
    "R10_R11": 3.0,
}

EVOLUTION_TIME_MAP = {
    "R3": 0,
    "R5": 2,
    "R9": 6,
    "R10": 7,
    "R11": 8,
}


def one_hot_encode(sequence: str, seq_length: int = 36) -> np.ndarray:
    encoding = np.zeros((4, seq_length), dtype=np.float32)
    for i, base in enumerate(sequence.upper()):
        if base in BASE_MAP and i < seq_length:
            encoding[BASE_MAP[base], i] = 1.0
    return encoding


def compute_structure_features(sequence: str) -> np.ndarray:
    """
    Compute simple structure features for a sequence.

    Features:
      1. GC content: fraction of G+C bases
      2. MFE proxy: negative log of sequence complexity (simplified)
      3. Paired prob proxy: alternating pattern score

    Returns: (3,) array
    """
    seq = sequence.upper()
    n = len(seq)
    if n == 0:
        return np.zeros(3, dtype=np.float32)

    gc_count = sum(1 for c in seq if c in 'GC')
    gc_content = gc_count / n

    base_counts = {b: seq.count(b) for b in 'ACGT'}
    max_count = max(base_counts.values())
    mfe_proxy = -np.log(max_count / n + 1e-8)
    mfe_proxy = np.clip(mfe_proxy, 0, 2) / 2

    pair_score = 0
    for i in range(0, n - 1, 2):
        b1, b2 = seq[i], seq[i + 1]
        if (b1 in 'GC' and b2 in 'GC') or (b1 in 'AT' and b2 in 'AT'):
            pair_score += 1
    paired_prob = pair_score / (n // 2 + 1)

    return np.array([gc_content, mfe_proxy, paired_prob], dtype=np.float32)


def load_real_data(filepath: str, max_sequences: int = None,
                   min_total_count: int = 10,
                   seq_length: int = 36) -> pd.DataFrame:
    """
    Load real MPO-SELEX data from Excel, CSV, TSV, or Parquet.

    Auto-detects sequence column and round count columns.

    Data cleaning:
      - Remove non-ACGT characters
      - Filter to exact seq_length
      - Remove empty sequences
      - Remove duplicate sequences

    Returns DataFrame with columns: sequence, R3, R5, R9, R10, R11
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.xlsx' or ext == '.xls':
        df_raw = pd.read_excel(filepath)
    elif ext == '.csv':
        df_raw = pd.read_csv(filepath)
    elif ext == '.tsv':
        df_raw = pd.read_csv(filepath, sep='\t')
    elif ext == '.parquet':
        df_raw = pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

    col_names = df_raw.columns.tolist()

    seq_col = None
    for col in col_names:
        col_str = str(col).lower()
        if 'seq' in col_str or '36' in col_str:
            seq_col = col
            break
    if seq_col is None:
        for col in col_names:
            sample = df_raw[col].dropna().astype(str).head(20)
            if sample.str.match(r'^[ACGTacgt]+$').mean() > 0.8:
                seq_col = col
                break
    if seq_col is None:
        seq_col = col_names[1]

    round_cols = {}
    for rname in ROUND_NAMES:
        for col in col_names:
            if str(col).strip() == rname:
                round_cols[rname] = col
                break
    if len(round_cols) < len(ROUND_NAMES):
        count_cols = [c for c in col_names if c != seq_col
                      and pd.api.types.is_numeric_dtype(df_raw[c])]
        count_cols = [c for c in count_cols if str(c) not in ('Unnamed: 0', '')]
        if len(count_cols) >= len(ROUND_NAMES):
            round_cols = {rname: count_cols[i]
                          for i, rname in enumerate(ROUND_NAMES)}

    print(f"Detected sequence column: '{seq_col}'")
    print(f"Detected round columns: {round_cols}")
    print(f"Raw rows: {len(df_raw)}")

    df = df_raw[[seq_col] + list(round_cols.values())].copy()
    df.columns = ['sequence'] + ROUND_NAMES

    df = df.dropna(subset=['sequence'])
    df['sequence'] = df['sequence'].astype(str).str.upper()
    df = df[df['sequence'].str.match(r'^[ACGT]+$')]
    df = df[df['sequence'].str.len() == seq_length]
    df = df.drop_duplicates(subset='sequence').reset_index(drop=True)

    for r in ROUND_NAMES:
        df[r] = pd.to_numeric(df[r], errors='coerce').fillna(0).astype(int)
        df[r] = df[r].clip(lower=0)

    total_counts = df[ROUND_NAMES].sum(axis=1)
    df = df[total_counts >= min_total_count].reset_index(drop=True)

    if max_sequences is not None and len(df) > max_sequences:
        df = df.head(max_sequences).reset_index(drop=True)

    print(f"After cleaning: {len(df)} valid {seq_length}nt sequences")
    return df


def generate_synthetic_data(n_sequences: int = 2000, n_families: int = 20,
                            seq_length: int = 36, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    records = []

    family_motifs = []
    for _ in range(n_families):
        motif_len = rng.randint(6, 12)
        motif = ''.join(rng.choice(list('ACGT'), size=motif_len))
        family_motifs.append(motif)

    seqs_per_family = n_sequences // n_families

    for fam_idx in range(n_families):
        motif = family_motifs[fam_idx]
        motif_len = len(motif)

        is_enriched = fam_idx < n_families // 3
        is_counter_selected = (fam_idx >= n_families // 3) and (fam_idx < n_families * 2 // 3)
        is_transient = fam_idx >= n_families * 2 // 3

        for j in range(seqs_per_family):
            seq_list = list(motif)
            while len(seq_list) < seq_length:
                seq_list.append(rng.choice(list('ACGT')))
            n_mut = rng.randint(0, 5)
            for _ in range(n_mut):
                pos = rng.randint(0, seq_length)
                seq_list[pos] = rng.choice(list('ACGT'))
            sequence = ''.join(seq_list)

            base_count = rng.randint(5, 50)

            if is_enriched:
                counts = {
                    'R3': base_count,
                    'R5': int(base_count * rng.uniform(1.5, 3.0)),
                    'R9': int(base_count * rng.uniform(0.01, 0.1)),
                    'R10': int(base_count * rng.uniform(5.0, 15.0)),
                    'R11': int(base_count * rng.uniform(8.0, 20.0)),
                }
            elif is_counter_selected:
                counts = {
                    'R3': base_count,
                    'R5': int(base_count * rng.uniform(1.0, 2.0)),
                    'R9': int(base_count * rng.uniform(0.0, 0.05)),
                    'R10': int(base_count * rng.uniform(0.5, 1.5)),
                    'R11': int(base_count * rng.uniform(0.1, 0.5)),
                }
            elif is_transient:
                counts = {
                    'R3': base_count,
                    'R5': int(base_count * rng.uniform(2.0, 5.0)),
                    'R9': int(base_count * rng.uniform(3.0, 8.0)),
                    'R10': int(base_count * rng.uniform(0.5, 1.0)),
                    'R11': int(base_count * rng.uniform(0.01, 0.1)),
                }
            else:
                counts = {
                    'R3': base_count,
                    'R5': int(base_count * rng.uniform(0.8, 1.5)),
                    'R9': int(base_count * rng.uniform(0.01, 0.05)),
                    'R10': int(base_count * rng.uniform(0.5, 1.0)),
                    'R11': int(base_count * rng.uniform(0.1, 0.3)),
                }

            noise = rng.poisson(1, size=5)
            for i, r in enumerate(ROUND_NAMES):
                counts[r] = max(0, counts[r] + noise[i])

            records.append({'sequence': sequence, **counts})

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset='sequence').reset_index(drop=True)
    print(f"Generated {len(df)} synthetic sequences across {n_families} families")
    return df


def compute_evolutionary_advantage(counts: Dict[str, np.ndarray],
                                   log_counts: Dict[str, np.ndarray],
                                   r_curr: str,
                                   future_rounds: List[str]
                                   ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute observational evolutionary advantage from round trajectory.

    EA = (normalize(enrichment) + normalize(persistence) + normalize(consistency)) / 3

    This is a self-supervised target, NOT ground truth affinity.

    Returns:
        evolutionary_advantage: (N,) array in [0, 1]
        ea_valid_mask: (N,) bool array — False where EA is undefined
                       (no future rounds to compute from)
    """
    n = log_counts[r_curr].shape[0]

    if not future_rounds:
        return np.zeros(n, dtype=np.float32), np.zeros(n, dtype=bool)

    r_final = future_rounds[-1]
    enrichment = np.log1p(counts[r_final]) - np.log1p(counts[r_curr])

    future_log_counts = np.stack(
        [log_counts[r] for r in future_rounds], axis=0
    )
    persistence = future_log_counts.mean(axis=0) - log_counts[r_curr]

    if len(future_rounds) >= 2:
        growth_rates = np.diff(future_log_counts, axis=0)
        if growth_rates.shape[0] >= 2:
            step_means = growth_rates.mean(axis=0)
            step_stds = growth_rates.std(axis=0) + 1e-8
            consistency = step_means / step_stds
            consistency = np.tanh(consistency)
        else:
            curr_log = log_counts[r_curr]
            future_log = log_counts[future_rounds[0]]
            growth_sign = np.sign(future_log - curr_log)
            consistency = growth_sign
    elif len(future_rounds) == 1:
        growth = log_counts[future_rounds[0]] - log_counts[r_curr]
        consistency = np.tanh(growth)
    else:
        persistence = np.zeros(n, dtype=np.float32)
        consistency = np.zeros(n, dtype=np.float32)

    def rank_norm(x):
        order = x.argsort()
        ranks = np.empty_like(order, dtype=np.float32)
        ranks[order] = np.arange(len(x), dtype=np.float32)
        return ranks / max(len(x) - 1, 1)

    e_norm = rank_norm(enrichment)
    p_norm = rank_norm(persistence)
    c_norm = rank_norm(consistency)

    ea = (e_norm + p_norm + c_norm) / 3.0
    valid = np.ones(n, dtype=bool)
    return ea, valid


def compute_features(df: pd.DataFrame, round_pairs: List[Tuple[str, str]] = None
                     ) -> Dict[str, np.ndarray]:
    """
    Compute all features for training.

    Outputs:
      - one_hot: (N, 4, L) sequence encoding
      - structure_features: (N, 3) GC content, MFE proxy, paired prob
      - counts: dict of round -> absolute counts
      - log_counts: dict of round -> log1p(count)
      - relative_abundance: dict of round -> count / total_count_round
      - velocities: dict of pair_key -> log count difference
      - evolutionary_advantage: dict of pair_key -> EA score
      - ea_valid_mask: dict of pair_key -> bool mask
      - round_diagnostics: dict of round -> {zero_frac, total_count, ...}
    """
    if round_pairs is None:
        round_pairs = ROUND_PAIRS

    n = len(df)
    seq_length = df['sequence'].str.len().iloc[0] if n > 0 else 36

    one_hot = np.zeros((n, 4, seq_length), dtype=np.float32)
    structure_features = np.zeros((n, 3), dtype=np.float32)
    for i, seq in enumerate(df['sequence'].values):
        one_hot[i] = one_hot_encode(seq, seq_length)
        structure_features[i] = compute_structure_features(seq)

    counts = {}
    for r in ROUND_NAMES:
        counts[r] = df[r].values.astype(np.float32)

    log_counts = {}
    for r in ROUND_NAMES:
        log_counts[r] = np.log1p(counts[r])

    relative_abundance = {}
    for r in ROUND_NAMES:
        total = counts[r].sum()
        relative_abundance[r] = counts[r] / (total + 1e-8)

    velocities = {}
    for (r_prev, r_curr) in round_pairs:
        v = np.log1p(counts[r_curr]) - np.log1p(counts[r_prev])
        velocities[f"{r_prev}_{r_curr}"] = v

    evolutionary_advantage = {}
    ea_valid_mask = {}

    future_rounds_map = {
        ("R3", "R5"): ["R9", "R10", "R11"],
        ("R5", "R9"): ["R10", "R11"],
        ("R9", "R10"): ["R11"],
        ("R10", "R11"): [],
    }

    for (r_prev, r_curr) in round_pairs:
        context_key = f"{r_prev}_{r_curr}"
        future_rounds = future_rounds_map.get((r_prev, r_curr), [])

        ea, valid = compute_evolutionary_advantage(
            counts, log_counts, r_curr, future_rounds
        )
        evolutionary_advantage[context_key] = ea
        ea_valid_mask[context_key] = valid

    round_diagnostics = {}
    for r in ROUND_NAMES:
        zero_frac = (counts[r] == 0).mean()
        round_diagnostics[r] = {
            'zero_frac': float(zero_frac),
            'total_count': float(counts[r].sum()),
            'mean_count': float(counts[r].mean()),
            'max_count': float(counts[r].max()),
        }
        if zero_frac > 0.5:
            print(f"  WARNING: {r} has {zero_frac*100:.1f}% zero counts — "
                  f"transitions involving {r} as current round have degraded signal")

    return {
        'one_hot': one_hot,
        'structure_features': structure_features,
        'counts': counts,
        'log_counts': log_counts,
        'relative_abundance': relative_abundance,
        'velocities': velocities,
        'evolutionary_advantage': evolutionary_advantage,
        'ea_valid_mask': ea_valid_mask,
        'round_diagnostics': round_diagnostics,
    }


def prepare_training_data(features: Dict, round_pairs: List[Tuple[str, str]] = None
                          ) -> Dict[str, np.ndarray]:
    """
    Build compact array-based training data (memory-efficient).

    Returns a dict of stacked arrays for all transitions.
    Use CompactSELEXDataset to index into these arrays.
    """
    if round_pairs is None:
        round_pairs = ROUND_PAIRS

    n = features['one_hot'].shape[0]
    n_pairs = len(round_pairs)
    total = n * n_pairs
    seq_length = features['one_hot'].shape[2]

    one_hot = np.zeros((total, 4, seq_length), dtype=np.float32)
    structure_features = np.zeros((total, 3), dtype=np.float32)
    x_prev = np.zeros((total, 2), dtype=np.float32)
    x_curr = np.zeros((total, 2), dtype=np.float32)
    count_prev = np.zeros(total, dtype=np.float32)
    count_curr = np.zeros(total, dtype=np.float32)
    log_count_prev = np.zeros(total, dtype=np.float32)
    log_count_curr = np.zeros(total, dtype=np.float32)
    relative_abundance_curr = np.zeros(total, dtype=np.float32)
    evolutionary_advantage = np.zeros(total, dtype=np.float32)
    ea_valid = np.zeros(total, dtype=bool)
    context_id = np.zeros(total, dtype=np.int64)
    round_idx = np.zeros(total, dtype=np.int64)
    continuous_time = np.zeros(total, dtype=np.float32)
    delta_tau = np.zeros(total, dtype=np.float32)
    pair_idx = np.zeros(total, dtype=np.int64)
    seq_idx = np.zeros(total, dtype=np.int64)

    for pi, (r_prev, r_curr) in enumerate(round_pairs):
        context_key = f"{r_prev}_{r_curr}"
        cid = CONTEXT_MAP[context_key]
        ridx = ROUND_INDEX_MAP[context_key]
        tau_prev = EVOLUTION_TIME_MAP[r_prev]
        tau_curr = EVOLUTION_TIME_MAP[r_curr]
        dt = tau_curr - tau_prev

        start = pi * n
        end = (pi + 1) * n

        one_hot[start:end] = features['one_hot']
        structure_features[start:end] = features['structure_features']

        x_prev[start:end, 0] = features['log_counts'][r_prev]
        x_prev[start:end, 1] = features['velocities'].get(
            context_key, np.zeros(n, dtype=np.float32))

        x_curr[start:end, 0] = features['log_counts'][r_curr]
        x_curr[start:end, 1] = (np.log1p(features['counts'][r_curr]) -
                                 np.log1p(features['counts'][r_prev]))

        count_prev[start:end] = features['counts'][r_prev]
        count_curr[start:end] = features['counts'][r_curr]
        log_count_prev[start:end] = features['log_counts'][r_prev]
        log_count_curr[start:end] = features['log_counts'][r_curr]
        relative_abundance_curr[start:end] = features['relative_abundance'][r_curr]
        evolutionary_advantage[start:end] = features['evolutionary_advantage'].get(
            context_key, np.zeros(n, dtype=np.float32))
        ea_valid[start:end] = features['ea_valid_mask'].get(
            context_key, np.ones(n, dtype=bool))
        context_id[start:end] = cid
        round_idx[start:end] = ridx
        continuous_time[start:end] = float(tau_curr)
        delta_tau[start:end] = float(dt)
        pair_idx[start:end] = pi
        seq_idx[start:end] = np.arange(n)

    return {
        'one_hot': one_hot,
        'structure_features': structure_features,
        'x_prev': x_prev,
        'x_curr': x_curr,
        'count_prev': count_prev,
        'count_curr': count_curr,
        'log_count_prev': log_count_prev,
        'log_count_curr': log_count_curr,
        'relative_abundance_curr': relative_abundance_curr,
        'evolutionary_advantage': evolutionary_advantage,
        'ea_valid': ea_valid,
        'context_id': context_id,
        'round_idx': round_idx,
        'continuous_time': continuous_time,
        'delta_tau': delta_tau,
        'pair_idx': pair_idx,
        'seq_idx': seq_idx,
    }

"""
Unified evaluation metrics for ETO and baselines.

Metrics:
  - Spearman correlation (EPS vs evolutionary_advantage)
  - NDCG@K
  - Precision@K
  - Low-abundance discovery rate
  - Per-transition breakdown (R3→R5, R5→R9, R9→R10, R10→R11)
  - EA validity mask respected (R10→R11 excluded from EA metrics)
"""

import torch
import numpy as np
from scipy import stats
from typing import Dict, List, Optional, Tuple


def compute_spearman(scores: np.ndarray, targets: np.ndarray) -> float:
    if len(scores) < 3:
        return float('nan')
    corr, pval = stats.spearmanr(scores, targets)
    return float(corr)


def compute_ndcg(scores: np.ndarray, targets: np.ndarray, k: int = 50) -> float:
    n = len(scores)
    k = min(k, n)
    relevance = targets.copy()

    order_pred = np.argsort(-scores)
    order_true = np.argsort(-relevance)

    dcg = 0.0
    for i in range(k):
        idx = order_pred[i]
        rel = relevance[idx]
        dcg += (2 ** rel - 1) / np.log2(i + 2)

    idcg = 0.0
    for i in range(k):
        rel = relevance[order_true[i]]
        idcg += (2 ** rel - 1) / np.log2(i + 2)

    return dcg / max(idcg, 1e-8)


def compute_precision_at_k(scores: np.ndarray, targets: np.ndarray,
                           k: int = 50, top_frac: float = 0.1) -> float:
    n = len(scores)
    k_pred = max(1, int(n * top_frac))

    top_pred = set(np.argsort(-scores)[:k_pred].tolist())
    threshold = np.percentile(targets, 100 * (1 - top_frac))
    true_positives = set(np.where(targets >= threshold)[0].tolist())

    if len(true_positives) == 0:
        return 0.0
    return len(top_pred & true_positives) / k_pred


def compute_low_abundance_discovery(scores: np.ndarray,
                                    targets: np.ndarray,
                                    current_abundance: np.ndarray,
                                    bottom_quantile: float = 0.25,
                                    top_frac: float = 0.1) -> float:
    low_abund_mask = current_abundance <= np.quantile(current_abundance, bottom_quantile)
    high_ea_threshold = np.quantile(targets, 1 - top_frac)
    hidden_gems = low_abund_mask & (targets >= high_ea_threshold)

    if hidden_gems.sum() == 0:
        return 0.0

    n_top = max(1, int(len(scores) * top_frac))
    top_pred = set(np.argsort(-scores)[:n_top].tolist())
    gem_indices = set(np.where(hidden_gems)[0].tolist())

    return len(top_pred & gem_indices) / max(1, len(gem_indices))


def unified_evaluation(scores: np.ndarray,
                       evolutionary_advantage: np.ndarray,
                       current_abundance: np.ndarray = None,
                       ea_valid: np.ndarray = None,
                       pair_indices: np.ndarray = None,
                       k_list: List[int] = None) -> Dict[str, float]:
    """
    Unified evaluation with per-transition breakdown.

    Args:
        scores: predicted scores (EPS, baseline scores, etc.)
        evolutionary_advantage: target EA labels
        current_abundance: current round counts (for low-abundance discovery)
        ea_valid: bool mask — True where EA is defined (excludes R10→R11)
        pair_indices: int array mapping each sample to its transition (0-3)
        k_list: list of K values for NDCG/Precision

    Returns:
        metrics dict with overall and per-transition keys
    """
    if k_list is None:
        k_list = [10, 50, 100]

    metrics = {}

    if ea_valid is None:
        ea_valid = np.ones(len(scores), dtype=bool)

    valid_mask = ea_valid.astype(bool)
    n_valid = valid_mask.sum()
    n_invalid = (~valid_mask).sum()
    metrics['n_valid_ea'] = int(n_valid)
    metrics['n_invalid_ea'] = int(n_invalid)

    if n_valid >= 3:
        scores_valid = scores[valid_mask]
        ea_valid_arr = evolutionary_advantage[valid_mask]

        metrics['spearman'] = compute_spearman(scores_valid, ea_valid_arr)

        for k in k_list:
            metrics[f'ndcg@{k}'] = compute_ndcg(scores_valid, ea_valid_arr, k=k)
            metrics[f'precision@{k}'] = compute_precision_at_k(
                scores_valid, ea_valid_arr, k=k
            )

        if current_abundance is not None:
            abund_valid = current_abundance[valid_mask]
            metrics['low_abundance_discovery'] = compute_low_abundance_discovery(
                scores_valid, ea_valid_arr, abund_valid
            )
            metrics['spearman_vs_abundance'] = compute_spearman(
                scores_valid, abund_valid
            )
    else:
        metrics['spearman'] = float('nan')

    TRANSITION_NAMES = ["R3_R5", "R5_R9", "R9_R10", "R10_R11"]
    if pair_indices is not None:
        for t_idx, t_name in enumerate(TRANSITION_NAMES):
            t_mask = pair_indices == t_idx
            t_valid = t_mask & valid_mask
            n_t = t_mask.sum()
            n_t_valid = t_valid.sum()

            prefix = f'transition/{t_name}'
            metrics[f'{prefix}/n_total'] = int(n_t)
            metrics[f'{prefix}/n_valid_ea'] = int(n_t_valid)

            if n_t_valid >= 3:
                t_scores = scores[t_valid]
                t_ea = evolutionary_advantage[t_valid]
                metrics[f'{prefix}/spearman'] = compute_spearman(t_scores, t_ea)
                metrics[f'{prefix}/ndcg@50'] = compute_ndcg(t_scores, t_ea, k=50)

                if current_abundance is not None:
                    t_abund = current_abundance[t_valid]
                    metrics[f'{prefix}/spearman_vs_abundance'] = compute_spearman(
                        t_scores, t_abund
                    )
                    metrics[f'{prefix}/low_abundance_discovery'] = \
                        compute_low_abundance_discovery(t_scores, t_ea, t_abund)

            if n_t >= 3:
                t_scores_all = scores[t_mask]
                t_counts = current_abundance[t_mask] if current_abundance is not None else None
                if t_counts is not None:
                    metrics[f'{prefix}/spearman_vs_count'] = compute_spearman(
                        t_scores_all, t_counts
                    )

    return metrics


def evaluate_model(model, dataloader, device: str = 'cuda') -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Evaluate ETO model with per-transition breakdown."""
    model.eval()
    all_eps = []
    all_ea = []
    all_abundance = []
    all_ea_valid = []
    all_pair_idx = []

    with torch.no_grad():
        for batch in dataloader:
            batch_device = {k: v.to(device) for k, v in batch.items()}
            outputs = model(batch_device)

            eps_key = 'eps' if 'eps' in outputs else 'phi'
            all_eps.append(outputs[eps_key].cpu().numpy())
            all_ea.append(batch['evolutionary_advantage'].numpy())
            all_abundance.append(batch['log_count_curr'].numpy())

            if 'ea_valid' in batch:
                all_ea_valid.append(batch['ea_valid'].numpy())
            else:
                all_ea_valid.append(np.ones(len(batch['evolutionary_advantage']), dtype=bool))

            if 'pair_idx' in batch:
                all_pair_idx.append(batch['pair_idx'].numpy())
            else:
                all_pair_idx.append(np.zeros(len(batch['evolutionary_advantage']), dtype=int))

    eps = np.concatenate(all_eps)
    ea = np.concatenate(all_ea)
    abundance = np.concatenate(all_abundance)
    ea_valid = np.concatenate(all_ea_valid)
    pair_idx = np.concatenate(all_pair_idx)

    metrics = unified_evaluation(
        eps, ea, abundance,
        ea_valid=ea_valid,
        pair_indices=pair_idx,
    )
    return metrics, eps, ea

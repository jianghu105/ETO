"""
Search Scoring: Prediction + Uncertainty + Novelty
===================================================

Scores candidates for budget-constrained search.
Supports ablation: prediction-only, +uncertainty, +novelty, full.
"""

import numpy as np
import torch
from typing import Dict, List, Optional
from collections import defaultdict


def compute_kmer_similarity(seq1: str, seq2: str, k: int = 4) -> float:
    """Jaccard similarity of k-mer sets."""
    kmers1 = {seq1[i:i + k] for i in range(len(seq1) - k + 1)}
    kmers2 = {seq2[i:i + k] for i in range(len(seq2) - k + 1)}
    if not kmers1 or not kmers2:
        return 0.0
    return len(kmers1 & kmers2) / len(kmers1 | kmers2)


def compute_novelty(candidate: str, selected: List[str], k: int = 4) -> float:
    """
    Novelty = 1 - max k-mer similarity to any selected sequence.
    Higher = more novel.
    """
    if not selected:
        return 1.0
    max_sim = max(compute_kmer_similarity(candidate, s, k) for s in selected)
    return 1.0 - max_sim


def ensemble_predict(models: List, batch: Dict, device: str = 'cpu'
                     ) -> tuple:
    """
    Run ensemble prediction. Returns (mean_pred, std_pred).
    """
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            batch_gpu = {k: v.to(device) for k, v in batch.items()}
            out = model(batch_gpu)
            predictions.append(out['prediction'].cpu().numpy())

    preds = np.stack(predictions, axis=0)
    return preds.mean(axis=0), preds.std(axis=0)


class SearchScorer:
    """
    Configurable scoring function:
      score = mean_pred + beta * uncertainty + gamma * novelty

    Ablation modes:
      'prediction': only mean prediction
      'prediction+uncertainty': + ensemble std
      'prediction+novelty': + diversity bonus
      'full': all three components
    """

    def __init__(self, beta: float = 0.5, gamma: float = 0.3,
                 mode: str = 'full', kmer_k: int = 4):
        self.beta = beta
        self.gamma = gamma
        self.mode = mode
        self.kmer_k = kmer_k

    def score(self, mean_pred: np.ndarray, std_pred: np.ndarray,
              candidates: List[str], selected: List[str]) -> np.ndarray:
        """
        Score a batch of candidates.

        Args:
            mean_pred: ensemble mean predictions (N,)
            std_pred: ensemble std predictions (N,)
            candidates: list of candidate sequences
            selected: already-selected sequences (for novelty)

        Returns:
            scores: (N,) array
        """
        n = len(candidates)
        scores = mean_pred.copy()

        if self.mode in ('prediction+uncertainty', 'full'):
            scores = scores + self.beta * std_pred

        if self.mode in ('prediction+novelty', 'full'):
            novelties = np.array([
                compute_novelty(c, selected, self.kmer_k)
                for c in candidates
            ])
            scores = scores + self.gamma * novelties

        return scores

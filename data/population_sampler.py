"""
Population Sampling Strategies for Large-Scale SELEX Data.

For 300K+ sequences, we need smart sampling to ensure:
  - High-abundance sequences are well-represented (70%)
  - Rare/diverse sequences are explored (30%)

This prevents the model from only learning common patterns
while missing potentially interesting low-abundance binders.
"""

import torch
import numpy as np
from torch.utils.data import Sampler, Dataset, DataLoader
from typing import Dict, List, Optional, Iterator


class PopulationSampler(Sampler):
    """
    Mixed sampling strategy for large SELEX populations.

    Combines:
      - Abundance-weighted sampling (fraction `abundance_frac`)
      - Uniform/random sampling for rare/diverse sequences (1 - abundance_frac)
    """

    def __init__(self, n_sequences: int,
                 counts: Optional[Dict[str, np.ndarray]] = None,
                 current_round: str = 'R10',
                 abundance_frac: float = 0.7,
                 batch_size: int = 256,
                 seed: int = 42):
        self.n_sequences = n_sequences
        self.abundance_frac = abundance_frac
        self.batch_size = batch_size
        self.rng = np.random.RandomState(seed)

        if counts is not None and current_round in counts:
            c = counts[current_round].astype(np.float64)
            self.abundance_weights = (c + 1.0)
            self.abundance_weights /= self.abundance_weights.sum()
        else:
            self.abundance_weights = np.ones(n_sequences) / n_sequences

        uniform_weights = np.ones(n_sequences) / n_sequences
        self.mixed_weights = (
            abundance_frac * self.abundance_weights +
            (1 - abundance_frac) * uniform_weights
        )
        self.mixed_weights /= self.mixed_weights.sum()

        self.n_samples = n_sequences

    def __iter__(self) -> Iterator[int]:
        indices = self.rng.choice(
            self.n_sequences,
            size=self.n_samples,
            replace=True,
            p=self.mixed_weights,
        )
        return iter(indices.tolist())

    def __len__(self) -> int:
        return self.n_samples


class CompactSELEXRealDataset(Dataset):
    """
    Memory-efficient dataset that indexes into compact arrays.
    """

    def __init__(self, arrays: Dict[str, np.ndarray],
                 count_dropout_prob: float = 0.0):
        self.arrays = arrays
        self.count_dropout_prob = count_dropout_prob
        self.n = len(arrays['pair_idx'])

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        sample = {
            'one_hot': torch.from_numpy(self.arrays['one_hot'][idx]),
            'structure_features': torch.from_numpy(self.arrays['structure_features'][idx]),
            'x_prev': torch.from_numpy(self.arrays['x_prev'][idx]),
            'x_curr': torch.from_numpy(self.arrays['x_curr'][idx]),
            'count_prev': torch.tensor(self.arrays['count_prev'][idx], dtype=torch.float32),
            'count_curr': torch.tensor(self.arrays['count_curr'][idx], dtype=torch.float32),
            'log_count_prev': torch.tensor(self.arrays['log_count_prev'][idx], dtype=torch.float32),
            'log_count_curr': torch.tensor(self.arrays['log_count_curr'][idx], dtype=torch.float32),
            'relative_abundance_curr': torch.tensor(self.arrays['relative_abundance_curr'][idx], dtype=torch.float32),
            'evolutionary_advantage': torch.tensor(self.arrays['evolutionary_advantage'][idx], dtype=torch.float32),
            'ea_valid': torch.tensor(self.arrays['ea_valid'][idx], dtype=torch.bool),
            'context_id': torch.tensor(self.arrays['context_id'][idx], dtype=torch.long),
            'round_idx': torch.tensor(self.arrays['round_idx'][idx], dtype=torch.long),
            'continuous_time': torch.tensor(self.arrays['continuous_time'][idx], dtype=torch.float32),
            'delta_tau': torch.tensor(self.arrays['delta_tau'][idx], dtype=torch.float32),
            'pair_idx': torch.tensor(self.arrays['pair_idx'][idx], dtype=torch.long),
            'seq_idx': torch.tensor(self.arrays['seq_idx'][idx], dtype=torch.long),
        }

        if self.count_dropout_prob > 0:
            if np.random.rand() < self.count_dropout_prob:
                sample['x_prev'][0] = 0.0
            if np.random.rand() < self.count_dropout_prob:
                sample['x_curr'][0] = 0.0

        return sample


class SELEXRealDataset(Dataset):
    """Legacy dict-based dataset (kept for backward compatibility)."""

    def __init__(self, samples: List[Dict[str, torch.Tensor]],
                 count_dropout_prob: float = 0.0):
        self.samples = samples
        self.count_dropout_prob = count_dropout_prob

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = {k: v.clone() for k, v in self.samples[idx].items()}

        if self.count_dropout_prob > 0:
            if torch.rand(1).item() < self.count_dropout_prob:
                if 'x_prev' in sample:
                    sample['x_prev'][0] = 0.0
            if torch.rand(1).item() < self.count_dropout_prob:
                if 'x_curr' in sample:
                    sample['x_curr'][0] = 0.0

        return sample


def build_population_dataloader(arrays: Dict[str, np.ndarray],
                                features: Dict,
                                config: dict,
                                round_pair_idx: int = None) -> DataLoader:
    """
    Build a DataLoader with population-aware sampling.

    Args:
        arrays: compact training arrays from prepare_training_data
        features: computed features (includes counts for weighting)
        config: training config
        round_pair_idx: if set, only sample from this round pair

    Returns:
        DataLoader with PopulationSampler
    """
    training_cfg = config.get('training', {})
    sampling_cfg = config.get('sampling', {})

    if round_pair_idx is not None:
        mask = arrays['pair_idx'] == round_pair_idx
        filtered = {k: v[mask] for k, v in arrays.items()}
    else:
        filtered = arrays

    dataset = CompactSELEXRealDataset(
        filtered,
        count_dropout_prob=training_cfg.get('count_dropout_prob', 0.0),
    )

    n_seq = features['one_hot'].shape[0]
    sampler = PopulationSampler(
        n_sequences=n_seq,
        counts=features.get('counts'),
        current_round=sampling_cfg.get('weight_round', 'R10'),
        abundance_frac=sampling_cfg.get('abundance_frac', 0.7),
        batch_size=training_cfg.get('batch_size', 256),
        seed=training_cfg.get('seed', 42),
    )

    return DataLoader(
        dataset,
        batch_size=training_cfg.get('batch_size', 256),
        sampler=sampler,
        num_workers=training_cfg.get('num_workers', 0),
        pin_memory=True,
        drop_last=False,
    )

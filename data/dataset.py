import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from .preprocessing import (
    load_real_data, generate_synthetic_data, compute_features,
    prepare_training_data, ROUND_NAMES, ROUND_PAIRS, CONTEXT_MAP,
    EVOLUTION_TIME_MAP,
)


class CompactSELEXDataset(Dataset):
    """
    Memory-efficient Dataset that indexes into compact arrays.

    Instead of storing 909K individual dicts, stores stacked numpy arrays
    and creates tensors on-the-fly in __getitem__.
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


class SELEXDataset(Dataset):
    """
    Legacy dict-based Dataset (kept for backward compatibility with baselines).
    """

    def __init__(self, samples: List[Dict[str, torch.Tensor]],
                 count_dropout_prob: float = 0.0,
                 trajectory_shuffle: bool = False):
        self.samples = samples
        self.count_dropout_prob = count_dropout_prob
        self.trajectory_shuffle = trajectory_shuffle

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = {k: v.clone() for k, v in self.samples[idx].items()}

        if self.count_dropout_prob > 0:
            if torch.rand(1).item() < self.count_dropout_prob:
                sample['x_prev'][0] = 0.0
            if torch.rand(1).item() < self.count_dropout_prob:
                sample['x_curr'][0] = 0.0

        return sample


class SELEXDataModule:
    def __init__(self, config: dict):
        self.config = config
        self.data_config = config['data']
        self.training_config = config['training']
        self.features = None
        self.arrays = None
        self.df = None

    def load_data(self):
        synthetic_cfg = self.data_config.get('synthetic', {})
        if synthetic_cfg.get('enabled', False):
            self.df = generate_synthetic_data(
                n_sequences=synthetic_cfg.get('n_sequences', 2000),
                n_families=synthetic_cfg.get('n_families', 20),
                seed=synthetic_cfg.get('seed', 42),
            )
        else:
            data_path = self.data_config.get('real_data_path',
                                             'dataset/20230531AptDMC双端-MPO.xlsx')
            self.df = load_real_data(
                data_path,
                max_sequences=self.data_config.get('max_sequences', None),
                min_total_count=self.data_config.get('min_total_count', 10),
                seq_length=self.data_config.get('seq_length', 36),
            )

        self.features = compute_features(self.df)
        self.arrays = prepare_training_data(self.features)
        print(f"Total training samples: {len(self.arrays['pair_idx'])}")

    def get_dataloader(self, shuffle: bool = True) -> DataLoader:
        dataset = CompactSELEXDataset(
            self.arrays,
            count_dropout_prob=self.training_config.get('count_dropout_prob', 0.0),
        )
        return DataLoader(
            dataset,
            batch_size=self.training_config.get('batch_size', 256),
            shuffle=shuffle,
            num_workers=self.training_config.get('num_workers', 0),
            pin_memory=True,
            drop_last=False,
        )

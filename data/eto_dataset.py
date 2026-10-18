"""
ETO-Core PyTorch Dataset for unified data interface.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional


class ETODataset(Dataset):
    """
    Dataset for ETO-Core unified interface.

    Indexes into the arrays produced by unified_dataset.build_transition_samples,
    restricted to a specific split via index_array.
    """

    def __init__(self, arrays: Dict[str, np.ndarray],
                 index_array: np.ndarray,
                 target_key: str = 'target_normalized'):
        self.arrays = arrays
        self.indices = index_array
        self.target_key = target_key
        self.n = len(index_array)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        real_idx = int(self.indices[idx])

        sample = {
            'one_hot': torch.from_numpy(self.arrays['one_hot'][real_idx]),
            'structure_features': torch.from_numpy(
                self.arrays['structure_features'][real_idx]),
            'history_counts': torch.from_numpy(
                self.arrays['history_counts'][real_idx]),
            'history_mask': torch.from_numpy(
                self.arrays['history_mask'][real_idx]),
            'transition_type': torch.tensor(
                int(self.arrays['transition_type'][real_idx]), dtype=torch.long),
            'round_from': torch.tensor(
                int(self.arrays['round_from'][real_idx]), dtype=torch.long),
            'round_to': torch.tensor(
                int(self.arrays['round_to'][real_idx]), dtype=torch.long),
            'delta_t': torch.tensor(
                float(self.arrays['delta_t'][real_idx]), dtype=torch.float32),
            'target': torch.tensor(
                float(self.arrays[self.target_key][real_idx]), dtype=torch.float32),
            'target_mask': torch.tensor(
                bool(self.arrays['target_mask'][real_idx])),
            'family_id': torch.tensor(
                int(self.arrays['family_id'][real_idx]), dtype=torch.long),
            'seq_idx': torch.tensor(
                int(self.arrays['seq_idx'][real_idx]), dtype=torch.long),
        }

        return sample


def build_dataloaders(dataset_dict: Dict, batch_size: int = 256,
                      num_workers: int = 0,
                      ) -> Dict[str, DataLoader]:
    """
    Build train/val/test DataLoaders from the unified dataset dict.
    """
    arrays = dataset_dict['arrays']
    splits = dataset_dict['splits']

    loaders = {}
    for split_name in ['train', 'val', 'test']:
        if split_name not in splits:
            continue
        ds = ETODataset(arrays, splits[split_name])
        loaders[split_name] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split_name == 'train'),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split_name == 'train'),
        )

    if 'test_ood_temporal' in splits and len(splits['test_ood_temporal']) > 0:
        ds_ood = ETODataset(arrays, splits['test_ood_temporal'])
        loaders['test_ood_temporal'] = DataLoader(
            ds_ood,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return loaders

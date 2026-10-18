"""
ETO-Core Data Pipeline Tests
=============================

Tests 1-6 from the task specification:
  1. Data field completeness
  2. Missing values and mask correctness
  3. R10→R11 invalid EA exclusion
  4. Time information leakage
  5. Sequence family split
  6. Target normalization (train-only fit)
"""

import sys
import os
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.preprocessing import generate_synthetic_data, ROUND_NAMES
from data.unified_dataset import (
    build_transition_samples, assign_family_ids,
    FamilySplitter, TargetNormalizer, build_etocore_dataset,
    VALID_TRANSITIONS, ROUND_TIME,
)
from data.eto_dataset import ETODataset, build_dataloaders


def make_test_df(n=200):
    return generate_synthetic_data(n_sequences=n, n_families=10, seed=42)


class TestDataFieldCompleteness:
    """Test 1: All required fields exist with correct shapes and types."""

    def test_all_fields_present(self):
        df = make_test_df(100)
        fam = assign_family_ids(df['sequence'].tolist())
        arrays = build_transition_samples(df, fam)

        required_fields = [
            'one_hot', 'structure_features', 'history_counts', 'history_mask',
            'transition_type', 'round_from', 'round_to', 'delta_t',
            'target_value', 'target_mask', 'family_id', 'seq_idx',
        ]
        for field in required_fields:
            assert field in arrays, f"Missing field: {field}"

    def test_shapes(self):
        df = make_test_df(100)
        fam = assign_family_ids(df['sequence'].tolist())
        arrays = build_transition_samples(df, fam, seq_length=36)
        n = len(df) * len(VALID_TRANSITIONS)

        assert arrays['one_hot'].shape == (n, 4, 36)
        assert arrays['structure_features'].shape == (n, 3)
        assert arrays['history_counts'].shape[0] == n
        assert arrays['history_mask'].shape[0] == n
        assert arrays['transition_type'].shape == (n,)
        assert arrays['round_from'].shape == (n,)
        assert arrays['round_to'].shape == (n,)
        assert arrays['delta_t'].shape == (n,)
        assert arrays['target_value'].shape == (n,)
        assert arrays['target_mask'].shape == (n,)
        assert arrays['family_id'].shape == (n,)

    def test_dtypes(self):
        df = make_test_df(50)
        fam = assign_family_ids(df['sequence'].tolist())
        arrays = build_transition_samples(df, fam)

        assert arrays['one_hot'].dtype == np.float32
        assert arrays['history_mask'].dtype == bool
        assert arrays['target_mask'].dtype == bool
        assert arrays['transition_type'].dtype == np.int64
        assert arrays['delta_t'].dtype == np.float32


class TestMaskCorrectness:
    """Test 2: Masks correctly represent valid/invalid data."""

    def test_history_mask_matches_rounds(self):
        df = make_test_df(50)
        fam = assign_family_ids(df['sequence'].tolist())
        arrays = build_transition_samples(df, fam)

        n_seqs = len(df)
        for ti, (r_from, r_to) in enumerate(VALID_TRANSITIONS):
            start = ti * n_seqs
            end = (ti + 1) * n_seqs
            mask = arrays['history_mask'][start:end]

            from data.preprocessing import ROUND_NAMES as RN
            from data.unified_dataset import ROUND_TO_INDEX
            n_history = sum(1 for r in RN
                           if ROUND_TO_INDEX[r] <= ROUND_TO_INDEX[r_from])
            n_history = min(n_history, 5)

            for i in range(n_seqs):
                assert mask[i, :n_history].all(), \
                    f"History mask should be True for first {n_history} entries"
                if n_history < 5:
                    assert not mask[i, n_history:].any(), \
                        f"History mask should be False after entry {n_history}"

    def test_all_targets_valid(self):
        df = make_test_df(50)
        fam = assign_family_ids(df['sequence'].tolist())
        arrays = build_transition_samples(df, fam)

        assert arrays['target_mask'].all(), \
            "All targets should be valid (R10→R11 excluded)"

    def test_history_zero_when_masked(self):
        df = make_test_df(50)
        fam = assign_family_ids(df['sequence'].tolist())
        arrays = build_transition_samples(df, fam)

        masked = arrays['history_counts'][~arrays['history_mask']]
        assert (masked == 0).all(), "Masked history entries should be zero"


class TestR10R11Exclusion:
    """Test 3: R10→R11 is excluded from the dataset."""

    def test_no_r10_r11_transition(self):
        df = make_test_df(50)
        fam = assign_family_ids(df['sequence'].tolist())
        arrays = build_transition_samples(df, fam)

        from data.unified_dataset import ROUND_TO_INDEX
        r10_idx = ROUND_TO_INDEX["R10"]
        r11_idx = ROUND_TO_INDEX["R11"]

        for i in range(len(arrays['round_from'])):
            rf = arrays['round_from'][i]
            rt = arrays['round_to'][i]
            assert not (rf == r10_idx and rt == r11_idx), \
                "R10→R11 should not appear in the dataset"

    def test_only_valid_transitions(self):
        df = make_test_df(50)
        fam = assign_family_ids(df['sequence'].tolist())
        arrays = build_transition_samples(df, fam)

        expected = set(VALID_TRANSITIONS)
        from data.unified_dataset import ROUND_TO_INDEX
        for i in range(len(arrays['round_from'])):
            rf_name = ROUND_NAMES[arrays['round_from'][i]]
            rt_name = ROUND_NAMES[arrays['round_to'][i]]
            assert (rf_name, rt_name) in expected


class TestTimeLeakage:
    """Test 4: History does not access future rounds."""

    def test_history_not_contains_target_round(self):
        df = make_test_df(50)
        fam = assign_family_ids(df['sequence'].tolist())
        arrays = build_transition_samples(df, fam)

        from data.unified_dataset import ROUND_TO_INDEX
        n_seqs = len(df)

        for ti, (r_from, r_to) in enumerate(VALID_TRANSITIONS):
            start = ti * n_seqs
            end = (ti + 1) * n_seqs
            mask = arrays['history_mask'][start:end]
            target_round_idx = ROUND_TO_INDEX[r_to]

            max_history_round = -1
            for hi in range(mask.shape[1]):
                if mask[0, hi]:
                    max_history_round = hi

            assert max_history_round < target_round_idx or \
                   max_history_round == ROUND_TO_INDEX[r_from], \
                f"History for {r_from}→{r_to} should not include {r_to}"

    def test_target_differs_from_last_history(self):
        df = make_test_df(100)
        fam = assign_family_ids(df['sequence'].tolist())
        arrays = build_transition_samples(df, fam)

        n_seqs = len(df)
        for ti, (r_from, r_to) in enumerate(VALID_TRANSITIONS):
            start = ti * n_seqs
            end = (ti + 1) * n_seqs

            target = arrays['target_value'][start:end]
            last_hist = arrays['history_counts'][start:end, -1]
            last_mask = arrays['history_mask'][start:end, -1]

            both_valid = last_mask & (np.abs(target) > 1e-6)
            if both_valid.any():
                diff = np.abs(target[both_valid] - last_hist[both_valid])
                assert diff.mean() > 1e-6, \
                    f"Target should differ from last history for {r_from}→{r_to}"


class TestFamilySplit:
    """Test 5: No family overlap between train/val/test."""

    def test_no_family_overlap(self):
        df = make_test_df(200)
        fam = assign_family_ids(df['sequence'].tolist())
        arrays = build_transition_samples(df, fam)

        splitter = FamilySplitter(train_frac=0.7, val_frac=0.15, seed=42)
        splits = splitter.split(arrays['family_id'],
                                arrays['transition_type'],
                                arrays['seq_idx'])

        train_fams = set(arrays['family_id'][splits['train']].tolist())
        val_fams = set(arrays['family_id'][splits['val']].tolist())
        test_fams = set(arrays['family_id'][splits['test']].tolist())

        assert len(train_fams & val_fams) == 0, "Train-val family overlap"
        assert len(train_fams & test_fams) == 0, "Train-test family overlap"
        assert len(val_fams & test_fams) == 0, "Val-test family overlap"

    def test_all_families_assigned(self):
        df = make_test_df(200)
        fam = assign_family_ids(df['sequence'].tolist())
        arrays = build_transition_samples(df, fam)

        splitter = FamilySplitter(train_frac=0.7, val_frac=0.15, seed=42)
        splits = splitter.split(arrays['family_id'],
                                arrays['transition_type'],
                                arrays['seq_idx'])

        all_assigned = (set(splits['train_families']) |
                       set(splits['val_families']) |
                       set(splits['test_families']))
        all_families = set(np.unique(arrays['family_id']).tolist())
        assert all_assigned == all_families, "Some families not assigned"

    def test_no_seq_overlap_across_splits(self):
        df = make_test_df(200)
        fam = assign_family_ids(df['sequence'].tolist())
        arrays = build_transition_samples(df, fam)

        splitter = FamilySplitter(train_frac=0.7, val_frac=0.15, seed=42)
        splits = splitter.split(arrays['family_id'],
                                arrays['transition_type'],
                                arrays['seq_idx'])

        train_seqs = set(arrays['seq_idx'][splits['train']].tolist())
        val_seqs = set(arrays['seq_idx'][splits['val']].tolist())
        test_seqs = set(arrays['seq_idx'][splits['test']].tolist())

        assert len(train_seqs & val_seqs) == 0
        assert len(train_seqs & test_seqs) == 0
        assert len(val_seqs & test_seqs) == 0


class TestTargetNormalization:
    """Test 6: Normalizer fit from training data only."""

    def test_normalizer_fit(self):
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        mask = np.array([True, True, True, True, True])

        norm = TargetNormalizer()
        norm.fit(values, mask)

        assert norm.fitted
        assert abs(norm.mean - 3.0) < 1e-6
        assert abs(norm.std - np.std(values)) < 1e-4

    def test_normalizer_partial_mask(self):
        values = np.array([1.0, 2.0, 999.0, 4.0, 5.0], dtype=np.float32)
        mask = np.array([True, True, False, True, True])

        norm = TargetNormalizer()
        norm.fit(values, mask)

        expected_mean = np.mean([1.0, 2.0, 4.0, 5.0])
        assert abs(norm.mean - expected_mean) < 1e-6

    def test_inverse_transform(self):
        values = np.random.randn(100).astype(np.float32) * 3 + 5
        mask = np.ones(100, dtype=bool)

        norm = TargetNormalizer()
        norm.fit(values, mask)

        transformed = norm.transform(values)
        recovered = norm.inverse_transform(transformed)

        np.testing.assert_allclose(recovered, values, atol=1e-5)

    def test_normalizer_not_fitted_raises(self):
        norm = TargetNormalizer()
        with pytest.raises(RuntimeError):
            norm.transform(np.array([1.0]))


class TestDatasetIntegration:
    """Integration test: full pipeline from DataFrame to DataLoader."""

    def test_full_pipeline(self):
        df = make_test_df(100)
        fam = assign_family_ids(df['sequence'].tolist())
        arrays = build_transition_samples(df, fam)

        splitter = FamilySplitter(train_frac=0.7, val_frac=0.15, seed=42)
        splits = splitter.split(arrays['family_id'],
                                arrays['transition_type'],
                                arrays['seq_idx'])

        norm = TargetNormalizer()
        norm.fit(arrays['target_value'][splits['train']],
                 arrays['target_mask'][splits['train']])

        target_norm = np.zeros_like(arrays['target_value'])
        valid = arrays['target_mask']
        target_norm[valid] = norm.transform(arrays['target_value'][valid])
        arrays['target_normalized'] = target_norm

        dataset_dict = {
            'arrays': arrays,
            'splits': splits,
            'normalizer': norm,
        }

        loaders = build_dataloaders(dataset_dict, batch_size=32)

        assert 'train' in loaders
        assert 'val' in loaders
        assert 'test' in loaders

        batch = next(iter(loaders['train']))
        assert batch['one_hot'].shape[1:] == (4, 36)
        assert batch['history_counts'].shape[1] == 5
        assert batch['target'].shape == (batch['one_hot'].shape[0],)
        assert batch['target_mask'].dtype == torch.bool

    def test_dataset_length(self):
        df = make_test_df(100)
        fam = assign_family_ids(df['sequence'].tolist())
        arrays = build_transition_samples(df, fam)

        splitter = FamilySplitter(train_frac=0.7, val_frac=0.15, seed=42)
        splits = splitter.split(arrays['family_id'],
                                arrays['transition_type'],
                                arrays['seq_idx'])

        ds = ETODataset(arrays, splits['train'])
        assert len(ds) == len(splits['train'])


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

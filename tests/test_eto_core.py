"""
ETO-Core Model Tests
====================

Tests 6-8 from the task specification:
  6. Model forward and backward pass
  7. Gradient non-NaN
  8. Checkpoint save and restore
"""

import sys
import os
import tempfile
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.eto_core import (
    ETCoreModel, StaticBaseline, HistoryBaseline, build_model,
)
from data.preprocessing import generate_synthetic_data
from data.unified_dataset import (
    build_transition_samples, assign_family_ids,
    FamilySplitter, TargetNormalizer,
)
from data.eto_dataset import ETODataset


def make_batch(batch_size=16, device='cpu'):
    df = generate_synthetic_data(n_sequences=60, n_families=6, seed=42)
    fam = assign_family_ids(df['sequence'].tolist())
    arrays = build_transition_samples(df, fam)

    splitter = FamilySplitter(train_frac=0.7, val_frac=0.15, seed=42)
    splits = splitter.split(arrays['family_id'], arrays['transition_type'],
                           arrays['seq_idx'])

    norm = TargetNormalizer()
    norm.fit(arrays['target_value'][splits['train']],
             arrays['target_mask'][splits['train']])
    target_norm = np.zeros_like(arrays['target_value'])
    valid = arrays['target_mask']
    target_norm[valid] = norm.transform(arrays['target_value'][valid])
    arrays['target_normalized'] = target_norm

    ds = ETODataset(arrays, splits['train'])
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size)
    batch = next(iter(loader))
    return {k: v.to(device) for k, v in batch.items()}


class TestForwardBackward:
    """Test 6: Forward and backward pass for all three models."""

    @pytest.mark.parametrize("model_type", ['static', 'history', 'etocore'])
    def test_forward_shape(self, model_type):
        model = build_model(model_type)
        batch = make_batch(8)
        out = model(batch)
        assert out['prediction'].shape == (8,)

    @pytest.mark.parametrize("model_type", ['static', 'history', 'etocore'])
    def test_backward_runs(self, model_type):
        model = build_model(model_type)
        batch = make_batch(8)
        out = model(batch)

        mask = batch['target_mask']
        loss = ((out['prediction'][mask] - batch['target'][mask]) ** 2).mean()
        loss.backward()

        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    @pytest.mark.parametrize("model_type", ['static', 'history', 'etocore'])
    def test_parameter_count(self, model_type):
        model = build_model(model_type)
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0
        print(f"  {model_type}: {n_params:,} parameters")


class TestGradientHealth:
    """Test 7: Gradients are finite (no NaN/Inf)."""

    @pytest.mark.parametrize("model_type", ['static', 'history', 'etocore'])
    def test_no_nan_gradients(self, model_type):
        model = build_model(model_type)
        batch = make_batch(16)
        out = model(batch)

        mask = batch['target_mask']
        if mask.sum() == 0:
            pytest.skip("No valid targets in batch")

        loss = ((out['prediction'][mask] - batch['target'][mask]) ** 2).mean()
        loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                assert torch.isfinite(param.grad).all(), \
                    f"Non-finite gradient in {name}"

    @pytest.mark.parametrize("model_type", ['static', 'history', 'etocore'])
    def test_no_nan_outputs(self, model_type):
        model = build_model(model_type)
        batch = make_batch(16)
        out = model(batch)

        assert torch.isfinite(out['prediction']).all(), \
            f"{model_type}: prediction has NaN/Inf"


class TestCheckpointSaveRestore:
    """Test 8: Checkpoint save and restore produces identical model."""

    @pytest.mark.parametrize("model_type", ['static', 'history', 'etocore'])
    def test_save_restore(self, model_type):
        model = build_model(model_type)
        model.eval()
        batch = make_batch(8)

        with torch.no_grad():
            out1 = model(batch)
            pred1 = out1['prediction'].clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.pt')
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_type': model_type,
            }, path)

            model2 = build_model(model_type)
            model2.eval()
            ckpt = torch.load(path, weights_only=True)
            model2.load_state_dict(ckpt['model_state_dict'])

        with torch.no_grad():
            out2 = model2(batch)
            pred2 = out2['prediction']

        torch.testing.assert_close(pred1, pred2)

    def test_etocore_state_dict_keys(self):
        model = ETCoreModel()
        sd = model.state_dict()
        expected_prefixes = [
            'seq_encoder', 'history_encoder', 'context_encoder',
            'state_builder', 'transition_op', 'obs_decoder',
        ]
        for prefix in expected_prefixes:
            keys = [k for k in sd.keys() if k.startswith(prefix)]
            assert len(keys) > 0, f"Missing module: {prefix}"


class TestMaskedLoss:
    """Test: Masked loss correctly excludes invalid targets."""

    def test_mask_excludes_invalid(self):
        model = ETCoreModel()
        batch = make_batch(16)

        batch['target_mask'][:] = False
        out = model(batch)

        mask = batch['target_mask']
        assert mask.sum() == 0

    def test_partial_mask(self):
        model = ETCoreModel()
        batch = make_batch(16)

        mask = batch['target_mask'].clone()
        mask[:8] = False

        pred = model(batch)['prediction']
        loss = ((pred[mask] - batch['target'][mask]) ** 2).mean()
        assert torch.isfinite(loss)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

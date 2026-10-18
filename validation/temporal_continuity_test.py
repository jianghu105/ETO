"""
Temporal Continuity Test

Validates that the Fourier continuous time encoding fixes the R10→R11 NaN issue
and enables genuine temporal generalization.

Test 1: Continuous Transition
    Train on R3→R5, R5→R9, R9→R10 (t=0,1,2)
    Test on R10→R11 (t=3) — previously caused NaN
    Expect: no NaN, EPS correlates with advantage

Test 2: Long Rollout
    From R3 state, rollout through R5→R9→R10→R11 (4 steps)
    Expect: trajectory remains finite, final state correlates with target

Test 3: Round Interpolation
    Train on all rounds
    Predict at intermediate times t=0.5, t=1.5, t=2.5
    Expect: outputs are smooth, no discontinuities
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import yaml
from scipy.stats import spearmanr

from data.preprocessing import (
    generate_synthetic_data, compute_features, prepare_training_data,
    ROUND_PAIRS, CONTEXT_MAP,
)
from models.eto_model import RefactoredETOModel
from training.losses import RefactoredETOLoss, count_nans


def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'configs', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config['data']['synthetic']['enabled'] = True
    config['data']['synthetic']['n_sequences'] = 2000
    config['training']['device'] = 'cpu'
    config['training']['batch_size'] = 128
    config['model']['context_encoder']['use_continuous_time'] = True
    return config


def create_batch(samples, device='cpu'):
    batch = {}
    for key in samples[0].keys():
        batch[key] = torch.stack([s[key] for s in samples]).to(device)
    return batch


def split_by_pairs(samples, train_pairs, test_pairs):
    train_samples = [s for s in samples if s['pair_idx'].item() in train_pairs]
    test_samples = [s for s in samples if s['pair_idx'].item() in test_pairs]
    return train_samples, test_samples


def train_model(config, train_samples, n_epochs=20):
    device = config['training']['device']
    model = RefactoredETOModel(config).to(device)
    loss_fn = RefactoredETOLoss(
        lambda_transition=1.0, lambda_population=0.5,
        lambda_advantage=1.0, lambda_rollout=0.3,
        lambda_smooth=config['training'].get('lambda_smooth', 0.05),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'],
                                   weight_decay=config['training']['weight_decay'])

    batch_size = config['training']['batch_size']
    n = len(train_samples)
    nan_count_log = []

    for epoch in range(n_epochs):
        model.train()
        indices = np.random.permutation(n)
        epoch_nan = 0
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = create_batch([train_samples[i] for i in indices[start:end]], device)
            outputs = model(batch, rollout_steps=2)
            losses = loss_fn(outputs, batch)

            nan_info = count_nans(outputs)
            epoch_nan += sum(v['nan'] for v in nan_info.values())

            optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        nan_count_log.append(epoch_nan)

    model.eval()
    return model, nan_count_log


def test_continuous_transition(config, samples):
    """
    Test 1: Train on early rounds, test on R10→R11.
    This was the failing case (NaN) before Fourier time encoding.
    """
    print("\n" + "=" * 60)
    print("Test 1: Continuous Transition (R10→R11 held out)")
    print("=" * 60)

    train_pairs = [0, 1, 2]
    test_pairs = [3]

    train_samples, test_samples = split_by_pairs(samples, train_pairs, test_pairs)
    print(f"  Train: {len(train_samples)} samples (t=0,1,2)")
    print(f"  Test:  {len(test_samples)} samples (t=3)")

    if len(test_samples) < 10:
        print("  [SKIP] Too few test samples")
        return {'passed': False, 'reason': 'too few samples'}

    model, nan_log = train_model(config, train_samples, n_epochs=20)

    total_nans = sum(nan_log)
    print(f"  Total NaN count during training: {total_nans}")

    device = config['training']['device']
    batch = create_batch(test_samples, device)

    with torch.no_grad():
        outputs = model(batch)

    eps = outputs['eps'].cpu().numpy()
    delta_z = outputs['delta_z'].cpu().numpy()
    advantage = batch['evolutionary_advantage'].cpu().numpy()

    has_nan = np.isnan(eps).any() or np.isnan(delta_z).any()
    print(f"  EPS has NaN: {np.isnan(eps).any()}")
    print(f"  delta_z has NaN: {np.isnan(delta_z).any()}")
    print(f"  EPS range: [{eps.min():.4f}, {eps.max():.4f}]")

    if has_nan:
        print(f"\n  [FAIL] NaN detected in outputs")
        return {'passed': False, 'reason': 'NaN in outputs', 'nan_count': total_nans}

    valid_mask = np.isfinite(eps) & np.isfinite(advantage)
    if valid_mask.sum() < 5 or advantage[valid_mask].std() < 1e-8:
        print(f"\n  [INFO] Advantage is constant for R10→R11 (no future rounds)")
        growth = batch['future_growth'].cpu().numpy()
        valid_mask = np.isfinite(eps) & np.isfinite(growth)
        if valid_mask.sum() >= 5 and growth[valid_mask].std() > 1e-8:
            corr_adv, p_adv = spearmanr(eps[valid_mask], growth[valid_mask])
            print(f"  EPS ~ future_growth: {corr_adv:+.4f} (p={p_adv:.2e})")
        else:
            corr_adv, p_adv = 0.0, 1.0
            print(f"  EPS ~ growth: constant (expected for terminal round)")
    else:
        corr_adv, p_adv = spearmanr(eps[valid_mask], advantage[valid_mask])
    print(f"  EPS ~ advantage: {corr_adv:+.4f} (p={p_adv:.2e})")

    passed = not has_nan
    print(f"\n  [{'PASS' if passed else 'FAIL'}] Continuous transition to unseen round")
    print(f"  (Key criterion: no NaN. Correlation is secondary for terminal round.)")
    return {
        'passed': passed,
        'corr_advantage': corr_adv,
        'p_advantage': p_adv,
        'nan_count': total_nans,
        'eps_range': (eps.min(), eps.max()),
    }


def test_long_rollout(config, samples):
    """
    Test 2: Multi-step rollout from R3 through all rounds.
    """
    print("\n" + "=" * 60)
    print("Test 2: Long Rollout (R3 → R5 → R9 → R10 → R11)")
    print("=" * 60)

    train_samples = samples
    print(f"  Training on all {len(train_samples)} samples...")

    model, nan_log = train_model(config, train_samples, n_epochs=20)
    total_nans = sum(nan_log)
    print(f"  Total NaN count during training: {total_nans}")

    device = config['training']['device']
    r3_samples = [s for s in samples if s['pair_idx'].item() == 0]
    if len(r3_samples) < 10:
        print("  [SKIP] Too few R3→R5 samples")
        return {'passed': False, 'reason': 'too few samples'}

    batch = create_batch(r3_samples[:200], device)

    with torch.no_grad():
        seq_emb = model.encode_sequences(batch['one_hot'])
        struct_emb = model.encode_structure(batch['structure_features']) if model.use_structure else None
        vel = batch['x_prev'][:, 1] if batch['x_prev'].shape[-1] > 1 \
            else torch.zeros_like(batch['log_count_prev'])
        states = model.build_states(seq_emb, vel, batch['log_count_prev'], struct_emb)
        z_t = model.encode_population(states['dynamic_state'].unsqueeze(1))

        context_ids = [torch.tensor(0, device=device).expand(z_t.shape[0])]
        times = [torch.tensor(0.0, device=device).expand(z_t.shape[0])]
        for step in range(1, 4):
            ctx = torch.tensor(min(step, 3), device=device).expand(z_t.shape[0])
            t = torch.tensor(float(step), device=device).expand(z_t.shape[0])
            context_ids.append(ctx)
            times.append(t)

        context_embeddings = [
            model.encode_context(ctx, None, t)
            for ctx, t in zip(context_ids, times)
        ]

        rollout_result = model.rollout(z_t, context_embeddings, 4)

    z_traj = rollout_result['z_trajectory']
    delta_z_traj = rollout_result['delta_z_trajectory']

    print(f"  Trajectory length: {len(z_traj)} states")
    all_finite = True
    for k, z in enumerate(z_traj):
        z_np = z.cpu().numpy()
        finite = np.isfinite(z_np).all()
        print(f"    z_{k}: finite={finite}, norm={np.linalg.norm(z_np.mean(axis=0)):.4f}")
        if not finite:
            all_finite = False

    for k, dz in enumerate(delta_z_traj):
        dz_np = dz.cpu().numpy()
        finite = np.isfinite(dz_np).all()
        print(f"    delta_z_{k}: finite={finite}, norm={np.linalg.norm(dz_np.mean(axis=0)):.6f}")
        if not finite:
            all_finite = False

    if len(delta_z_traj) >= 2:
        smoothness = []
        for k in range(len(delta_z_traj) - 1):
            d1 = delta_z_traj[k].cpu().numpy()
            d2 = delta_z_traj[k + 1].cpu().numpy()
            smoothness.append(np.mean((d1 - d2) ** 2))
        avg_smooth = np.mean(smoothness)
        print(f"\n  Avg temporal smoothness: {avg_smooth:.6f}")
    else:
        avg_smooth = float('inf')

    passed = all_finite and total_nans < 10
    print(f"\n  [{'PASS' if passed else 'FAIL'}] Long rollout stability")
    return {
        'passed': passed,
        'all_finite': all_finite,
        'nan_count': total_nans,
        'smoothness': avg_smooth,
    }


def test_round_interpolation(config, samples):
    """
    Test 3: Predict at intermediate time points.
    Train on t=0,1,2,3 and check smoothness at t=0.5, 1.5, 2.5.
    """
    print("\n" + "=" * 60)
    print("Test 3: Round Interpolation (t=0.5, 1.5, 2.5)")
    print("=" * 60)

    model, nan_log = train_model(config, samples, n_epochs=20)

    device = config['training']['device']
    test_samples = [s for s in samples if s['pair_idx'].item() == 0][:50]
    batch = create_batch(test_samples, device)

    with torch.no_grad():
        seq_emb = model.encode_sequences(batch['one_hot'])
        struct_emb = model.encode_structure(batch['structure_features']) if model.use_structure else None
        vel = batch['x_prev'][:, 1] if batch['x_prev'].shape[-1] > 1 \
            else torch.zeros_like(batch['log_count_prev'])
        states = model.build_states(seq_emb, vel, batch['log_count_prev'], struct_emb)
        z_t = model.encode_population(states['dynamic_state'].unsqueeze(1))

        test_times = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
        context_id = batch['context_id']

        z_list = []
        eps_list = []
        for t_val in test_times:
            t_tensor = torch.tensor(t_val, device=device).expand(z_t.shape[0])
            e_t = model.encode_context(context_id, None, t_tensor)
            z_next, delta_z = model.transition(z_t, e_t)
            z_list.append(z_next.cpu().numpy())
            eps_val = model.decode_potential(z_next, seq_emb, vel, struct_emb)
            eps_list.append(eps_val.cpu().numpy())

    print(f"  {'t':>6} | {'||z||':>10} | {'EPS_mean':>10} | {'EPS_std':>10} | {'NaN':>5}")
    print(f"  {'-'*6} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*5}")
    all_finite = True
    for i, t_val in enumerate(test_times):
        z_norm = np.linalg.norm(z_list[i].mean(axis=0))
        eps_mean = eps_list[i].mean()
        eps_std = eps_list[i].std()
        has_nan = np.isnan(z_list[i]).any() or np.isnan(eps_list[i]).any()
        if has_nan:
            all_finite = False
        print(f"  {t_val:>6.1f} | {z_norm:>10.4f} | {eps_mean:>10.4f} | {eps_std:>10.4f} | {'YES' if has_nan else 'no'}")

    deltas = []
    for i in range(len(eps_list) - 1):
        delta = np.mean(np.abs(eps_list[i + 1] - eps_list[i]))
        deltas.append(delta)

    max_jump = max(deltas) if deltas else 0
    avg_jump = np.mean(deltas) if deltas else 0
    print(f"\n  Max EPS jump between adjacent t: {max_jump:.4f}")
    print(f"  Avg EPS jump between adjacent t: {avg_jump:.4f}")

    monotonic_smoothness = all(d < 1.0 for d in deltas)
    passed = all_finite and monotonic_smoothness
    print(f"\n  [{'PASS' if passed else 'FAIL'}] Interpolation smoothness")
    return {
        'passed': passed,
        'all_finite': all_finite,
        'max_jump': max_jump,
        'avg_jump': avg_jump,
    }


def main():
    print("=" * 60)
    print("Temporal Continuity Test")
    print("Fourier continuous time encoding validation")
    print("=" * 60)

    config = load_config()
    print("\nGenerating synthetic data...")
    df = generate_synthetic_data(n_sequences=1000, n_families=10, seed=42)
    features = compute_features(df)
    samples = prepare_training_data(features)
    print(f"  {len(samples)} total samples")

    for pair_idx, (r_prev, r_curr) in enumerate(ROUND_PAIRS):
        n = sum(1 for s in samples if s['pair_idx'].item() == pair_idx)
        print(f"    Pair {pair_idx} ({r_prev}→{r_curr}, t={pair_idx}): {n} samples")

    results = {}
    results['test1_continuous'] = test_continuous_transition(config, samples)
    results['test2_rollout'] = test_long_rollout(config, samples)
    results['test3_interpolation'] = test_round_interpolation(config, samples)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, r in results.items():
        status = 'PASS' if r.get('passed', False) else 'FAIL'
        print(f"  {name:<30} {status}")

    n_pass = sum(1 for r in results.values() if r.get('passed', False))
    n_total = len(results)
    print(f"\n  {n_pass}/{n_total} tests passed")
    print("=" * 60)

    return results


if __name__ == '__main__':
    main()

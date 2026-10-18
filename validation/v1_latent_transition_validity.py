"""
Validation 1: Latent Transition Validity

Verifies that the transition operator truly learns evolutionary dynamics,
not just memorizing input-output mappings.

Tests:
  1. Cosine similarity: Z(t+1)_pred vs Encoder(R_t+1)
  2. Latent consistency: same sequence → same Z across different batches
  3. Trajectory smoothness: ||Z_{k+1} - Z_k|| should be bounded
  4. Rollout error accumulation: multi-step prediction error growth
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import yaml
from scipy.stats import spearmanr

from data.preprocessing import generate_synthetic_data, compute_features, prepare_training_data
from models.eto_model import RefactoredETOModel
from training.losses import RefactoredETOLoss


def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'configs', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config['data']['synthetic']['enabled'] = True
    config['data']['synthetic']['n_sequences'] = 2000
    config['training']['device'] = 'cpu'
    config['training']['batch_size'] = 128
    return config


def create_batch(samples, device='cpu'):
    batch = {}
    for key in samples[0].keys():
        batch[key] = torch.stack([s[key] for s in samples]).to(device)
    return batch


def train_baseline(config, samples, n_epochs=30):
    device = config['training']['device']
    model = RefactoredETOModel(config).to(device)
    loss_fn = RefactoredETOLoss(
        lambda_transition=1.0, lambda_population=0.5,
        lambda_advantage=1.0, lambda_rollout=0.3,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'],
                                   weight_decay=config['training']['weight_decay'])

    batch_size = config['training']['batch_size']
    n = len(samples)

    for epoch in range(n_epochs):
        model.train()
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = create_batch([samples[i] for i in indices[start:end]], device)
            outputs = model(batch)
            losses = loss_fn(outputs, batch)
            optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    return model


def test_cosine_similarity(model, samples, config):
    """
    Test 1: Cosine similarity between predicted Z(t+1) and encoded Z(t+1).
    High cosine = transition operator is learning meaningful dynamics.
    """
    print("\n--- Test 1: Cosine Latent Prediction ---")
    device = config['training']['device']
    n_eval = min(500, len(samples))
    batch = create_batch(samples[:n_eval], device)

    with torch.no_grad():
        outputs = model(batch, return_intermediates=True)

    z_pred = outputs['z_next_pred']
    z_target = outputs['z_t_target']

    cos_sim = torch.nn.functional.cosine_similarity(z_pred, z_target, dim=-1)
    mean_cos = cos_sim.mean().item()
    std_cos = cos_sim.std().item()

    print(f"  Cosine similarity (Z_pred vs Z_target):")
    print(f"    Mean = {mean_cos:.4f} +/- {std_cos:.4f}")
    print(f"    Median = {cos_sim.median().item():.4f}")
    print(f"    Min = {cos_sim.min().item():.4f}, Max = {cos_sim.max().item():.4f}")

    passed = mean_cos > 0.5
    print(f"  Criterion: mean cosine > 0.5")
    print(f"  [{'PASS' if passed else 'FAIL'}] Cosine similarity = {mean_cos:.4f}")
    return {'mean_cosine': mean_cos, 'std_cosine': std_cos, 'passed': passed}


def test_latent_consistency(model, samples, config, n_trials=5):
    """
    Test 2: Same input → same Z across different forward passes.
    Tests determinism and stability of the encoder.
    """
    print("\n--- Test 2: Latent Consistency ---")
    device = config['training']['device']
    n_eval = min(200, len(samples))
    batch = create_batch(samples[:n_eval], device)

    z_list = []
    with torch.no_grad():
        for trial in range(n_trials):
            outputs = model(batch)
            z_list.append(outputs['z_t'].cpu().numpy())

    z_stack = np.stack(z_list, axis=0)
    z_mean = z_stack.mean(axis=0)
    z_std = z_stack.std(axis=0)

    cv = z_std / (np.abs(z_mean) + 1e-8)
    mean_cv = cv.mean()

    pairwise_cos = []
    for i in range(n_trials):
        for j in range(i+1, n_trials):
            cos = np.mean([
                np.dot(z_list[i][k], z_list[j][k]) /
                (np.linalg.norm(z_list[i][k]) * np.linalg.norm(z_list[j][k]) + 1e-8)
                for k in range(n_eval)
            ])
            pairwise_cos.append(cos)

    mean_pairwise = np.mean(pairwise_cos)

    print(f"  {n_trials} forward passes on same input:")
    print(f"    Mean coefficient of variation: {mean_cv:.6f}")
    print(f"    Mean pairwise cosine: {mean_pairwise:.6f}")

    passed = mean_pairwise > 0.99
    print(f"  Criterion: pairwise cosine > 0.99")
    print(f"  [{'PASS' if passed else 'FAIL'}] Consistency = {mean_pairwise:.6f}")
    return {'mean_cv': mean_cv, 'pairwise_cos': mean_pairwise, 'passed': passed}


def test_trajectory_smoothness(model, samples, config):
    """
    Test 3: Rollout trajectory should be smooth.
    ||Z_{k+1} - Z_k|| should not explode.
    """
    print("\n--- Test 3: Trajectory Smoothness ---")
    device = config['training']['device']
    n_eval = min(100, len(samples))
    batch = create_batch(samples[:n_eval], device)

    with torch.no_grad():
        z_start = model(batch)['z_t']

        context_ids = batch['context_id']
        round_idxs = batch.get('round_idx', context_ids.clone())

        context_sequence = []
        for step in range(4):
            ctx = (context_ids + step).clamp(max=3)
            rnd = (round_idxs + step).clamp(max=9) if round_idxs is not None else None
            e_step = model.encode_context(ctx, rnd)
            context_sequence.append(e_step)

        result = model.rollout(z_start, context_sequence, 4)

    z_traj = result['z_trajectory']
    step_distances = []
    for k in range(len(z_traj) - 1):
        d = (z_traj[k+1] - z_traj[k]).norm(dim=-1)
        step_distances.append(d.mean().item())

    norms = [z.norm(dim=-1).mean().item() for z in z_traj]

    print(f"  4-step rollout trajectory:")
    print(f"    Step distances: {[f'{d:.4f}' for d in step_distances]}")
    print(f"    Z norms: {[f'{n:.4f}' for n in norms]}")

    max_dist = max(step_distances)
    norm_growth = norms[-1] / (norms[0] + 1e-8)

    print(f"    Max step distance: {max_dist:.4f}")
    print(f"    Norm growth ratio: {norm_growth:.4f}")

    dist_stable = max_dist < 5.0
    norm_stable = norm_growth < 2.0
    passed = dist_stable and norm_stable

    print(f"  Criterion: max_dist < 5.0 AND norm_growth < 2.0")
    print(f"  [{'PASS' if passed else 'FAIL'}] Smoothness")
    return {
        'step_distances': step_distances,
        'norms': norms,
        'max_dist': max_dist,
        'norm_growth': norm_growth,
        'passed': passed,
    }


def test_rollout_error(model, samples, config):
    """
    Test 4: Multi-step prediction error should grow sub-linearly.
    """
    print("\n--- Test 4: Rollout Error Accumulation ---")
    device = config['training']['device']
    n_eval = min(200, len(samples))

    pair_groups = {}
    for i, s in enumerate(samples[:n_eval]):
        pidx = s['pair_idx'].item()
        if pidx not in pair_groups:
            pair_groups[pidx] = []
        pair_groups[pidx].append(i)

    errors_per_step = {1: [], 2: [], 3: [], 4: []}

    with torch.no_grad():
        for pidx, indices in pair_groups.items():
            batch = create_batch([samples[i] for i in indices], device)
            outputs = model(batch, return_intermediates=True)
            z_start = outputs['z_t']
            z_true_next = outputs['z_t_target']

            context_ids = batch['context_id']
            round_idxs = batch.get('round_idx', context_ids.clone())

            for n_steps in [1, 2, 3, 4]:
                context_sequence = []
                for step in range(n_steps):
                    ctx = (context_ids + step).clamp(max=3)
                    rnd = (round_idxs + step).clamp(max=9) if round_idxs is not None else None
                    e_step = model.encode_context(ctx, rnd)
                    context_sequence.append(e_step)

                result = model.rollout(z_start, context_sequence, n_steps)
                z_final_pred = result['z_trajectory'][-1]

                cos = torch.nn.functional.cosine_similarity(z_final_pred, z_true_next, dim=-1)
                errors_per_step[n_steps].append(cos.mean().item())

    print(f"  Cosine similarity at each rollout depth:")
    print(f"  {'Steps':<10} {'Mean Cos':<12} {'Std':<12}")
    print(f"  {'-'*34}")
    for n_steps in [1, 2, 3, 4]:
        vals = errors_per_step[n_steps]
        if vals:
            print(f"  {n_steps:<10} {np.mean(vals):+.4f}{'':<5} {np.std(vals):.4f}")

    cos_1 = np.mean(errors_per_step[1]) if errors_per_step[1] else 0
    cos_4 = np.mean(errors_per_step[4]) if errors_per_step[4] else 0

    degradation = cos_1 - cos_4
    print(f"\n  Degradation (1-step → 4-step): {degradation:.4f}")

    passed = cos_1 > 0.3 and degradation < 0.5
    print(f"  Criterion: cos_1 > 0.3 AND degradation < 0.5")
    print(f"  [{'PASS' if passed else 'FAIL'}] Rollout error")
    return {'cos_1': cos_1, 'cos_4': cos_4, 'degradation': degradation, 'passed': passed}


def main():
    print("=" * 60)
    print("Validation 1: Latent Transition Validity")
    print("=" * 60)

    config = load_config()
    print("Generating data...")
    df = generate_synthetic_data(n_sequences=2000, n_families=20, seed=42)
    features = compute_features(df)
    samples = prepare_training_data(features)
    print(f"  {len(samples)} samples")

    print("Training model (30 epochs)...")
    model = train_baseline(config, samples, n_epochs=30)

    results = {}
    results['cosine'] = test_cosine_similarity(model, samples, config)
    results['consistency'] = test_latent_consistency(model, samples, config)
    results['smoothness'] = test_trajectory_smoothness(model, samples, config)
    results['rollout_error'] = test_rollout_error(model, samples, config)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    n_pass = sum(1 for r in results.values() if r['passed'])
    for name, r in results.items():
        print(f"  {name:<20} {'PASS' if r['passed'] else 'FAIL'}")
    print(f"\n  {n_pass}/4 passed")
    print("=" * 60)

    return results


if __name__ == '__main__':
    main()

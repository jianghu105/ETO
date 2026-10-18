"""
Validation 4: Real SELEX Noise Simulation

Simulates real experimental noise to test robustness:
  1. PCR bias: count *= exp(beta), beta ~ N(0, sigma_pcr)
  2. Sequencing dropout: low-abundance random removal
  3. Sampling depth variation: 300K, 100K, 50K, 10K

Tests:
  - EPS stability under noise
  - Ranking robustness (top-k overlap)
  - Trajectory prediction accuracy
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


def train_model(config, samples, n_epochs=30):
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


def apply_pcr_bias(batch, sigma_pcr=0.3, device='cpu'):
    B = batch['count_curr'].shape[0]
    beta = torch.randn(B, device=device) * sigma_pcr
    pcr_factor = torch.exp(beta)

    noisy_batch = {k: v.clone() for k, v in batch.items()}
    noisy_batch['count_prev'] = batch['count_prev'] * pcr_factor
    noisy_batch['count_curr'] = batch['count_curr'] * pcr_factor
    noisy_batch['log_count_prev'] = torch.log1p(noisy_batch['count_prev'])
    noisy_batch['log_count_curr'] = torch.log1p(noisy_batch['count_curr'])
    noisy_batch['x_prev'] = batch['x_prev'].clone()
    noisy_batch['x_curr'] = batch['x_curr'].clone()
    noisy_batch['x_prev'][:, 0] = noisy_batch['log_count_prev']
    noisy_batch['x_curr'][:, 0] = noisy_batch['log_count_curr']
    return noisy_batch


def apply_sequencing_dropout(batch, dropout_rate=0.1, device='cpu'):
    B = batch['count_curr'].shape[0]
    mask = torch.rand(B, device=device) > dropout_rate

    noisy_batch = {k: v.clone() for k, v in batch.items()}
    noisy_batch['count_prev'] = batch['count_prev'] * mask.float()
    noisy_batch['count_curr'] = batch['count_curr'] * mask.float()
    noisy_batch['log_count_prev'] = torch.log1p(noisy_batch['count_prev'])
    noisy_batch['log_count_curr'] = torch.log1p(noisy_batch['count_curr'])
    noisy_batch['x_prev'] = batch['x_prev'].clone()
    noisy_batch['x_curr'] = batch['x_curr'].clone()
    noisy_batch['x_prev'][:, 0] = noisy_batch['log_count_prev']
    noisy_batch['x_curr'][:, 0] = noisy_batch['log_count_curr']
    return noisy_batch


def apply_depth_subsampling(samples, target_depth, original_depth=300000, seed=42):
    rng = np.random.RandomState(seed)
    subsample_ratio = target_depth / original_depth

    subsampled = []
    for s in samples:
        if rng.random() < subsample_ratio:
            subsampled.append(s)

    if len(subsampled) < 50:
        indices = rng.choice(len(samples), 50, replace=False)
        subsampled = [samples[i] for i in indices]

    return subsampled


def test_pcr_bias(model, samples, config, sigmas=[0.0, 0.1, 0.3, 0.5, 1.0]):
    print("\n--- Test 1: PCR Bias Robustness ---")
    device = config['training']['device']
    n_eval = min(500, len(samples))
    batch_clean = create_batch(samples[:n_eval], device)

    with torch.no_grad():
        out_clean = model(batch_clean)
    eps_clean = out_clean['eps'].cpu().numpy()
    advantage = batch_clean['evolutionary_advantage'].cpu().numpy()

    print(f"  {'PCR sigma':<12} {'EPS stability':<15} {'Adv correlation':<18} {'Rank overlap@20'}")
    print(f"  {'-'*60}")

    results = []
    for sigma in sigmas:
        if sigma == 0.0:
            eps_noisy = eps_clean
        else:
            batch_noisy = apply_pcr_bias(batch_clean, sigma_pcr=sigma, device=device)
            with torch.no_grad():
                out_noisy = model(batch_noisy)
            eps_noisy = out_noisy['eps'].cpu().numpy()

        stability, _ = spearmanr(eps_clean, eps_noisy)
        corr_adv, _ = spearmanr(eps_noisy, advantage)

        top_k = min(20, n_eval // 10)
        top_clean = set(np.argsort(eps_clean)[-top_k:])
        top_noisy = set(np.argsort(eps_noisy)[-top_k:])
        overlap = len(top_clean & top_noisy) / top_k

        print(f"  {sigma:<12.1f} {stability:+.4f}{'':<6} {corr_adv:+.4f}{'':<10} {overlap:.4f}")
        results.append({'sigma': sigma, 'stability': stability, 'corr_adv': corr_adv, 'overlap': overlap})

    high_noise = [r for r in results if r['sigma'] >= 0.3]
    mean_stability = np.mean([r['stability'] for r in high_noise])
    passed = mean_stability > 0.7
    print(f"\n  Mean stability at high noise (sigma>=0.3): {mean_stability:.4f}")
    print(f"  Criterion: stability > 0.7")
    print(f"  [{'PASS' if passed else 'FAIL'}] PCR bias robustness")
    return {'results': results, 'mean_stability': mean_stability, 'passed': passed}


def test_sequencing_dropout(model, samples, config, rates=[0.0, 0.05, 0.1, 0.2, 0.5]):
    print("\n--- Test 2: Sequencing Dropout Robustness ---")
    device = config['training']['device']
    n_eval = min(500, len(samples))
    batch_clean = create_batch(samples[:n_eval], device)

    with torch.no_grad():
        out_clean = model(batch_clean)
    eps_clean = out_clean['eps'].cpu().numpy()
    advantage = batch_clean['evolutionary_advantage'].cpu().numpy()

    print(f"  {'Dropout rate':<15} {'EPS stability':<15} {'Adv correlation':<18} {'Rank overlap@20'}")
    print(f"  {'-'*60}")

    results = []
    for rate in rates:
        if rate == 0.0:
            eps_noisy = eps_clean
        else:
            batch_noisy = apply_sequencing_dropout(batch_clean, dropout_rate=rate, device=device)
            with torch.no_grad():
                out_noisy = model(batch_noisy)
            eps_noisy = out_noisy['eps'].cpu().numpy()

        stability, _ = spearmanr(eps_clean, eps_noisy)
        corr_adv, _ = spearmanr(eps_noisy, advantage)

        top_k = min(20, n_eval // 10)
        top_clean = set(np.argsort(eps_clean)[-top_k:])
        top_noisy = set(np.argsort(eps_noisy)[-top_k:])
        overlap = len(top_clean & top_noisy) / top_k

        print(f"  {rate:<15.2f} {stability:+.4f}{'':<6} {corr_adv:+.4f}{'':<10} {overlap:.4f}")
        results.append({'rate': rate, 'stability': stability, 'corr_adv': corr_adv, 'overlap': overlap})

    high_noise = [r for r in results if r['rate'] >= 0.1]
    mean_stability = np.mean([r['stability'] for r in high_noise])
    passed = mean_stability > 0.7
    print(f"\n  Mean stability at high dropout (rate>=0.1): {mean_stability:.4f}")
    print(f"  Criterion: stability > 0.7")
    print(f"  [{'PASS' if passed else 'FAIL'}] Sequencing dropout robustness")
    return {'results': results, 'mean_stability': mean_stability, 'passed': passed}


def test_sampling_depth(model, samples, config, depths=[300000, 100000, 50000, 10000]):
    print("\n--- Test 3: Sampling Depth Variation ---")
    device = config['training']['device']

    batch_full = create_batch(samples[:500], device)
    with torch.no_grad():
        out_full = model(batch_full)
    eps_full = out_full['eps'].cpu().numpy()
    advantage_full = batch_full['evolutionary_advantage'].cpu().numpy()

    top_k = min(20, len(eps_full) // 10)
    top_full = set(np.argsort(eps_full)[-top_k:])

    print(f"  {'Depth':<12} {'N samples':<12} {'Adv corr':<12} {'Top-k recall':<15} {'Rank stability'}")
    print(f"  {'-'*65}")

    results = []
    for depth in depths:
        sub_samples = apply_depth_subsampling(samples, depth, original_depth=300000)
        n_sub = min(500, len(sub_samples))
        batch_sub = create_batch(sub_samples[:n_sub], device)

        with torch.no_grad():
            out_sub = model(batch_sub)
        eps_sub = out_sub['eps'].cpu().numpy()
        advantage_sub = batch_sub['evolutionary_advantage'].cpu().numpy()

        corr_adv, _ = spearmanr(eps_sub, advantage_sub)

        top_sub = set(np.argsort(eps_sub)[-min(top_k, n_sub // 10):])
        recall = len(top_full & top_sub) / max(len(top_full), 1)

        print(f"  {depth:<12,} {n_sub:<12} {corr_adv:+.4f}{'':<5} {recall:.4f}{'':<8}")
        results.append({'depth': depth, 'n_samples': n_sub, 'corr_adv': corr_adv, 'recall': recall})

    low_depth = [r for r in results if r['depth'] <= 50000]
    mean_corr = np.mean([r['corr_adv'] for r in low_depth])
    passed = mean_corr > 0.05
    print(f"\n  Mean advantage corr at low depth (<=50K): {mean_corr:+.4f}")
    print(f"  Criterion: corr > 0.05")
    print(f"  [{'PASS' if passed else 'FAIL'}] Sampling depth robustness")
    return {'results': results, 'mean_corr': mean_corr, 'passed': passed}


def main():
    print("=" * 60)
    print("Validation 4: Real SELEX Noise Simulation")
    print("=" * 60)

    config = load_config()
    print("Generating data...")
    df = generate_synthetic_data(n_sequences=2000, n_families=20, seed=42)
    features = compute_features(df)
    samples = prepare_training_data(features)
    print(f"  {len(samples)} samples")

    print("Training model (30 epochs)...")
    model = train_model(config, samples, n_epochs=30)

    results = {}
    results['pcr'] = test_pcr_bias(model, samples, config)
    results['dropout'] = test_sequencing_dropout(model, samples, config)
    results['depth'] = test_sampling_depth(model, samples, config)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    n_pass = sum(1 for r in results.values() if r['passed'])
    for name, r in results.items():
        print(f"  {name:<15} {'PASS' if r['passed'] else 'FAIL'}")
    print(f"\n  {n_pass}/3 tests passed")

    if n_pass == 3:
        print("\n  [READY] ETO is robust to real SELEX noise")
    elif n_pass >= 2:
        print("\n  [CAUTION] Some noise sensitivities detected")
    else:
        print("\n  [RISK] Significant noise sensitivity")
    print("=" * 60)

    return results


if __name__ == '__main__':
    main()

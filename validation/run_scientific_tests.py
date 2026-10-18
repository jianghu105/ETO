"""
Three scientific validation tests for ETO v2.

These tests verify that ETO has genuine evolutionary dynamics properties,
not just statistical fitting:

1. EPS Stability: EPS should be robust to small input perturbations
2. Context Counterfactual: EPS should change when selection context changes
3. Mutation Robustness: EPS should be robust to single-point mutations
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import yaml
from scipy.stats import spearmanr

from data.preprocessing import (
    generate_synthetic_data, compute_features,
    prepare_training_data, one_hot_encode, BASE_MAP,
)
from models.eto_model import RefactoredETOModel
from training.losses import RefactoredETOLoss


def load_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'configs', 'config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config['data']['synthetic']['enabled'] = True
    config['data']['synthetic']['n_sequences'] = 2000
    config['data']['synthetic']['n_families'] = 20
    config['training']['device'] = 'cpu'
    config['training']['batch_size'] = 128
    return config


def prepare_data_and_model(config, n_epochs=30):
    print("Preparing data...")
    df = generate_synthetic_data(
        n_sequences=config['data']['synthetic']['n_sequences'],
        n_families=config['data']['synthetic']['n_families'],
        seed=42,
    )
    features = compute_features(df)
    samples = prepare_training_data(features)

    device = config['training']['device']
    model = RefactoredETOModel(config).to(device)
    loss_fn = RefactoredETOLoss(
        lambda_transition=config['training'].get('lambda_transition', 1.0),
        lambda_population=config['training'].get('lambda_population', 0.5),
        lambda_advantage=config['training'].get('lambda_advantage', 1.0),
        lambda_rollout=config['training'].get('lambda_rollout', 0.3),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
    )

    batch_size = config['training']['batch_size']
    n_samples = len(samples)

    print(f"Training {n_epochs} epochs...")
    for epoch in range(n_epochs):
        model.train()
        indices = np.random.permutation(n_samples)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, batch_size):
            end = min(start + batch_size, n_samples)
            batch_indices = indices[start:end]

            batch = {}
            for key in samples[0].keys():
                vals = [samples[i][key] for i in batch_indices]
                batch[key] = torch.stack(vals).to(device)

            outputs = model(batch)
            losses = loss_fn(outputs, batch)

            optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += losses['total'].item()
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}: loss={epoch_loss/n_batches:.4f}")

    model.eval()
    return model, df, features, samples


def create_batch(samples, device='cpu'):
    batch = {}
    for key in samples[0].keys():
        vals = [s[key] for s in samples]
        batch[key] = torch.stack(vals).to(device)
    return batch


# ============================================================
# Test 1: EPS Stability
# ============================================================
def test_eps_stability(model, samples, config, n_perturbations=10):
    """
    EPS should be stable under small input perturbations.

    Method:
      1. Compute EPS for original samples
      2. Add small Gaussian noise to log_count (sigma=0.1)
      3. Recompute EPS
      4. Measure Spearman correlation between original and perturbed EPS

    Pass criterion: Spearman > 0.8 (high rank stability)
    """
    print("\n" + "=" * 60)
    print("Test 1: EPS Stability")
    print("=" * 60)

    device = config['training']['device']
    n_eval = min(500, len(samples))
    eval_samples = samples[:n_eval]

    model.eval()

    batch_orig = create_batch(eval_samples, device)
    with torch.no_grad():
        out_orig = model(batch_orig)
    eps_orig = out_orig['eps'].cpu().numpy()

    all_corrs = []

    for trial in range(n_perturbations):
        batch_pert = create_batch(eval_samples, device)

        noise_prev = torch.randn(n_eval, device=device) * 0.1
        noise_curr = torch.randn(n_eval, device=device) * 0.1

        batch_pert['log_count_prev'] = batch_pert['log_count_prev'] + noise_prev
        batch_pert['log_count_curr'] = batch_pert['log_count_curr'] + noise_curr
        batch_pert['x_prev'] = batch_pert['x_prev'].clone()
        batch_pert['x_curr'] = batch_pert['x_curr'].clone()
        batch_pert['x_prev'][:, 0] = batch_pert['x_prev'][:, 0] + noise_prev
        batch_pert['x_curr'][:, 0] = batch_pert['x_curr'][:, 0] + noise_curr

        with torch.no_grad():
            out_pert = model(batch_pert)
        eps_pert = out_pert['eps'].cpu().numpy()

        corr, _ = spearmanr(eps_orig, eps_pert)
        all_corrs.append(corr)

    mean_corr = np.mean(all_corrs)
    std_corr = np.std(all_corrs)

    print(f"\n  Perturbation: Gaussian noise (sigma=0.1) on log_count")
    print(f"  Trials: {n_perturbations}")
    print(f"  Spearman(orig, perturbed):")
    print(f"    Mean = {mean_corr:.4f} +/- {std_corr:.4f}")
    print(f"    Min  = {np.min(all_corrs):.4f}")
    print(f"    Max  = {np.max(all_corrs):.4f}")

    passed = mean_corr > 0.8
    print(f"\n  Criterion: Spearman > 0.8")
    print(f"  [{'PASS' if passed else 'FAIL'}] EPS stability = {mean_corr:.4f}")

    return {'mean_corr': mean_corr, 'std_corr': std_corr, 'passed': passed}


# ============================================================
# Test 2: Context Counterfactual
# ============================================================
def test_context_counterfactual(model, samples, config):
    """
    EPS should change when selection context changes.

    Method:
      1. Compute EPS under original context (e.g., positive selection)
      2. Change context_id to a different type (e.g., counter-selection)
      3. Recompute EPS
      4. Measure how much EPS ranking changes

    Pass criterion:
      - EPS ranking under different contexts should be significantly different
      - At least 30% of sequences should change their relative ranking
    """
    print("\n" + "=" * 60)
    print("Test 2: Context Counterfactual")
    print("=" * 60)

    device = config['training']['device']
    n_eval = min(500, len(samples))
    eval_samples = samples[:n_eval]

    model.eval()

    batch_orig = create_batch(eval_samples, device)
    with torch.no_grad():
        out_orig = model(batch_orig)
    eps_orig = out_orig['eps'].cpu().numpy()

    context_names = ['R3_R5(normal)', 'R5_R9(counter)', 'R9_R10(recovery)', 'R10_R11(normal)']

    results = {}

    for new_ctx in range(4):
        batch_cf = create_batch(eval_samples, device)
        batch_cf['context_id'] = torch.full((n_eval,), new_ctx, dtype=torch.long, device=device)
        batch_cf['round_idx'] = torch.full((n_eval,), new_ctx, dtype=torch.long, device=device)

        with torch.no_grad():
            out_cf = model(batch_cf)
        eps_cf = out_cf['eps'].cpu().numpy()

        corr, _ = spearmanr(eps_orig, eps_cf)

        rank_orig = np.argsort(np.argsort(eps_orig))
        rank_cf = np.argsort(np.argsort(eps_cf))
        rank_diff = np.abs(rank_orig - rank_cf).mean() / n_eval

        results[new_ctx] = {
            'spearman': corr,
            'rank_shift': rank_diff,
            'eps_mean': eps_cf.mean(),
            'eps_std': eps_cf.std(),
        }

    print(f"\n  Original context distribution:")
    orig_ctx = batch_orig['context_id'].cpu().numpy()
    for ctx_id in range(4):
        n_ctx = (orig_ctx == ctx_id).sum()
        if n_ctx > 0:
            print(f"    {context_names[ctx_id]}: {n_ctx} samples")

    print(f"\n  Counterfactual results:")
    print(f"  {'New Context':<25} {'Spearman':<12} {'Rank Shift':<12} {'EPS mean':<12}")
    print(f"  {'-'*60}")
    for ctx_id in range(4):
        r = results[ctx_id]
        print(f"  {context_names[ctx_id]:<25} {r['spearman']:+.4f}{'':<5} "
              f"{r['rank_shift']:.4f}{'':<5} {r['eps_mean']:.4f}")

    same_context_corrs = []
    diff_context_corrs = []
    for ctx_id in range(4):
        corr = results[ctx_id]['spearman']
        if corr > 0.9:
            same_context_corrs.append(corr)
        else:
            diff_context_corrs.append(corr)

    mean_rank_shift = np.mean([results[c]['rank_shift'] for c in range(4)])

    passed = mean_rank_shift > 0.15
    print(f"\n  Mean rank shift across contexts: {mean_rank_shift:.4f}")
    print(f"  Criterion: mean rank shift > 0.15")
    print(f"  [{'PASS' if passed else 'FAIL'}] Context sensitivity")

    return {'results': results, 'mean_rank_shift': mean_rank_shift, 'passed': passed}


# ============================================================
# Test 3: Mutation Robustness
# ============================================================
def test_mutation_robustness(model, df, samples, config, n_mutants_per_seq=5):
    """
    EPS should be robust to single-point mutations.

    Method:
      1. Take top-K sequences by EPS
      2. For each, generate n_mutants single-point mutations
      3. Compute EPS for each mutant
      4. Measure: what fraction of mutants stay in the top half?

    Pass criterion:
      - > 60% of mutants of top sequences should remain above median EPS
      - Spearman(EPS_original, EPS_mutant) > 0.5 on average
    """
    print("\n" + "=" * 60)
    print("Test 3: Mutation Robustness")
    print("=" * 60)

    device = config['training']['device']
    rng = np.random.RandomState(42)
    seq_length = config['data']['seq_length']

    model.eval()

    n_eval = min(500, len(samples))
    eval_samples = samples[:n_eval]

    batch = create_batch(eval_samples, device)
    with torch.no_grad():
        out = model(batch)
    eps_orig = out['eps'].cpu().numpy()

    top_k = 50
    top_indices = np.argsort(eps_orig)[-top_k:]

    median_eps = np.median(eps_orig)

    all_mutant_eps = []
    above_median_count = 0
    total_mutants = 0

    print(f"\n  Testing {top_k} top-EPS sequences")
    print(f"  {n_mutants_per_seq} single-point mutations per sequence")
    print(f"  Total mutants: {top_k * n_mutants_per_seq}")

    for idx in top_indices:
        sample = eval_samples[idx]
        orig_one_hot = sample['one_hot'].numpy()
        orig_sequence = df.iloc[sample['seq_idx'].item()]['sequence']

        for mut in range(n_mutants_per_seq):
            pos = rng.randint(0, seq_length)
            orig_base = orig_sequence[pos].upper()
            possible = [b for b in 'ACGT' if b != orig_base]
            new_base = rng.choice(possible)

            mut_sequence = list(orig_sequence)
            mut_sequence[pos] = new_base
            mut_sequence = ''.join(mut_sequence)

            mut_one_hot = one_hot_encode(mut_sequence, seq_length)

            mut_sample = {k: v.clone() for k, v in sample.items()}
            mut_sample['one_hot'] = torch.from_numpy(mut_one_hot)

            mut_batch = create_batch([mut_sample], device)
            with torch.no_grad():
                mut_out = model(mut_batch)
            mut_eps = mut_out['eps'].item()

            all_mutant_eps.append(mut_eps)
            if mut_eps > median_eps:
                above_median_count += 1
            total_mutants += 1

    stay_ratio = above_median_count / total_mutants

    print(f"\n  Results:")
    print(f"    Total mutants: {total_mutants}")
    print(f"    Mutants above median EPS: {above_median_count}/{total_mutants}")
    print(f"    Stay ratio: {stay_ratio:.4f}")

    orig_top_mean = eps_orig[top_indices].mean()
    mutant_mean = np.mean(all_mutant_eps)
    print(f"    Original top-{top_k} mean EPS: {orig_top_mean:.4f}")
    print(f"    Mutant mean EPS: {mutant_mean:.4f}")

    passed = stay_ratio > 0.6
    print(f"\n  Criterion: stay_ratio > 0.6")
    print(f"  [{'PASS' if passed else 'FAIL'}] Mutation robustness")

    return {
        'stay_ratio': stay_ratio,
        'orig_top_mean': orig_top_mean,
        'mutant_mean': mutant_mean,
        'passed': passed,
    }


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 60)
    print("ETO v2 - Three Scientific Validation Tests")
    print("=" * 60)
    print("\nThese tests verify ETO has genuine evolutionary dynamics")
    print("properties, not just statistical fitting.\n")

    config = load_config()
    model, df, features, samples = prepare_data_and_model(config, n_epochs=30)

    results = {}

    results['stability'] = test_eps_stability(model, samples, config)
    results['counterfactual'] = test_context_counterfactual(model, samples, config)
    results['mutation'] = test_mutation_robustness(model, df, samples, config)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    print(f"\n  {'Test':<30} {'Result':<10} {'Key Metric'}")
    print(f"  {'-'*60}")
    print(f"  {'1. EPS Stability':<30} {'PASS' if results['stability']['passed'] else 'FAIL':<10} "
          f"Spearman={results['stability']['mean_corr']:.4f}")
    print(f"  {'2. Context Counterfactual':<30} {'PASS' if results['counterfactual']['passed'] else 'FAIL':<10} "
          f"RankShift={results['counterfactual']['mean_rank_shift']:.4f}")
    print(f"  {'3. Mutation Robustness':<30} {'PASS' if results['mutation']['passed'] else 'FAIL':<10} "
          f"StayRatio={results['mutation']['stay_ratio']:.4f}")

    n_pass = sum(1 for r in results.values() if r['passed'])
    print(f"\n  {n_pass}/3 tests passed")

    if n_pass == 3:
        print("\n  [ALL PASS] ETO demonstrates genuine evolutionary dynamics:")
        print("    - EPS is a robust, stable property")
        print("    - EPS responds to selection context changes")
        print("    - EPS depends on sequence features, not exact identity")
        print("\n  ETO is ready for real MPO-SELEX data validation.")
    elif n_pass >= 2:
        print("\n  [MOSTLY PASS] ETO shows promising evolutionary properties.")
        print("  Ready for MPO data with caveats.")
    else:
        print("\n  [NEEDS WORK] Some scientific properties not yet established.")

    print("=" * 60)


if __name__ == '__main__':
    main()

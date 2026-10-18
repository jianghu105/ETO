"""
Validation 2: Temporal Generalization Test

Simulates predicting unseen future rounds.

Experiment A:
  Train: R3→R5, R5→R9, R9→R10
  Test:  R10→R11

Experiment B:
  Train: R3→R5, R5→R9, R10→R11
  Test:  R9→R10

Evaluates:
  - EPS correlation with evolutionary advantage on held-out rounds
  - Ranking stability (top-k overlap)
  - Whether model generalizes temporally or just memorizes
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import yaml
from scipy.stats import spearmanr

from data.preprocessing import (
    generate_synthetic_data, compute_features, prepare_training_data,
    ROUND_PAIRS, CONTEXT_MAP, ROUND_INDEX_MAP,
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
    config['training']['device'] = 'cpu'
    config['training']['batch_size'] = 128
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


def train_on_split(config, train_samples, n_epochs=30):
    device = config['training']['device']
    model = RefactoredETOModel(config).to(device)
    loss_fn = RefactoredETOLoss(
        lambda_transition=1.0, lambda_population=0.5,
        lambda_advantage=1.0, lambda_rollout=0.3,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'],
                                   weight_decay=config['training']['weight_decay'])

    batch_size = config['training']['batch_size']
    n = len(train_samples)

    for epoch in range(n_epochs):
        model.train()
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = create_batch([train_samples[i] for i in indices[start:end]], device)
            outputs = model(batch)
            losses = loss_fn(outputs, batch)
            optimizer.zero_grad()
            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    return model


def evaluate_on_test(model, test_samples, config, pair_name):
    device = config['training']['device']
    batch = create_batch(test_samples, device)

    with torch.no_grad():
        outputs = model(batch)

    eps = outputs['eps'].cpu().numpy()
    advantage = batch['evolutionary_advantage'].cpu().numpy()
    growth = batch['future_growth'].cpu().numpy()
    count = batch['count_curr'].cpu().numpy()

    corr_adv, p_adv = spearmanr(eps, advantage)
    corr_growth, p_growth = spearmanr(eps, growth)
    corr_count, p_count = spearmanr(eps, count)

    top_k = min(20, len(eps) // 5)
    top_eps_idx = set(np.argsort(eps)[-top_k:])
    top_adv_idx = set(np.argsort(advantage)[-top_k:])
    overlap = len(top_eps_idx & top_adv_idx) / top_k

    return {
        'corr_advantage': corr_adv,
        'p_advantage': p_adv,
        'corr_growth': corr_growth,
        'corr_count': corr_count,
        'precision_at_k': overlap,
        'eps_mean': eps.mean(),
        'eps_std': eps.std(),
        'n_samples': len(eps),
    }


def experiment_a(config, samples):
    """
    Train on early rounds (R3→R5, R5→R9, R9→R10), test on late round (R10→R11).
    """
    print("\n" + "=" * 60)
    print("Experiment A: Train on early rounds → Test on R10→R11")
    print("=" * 60)

    train_pairs = [0, 1, 2]  # R3→R5, R5→R9, R9→R10
    test_pairs = [3]          # R10→R11

    train_samples, test_samples = split_by_pairs(samples, train_pairs, test_pairs)
    print(f"  Train: {len(train_samples)} samples (pairs: R3→R5, R5→R9, R9→R10)")
    print(f"  Test:  {len(test_samples)} samples (pair: R10→R11)")

    if len(test_samples) < 10:
        print("  [SKIP] Too few test samples")
        return None

    print(f"  Training 30 epochs...")
    model = train_on_split(config, train_samples, n_epochs=30)

    results = evaluate_on_test(model, test_samples, config, "R10→R11")

    print(f"\n  Results on held-out R10→R11:")
    print(f"    EPS ~ advantage:  {results['corr_advantage']:+.4f} (p={results['p_advantage']:.2e})")
    print(f"    EPS ~ growth:     {results['corr_growth']:+.4f}")
    print(f"    EPS ~ count:      {results['corr_count']:+.4f}")
    print(f"    Precision@{min(20, len(test_samples)//5)}:     {results['precision_at_k']:.4f}")

    passed = results['corr_advantage'] > 0.1
    print(f"\n  Criterion: EPS ~ advantage > 0.1 on held-out round")
    print(f"  [{'PASS' if passed else 'FAIL'}] Temporal generalization (A)")
    results['passed'] = passed
    return results


def experiment_b(config, samples):
    """
    Train on non-contiguous rounds (R3→R5, R5→R9, R10→R11), test on gap (R9→R10).
    """
    print("\n" + "=" * 60)
    print("Experiment B: Train on non-contiguous → Test on R9→R10")
    print("=" * 60)

    train_pairs = [0, 1, 3]  # R3→R5, R5→R9, R10→R11
    test_pairs = [2]          # R9→R10

    train_samples, test_samples = split_by_pairs(samples, train_pairs, test_pairs)
    print(f"  Train: {len(train_samples)} samples (pairs: R3→R5, R5→R9, R10→R11)")
    print(f"  Test:  {len(test_samples)} samples (pair: R9→R10)")

    if len(test_samples) < 10:
        print("  [SKIP] Too few test samples")
        return None

    print(f"  Training 30 epochs...")
    model = train_on_split(config, train_samples, n_epochs=30)

    results = evaluate_on_test(model, test_samples, config, "R9→R10")

    print(f"\n  Results on held-out R9→R10:")
    print(f"    EPS ~ advantage:  {results['corr_advantage']:+.4f} (p={results['p_advantage']:.2e})")
    print(f"    EPS ~ growth:     {results['corr_growth']:+.4f}")
    print(f"    EPS ~ count:      {results['corr_count']:+.4f}")
    print(f"    Precision@{min(20, len(test_samples)//5)}:     {results['precision_at_k']:.4f}")

    passed = results['corr_advantage'] > 0.1
    print(f"\n  Criterion: EPS ~ advantage > 0.1 on held-out round")
    print(f"  [{'PASS' if passed else 'FAIL'}] Temporal generalization (B)")
    results['passed'] = passed
    return results


def experiment_c_full_vs_heldout(config, samples):
    """
    Compare: full model vs held-out model on the same test set.
    Measures how much temporal information matters.
    """
    print("\n" + "=" * 60)
    print("Experiment C: Full model vs Held-out model comparison")
    print("=" * 60)

    test_pairs = [3]
    train_full = [0, 1, 2, 3]
    train_partial = [0, 1, 2]

    _, test_samples = split_by_pairs(samples, train_full, test_pairs)
    train_full_samples, _ = split_by_pairs(samples, train_full, [])
    train_partial_samples, _ = split_by_pairs(samples, train_partial, [])

    if len(test_samples) < 10:
        print("  [SKIP] Too few test samples")
        return None

    print(f"  Training full model (all 4 pairs)...")
    model_full = train_on_split(config, train_full_samples, n_epochs=30)
    print(f"  Training partial model (3 pairs, no R10→R11)...")
    model_partial = train_on_split(config, train_partial_samples, n_epochs=30)

    device = config['training']['device']
    batch = create_batch(test_samples, device)

    with torch.no_grad():
        out_full = model_full(batch)
        out_partial = model_partial(batch)

    eps_full = out_full['eps'].cpu().numpy()
    eps_partial = out_partial['eps'].cpu().numpy()
    advantage = batch['evolutionary_advantage'].cpu().numpy()

    corr_full, _ = spearmanr(eps_full, advantage)
    corr_partial, _ = spearmanr(eps_partial, advantage)
    corr_eps, _ = spearmanr(eps_full, eps_partial)

    print(f"\n  On R10→R11 test set:")
    print(f"    Full model EPS ~ advantage:    {corr_full:+.4f}")
    print(f"    Partial model EPS ~ advantage:  {corr_partial:+.4f}")
    print(f"    Full EPS ~ Partial EPS:         {corr_eps:+.4f}")

    gap = corr_full - corr_partial
    print(f"\n  Performance gap: {gap:+.4f}")
    if gap < 0.1:
        print(f"  [INFO] Small gap: partial model generalizes well")
    elif gap < 0.3:
        print(f"  [INFO] Moderate gap: some round-specific learning")
    else:
        print(f"  [WARN] Large gap: model may be memorizing round-specific patterns")

    return {
        'corr_full': corr_full,
        'corr_partial': corr_partial,
        'corr_eps': corr_eps,
        'gap': gap,
    }


def main():
    print("=" * 60)
    print("Validation 2: Temporal Generalization Test")
    print("=" * 60)

    config = load_config()
    print("Generating data...")
    df = generate_synthetic_data(n_sequences=2000, n_families=20, seed=42)
    features = compute_features(df)
    samples = prepare_training_data(features)
    print(f"  {len(samples)} total samples")

    for pair_idx, (r_prev, r_curr) in enumerate(ROUND_PAIRS):
        n = sum(1 for s in samples if s['pair_idx'].item() == pair_idx)
        print(f"    Pair {pair_idx} ({r_prev}→{r_curr}): {n} samples")

    results = {}
    results['exp_a'] = experiment_a(config, samples)
    results['exp_b'] = experiment_b(config, samples)
    results['exp_c'] = experiment_c_full_vs_heldout(config, samples)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    for name, r in results.items():
        if r is not None:
            status = 'PASS' if r.get('passed', True) else 'FAIL'
            print(f"  {name:<15} {status}")
        else:
            print(f"  {name:<15} SKIP")

    n_pass = sum(1 for r in results.values() if r is not None and r.get('passed', False))
    n_total = sum(1 for r in results.values() if r is not None)
    print(f"\n  {n_pass}/{n_total} experiments passed")
    print("=" * 60)

    return results


if __name__ == '__main__':
    main()

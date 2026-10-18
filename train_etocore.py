"""
ETO-Core Training Entry Point
==============================

Trains Static, History, and ETO-Core models on real MPO-SELEX data.
Designed for Google Colab with GPU support.

Usage:
    python train_etocore.py [--data_path PATH] [--output_dir DIR] [--seed SEED]
"""

import os
import sys
import json
import argparse
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.unified_dataset import build_etocore_dataset
from data.eto_dataset import build_dataloaders
from models.eto_core import build_model
from training.eto_core_trainer import Trainer, evaluate_on_test


def detect_environment() -> dict:
    """Detect and report the runtime environment."""
    env = {
        'python_version': sys.version,
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        env['gpu_name'] = torch.cuda.get_device_name(0)
        env['gpu_memory_gb'] = round(
            torch.cuda.get_device_properties(0).total_mem / 1e9, 1)
    else:
        env['gpu_name'] = None
        env['gpu_memory_gb'] = 0
    return env


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def auto_batch_size(gpu_memory_gb: float) -> int:
    if gpu_memory_gb >= 14:
        return 512
    elif gpu_memory_gb >= 7:
        return 256
    else:
        return 128


def train_single_model(model_type: str, dataset_dict: dict,
                       config: dict, output_dir: str,
                       device: str) -> dict:
    """Train one model variant and return results."""
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} model")
    print(f"{'='*60}")

    set_seed(config['seed'])

    loaders = build_dataloaders(
        dataset_dict,
        batch_size=config['batch_size'],
        num_workers=0,
    )

    model = build_model(model_type)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    trainer = Trainer(model, config, device=device)

    checkpoint_path = os.path.join(output_dir, 'checkpoint_last.pt')
    if os.path.exists(checkpoint_path):
        print(f"  Resuming from {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path)

    t0 = time.time()
    train_result = trainer.train(
        loaders['train'], loaders['val'], output_dir)
    train_time = time.time() - t0

    print(f"\n  Training completed in {train_time:.1f}s")
    print(f"  Best epoch: {train_result['best_epoch']}, "
          f"val_loss: {train_result['best_val_loss']:.4f}")

    test_results = {}
    if 'test' in loaders:
        print(f"\n  Evaluating on test set...")
        test_results = evaluate_on_test(
            model, loaders['test'], dataset_dict['normalizer'], device)

        print(f"  Test Spearman: {test_results['overall']['spearman']:.4f}")
        print(f"  Test NDCG@50:  {test_results['overall']['ndcg@50']:.4f}")
        print(f"  Test MAE:      {test_results['overall']['mae']:.4f}")

        for t_name, t_metrics in test_results['per_transition'].items():
            print(f"    {t_name}: Spearman={t_metrics['spearman']:.4f}, "
                  f"NDCG@50={t_metrics['ndcg@50']:.4f}")

    ood_results = {}
    if 'test_ood_temporal' in loaders and len(loaders['test_ood_temporal'].dataset) > 0:
        print(f"\n  Evaluating on temporal OOD test set...")
        ood_results = evaluate_on_test(
            model, loaders['test_ood_temporal'],
            dataset_dict['normalizer'], device)
        print(f"  OOD Spearman: {ood_results['overall']['spearman']:.4f}")

    results = {
        'model_type': model_type,
        'n_params': n_params,
        'train_time_s': train_time,
        'best_epoch': train_result['best_epoch'],
        'best_val_loss': train_result['best_val_loss'],
        'test': {
            'overall': test_results.get('overall', {}),
            'per_transition': test_results.get('per_transition', {}),
        },
        'ood_temporal': {
            'overall': ood_results.get('overall', {}),
        },
        'seed': config['seed'],
    }

    results_path = os.path.join(output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return results


def main():
    parser = argparse.ArgumentParser(description='ETO-Core Training')
    parser.add_argument('--data_path', type=str,
                       default='dataset/20230531AptDMC双端-MPO.xlsx')
    parser.add_argument('--output_dir', type=str,
                       default='experiments/etocore_v1')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=0,
                       help='0 = auto-detect from GPU memory')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--models', nargs='+',
                       default=['static', 'history', 'etocore'])
    parser.add_argument('--smoke_test', action='store_true',
                       help='Run a quick smoke test instead of full training')
    args = parser.parse_args()

    env = detect_environment()
    print("=" * 60)
    print("ETO-Core Training Pipeline")
    print("=" * 60)
    print(f"Python: {env['python_version'].split()[0]}")
    print(f"PyTorch: {env['torch_version']}")
    print(f"CUDA: {env['cuda_available']}")
    if env['cuda_available']:
        print(f"GPU: {env['gpu_name']} ({env['gpu_memory_gb']} GB)")
    print()

    device = 'cuda' if env['cuda_available'] else 'cpu'
    if not env['cuda_available']:
        print("WARNING: No GPU detected. Training will be slow on CPU.")
        print("For production use, run on Google Colab with GPU enabled.\n")

    batch_size = args.batch_size
    if batch_size == 0:
        batch_size = auto_batch_size(env['gpu_memory_gb'])
    print(f"Batch size: {batch_size}")

    epochs = 5 if args.smoke_test else args.epochs

    config = {
        'seed': args.seed,
        'batch_size': batch_size,
        'training': {
            'lr': args.lr,
            'weight_decay': 1e-4,
            'max_grad_norm': 1.0,
            'epochs': epochs,
            'patience': 15 if not args.smoke_test else 3,
            'huber_delta': 1.0,
            'use_amp': True,
        },
    }

    print(f"\nLoading and preparing data...")
    dataset_dict = build_etocore_dataset(
        data_path=args.data_path,
        seq_length=36,
        min_total_count=10,
        train_frac=0.7,
        val_frac=0.15,
        seed=args.seed,
    )

    metadata = dataset_dict['metadata']
    print(f"\nData summary:")
    print(f"  Sequences: {metadata['n_sequences']}")
    print(f"  Families: {metadata['n_families']}")
    print(f"  Total samples: {metadata['total_samples']}")
    print(f"  Transitions: {metadata['valid_transitions']}")

    os.makedirs(args.output_dir, exist_ok=True)

    env_path = os.path.join(args.output_dir, 'environment.json')
    with open(env_path, 'w') as f:
        json.dump({**env, 'config': config, 'metadata': metadata}, f,
                 indent=2, default=str)

    all_results = {}
    for model_type in args.models:
        model_dir = os.path.join(args.output_dir, model_type)
        os.makedirs(model_dir, exist_ok=True)
        results = train_single_model(
            model_type, dataset_dict, config, model_dir, device)
        all_results[model_type] = results

    summary_path = os.path.join(args.output_dir, 'all_results.json')
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<12} {'Spearman':>10} {'NDCG@50':>10} {'MAE':>10} "
          f"{'Params':>10}")
    print("-" * 55)
    for mt, r in all_results.items():
        test = r.get('test', {}).get('overall', {})
        print(f"{mt:<12} "
              f"{test.get('spearman', 0):>10.4f} "
              f"{test.get('ndcg@50', 0):>10.4f} "
              f"{test.get('mae', 0):>10.4f} "
              f"{r['n_params']:>10,}")

    print(f"\nResults saved to {args.output_dir}/")
    return all_results


if __name__ == '__main__':
    main()

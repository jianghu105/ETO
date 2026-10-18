import os
import sys
import yaml
import logging
import argparse
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.preprocessing import load_real_data, compute_features, prepare_training_data
from data.dataset import CompactSELEXDataset
from data.population_sampler import build_population_dataloader
from models.eto_model import RefactoredETOModel
from evaluation.metrics import evaluate_model
from evaluation.baselines import evaluate_baselines
from evaluation.visualization import (
    plot_phi_vs_future_count, plot_phi_distribution,
    plot_round_trajectory
)


def main():
    parser = argparse.ArgumentParser(description='ETO v2 Evaluation')
    parser.add_argument('--config', type=str, default='configs/train_real.yaml')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Checkpoint path. If None, auto-detects best checkpoint in output_dir')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory. If None, uses config evaluation.output_dir')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger('ETO-eval')

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.output_dir is None:
        args.output_dir = config.get('evaluation', {}).get('output_dir', 'experiments/real_v2')

    device = config['training'].get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    config['training']['device'] = device

    logger.info(f"Device: {device}")
    logger.info("Loading data...")

    data_cfg = config['data']
    df = load_real_data(
        data_cfg['real_data_path'],
        max_sequences=data_cfg.get('max_sequences', None),
        min_total_count=data_cfg.get('min_total_count', 1),
        seq_length=data_cfg.get('seq_length', 36),
    )
    features = compute_features(df)
    arrays = prepare_training_data(features)

    dataset = CompactSELEXDataset(arrays)
    eval_loader = torch.utils.data.DataLoader(
        dataset, batch_size=config['training'].get('batch_size', 512),
        shuffle=False, num_workers=0
    )

    if args.checkpoint is None:
        best_path = os.path.join(args.output_dir, 'checkpoint_best.pt')
        regular_path = os.path.join(args.output_dir, 'checkpoint.pt')
        if os.path.exists(best_path):
            args.checkpoint = best_path
        elif os.path.exists(regular_path):
            args.checkpoint = regular_path
        else:
            logger.error(f"No checkpoint found in {args.output_dir}")
            return

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model = RefactoredETOModel(config)
    model.to(device)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        logger.info(f"Loaded epoch {ckpt['epoch']}, best loss: {ckpt['best_loss']:.4f}")
    else:
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        return

    logger.info("Running evaluation...")
    metrics, phi, ea = evaluate_model(model, eval_loader, device)

    logger.info("Evaluation Results:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)

    plot_phi_vs_future_count(phi, ea, args.output_dir)
    plot_phi_distribution(phi, args.output_dir)

    results_df = pd.DataFrame({
        'sequence': df['sequence'].values[:len(phi)],
        'EPS_score': phi,
        'rank': pd.Series(phi).rank(ascending=False).astype(int).values,
    })
    results_df = results_df.sort_values('rank')
    results_df.to_csv(os.path.join(args.output_dir, 'predicted_potential.csv'), index=False)

    pd.DataFrame([metrics]).to_csv(
        os.path.join(args.output_dir, 'metrics.csv'), index=False)

    logger.info("Running baselines...")
    baseline_results = evaluate_baselines(arrays, df, features)
    for name, res in baseline_results.items():
        logger.info(f"  Baseline [{name}]:")
        for k, v in res.items():
            logger.info(f"    {k}: {v:.4f}")

    import json
    with open(os.path.join(args.output_dir, 'baseline_results.json'), 'w') as f:
        json.dump(baseline_results, f, indent=2)

    with open(os.path.join(args.output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()

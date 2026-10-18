import os
import sys
import yaml
import logging
import argparse
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.dataset import SELEXDataModule
from models.eto_model import ETOModel
from evaluation.metrics import evaluate_model
from evaluation.visualization import (
    plot_phi_vs_future_count, plot_phi_distribution,
    plot_round_trajectory
)


def main():
    parser = argparse.ArgumentParser(description='ETO v1 Evaluation')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--checkpoint', type=str, default='results/checkpoint.pt')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    logger = logging.getLogger('ETO-eval')

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.synthetic:
        config['data']['synthetic'] = {'enabled': True, 'n_sequences': 2000,
                                        'n_families': 20, 'seed': 42}

    device = config['training'].get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    config['training']['device'] = device

    logger.info("Loading data...")
    data_module = SELEXDataModule(config)
    data_module.load_data()
    eval_loader = data_module.get_dataloader(shuffle=False)

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model = ETOModel(config)
    model.to(device)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        logger.info(f"Loaded epoch {ckpt['epoch']}, best loss: {ckpt['best_loss']:.4f}")
    else:
        logger.warning(f"Checkpoint not found: {args.checkpoint}")
        return

    logger.info("Evaluating...")
    metrics, phi, counts = evaluate_model(model, eval_loader, device)

    logger.info("Evaluation Results:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)

    plot_phi_vs_future_count(phi, counts, args.output_dir)
    plot_phi_distribution(phi, args.output_dir)
    plot_round_trajectory(model, eval_loader, device, args.output_dir)

    results_df = pd.DataFrame({
        'sequence': data_module.df['sequence'].values[:len(phi)],
        'EPS_score': phi,
        'rank': pd.Series(phi).rank(ascending=False).astype(int).values,
    })
    results_df = results_df.sort_values('rank')
    results_df.to_csv(os.path.join(args.output_dir, 'predicted_potential.csv'), index=False)

    pd.DataFrame([metrics]).to_csv(
        os.path.join(args.output_dir, 'metrics.csv'), index=False)

    logger.info(f"Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()

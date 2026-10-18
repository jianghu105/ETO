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
from data.preprocessing import ROUND_NAMES, ROUND_PAIRS
from models.eto_model import RefactoredETOModel
from training.trainer import Trainer
from evaluation.metrics import evaluate_model, compute_spearman_correlation
from evaluation.visualization import (
    plot_training_loss, plot_phi_vs_future_count,
    plot_ablation_comparison, plot_round_trajectory, plot_phi_distribution
)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def setup_logging(output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'training.log')),
            logging.StreamHandler(sys.stdout),
        ]
    )


def main():
    parser = argparse.ArgumentParser(description='ETO v2 Training')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                        help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--synthetic', action='store_true',
                        help='Use synthetic data')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    if args.synthetic:
        config['data']['synthetic'] = {'enabled': True, 'n_sequences': 2000,
                                        'n_families': 20, 'seed': 42}
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    setup_logging(output_dir)
    logger = logging.getLogger('ETO')

    seed = config['training'].get('seed', 42)
    set_seed(seed)

    device = config['training'].get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        logger.info("CUDA not available, falling back to CPU")
    config['training']['device'] = device
    logger.info(f"Using device: {device}")

    logger.info("Loading data...")
    data_module = SELEXDataModule(config)
    data_module.load_data()
    train_loader = data_module.get_dataloader()

    logger.info("Building RefactoredETOModel...")
    model = RefactoredETOModel(config)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    trainer = Trainer(model, config, device)
    logger.info("Starting training...")
    history = trainer.train(train_loader, output_dir=output_dir)

    logger.info("Generating visualizations...")
    plot_training_loss(history, output_dir)

    logger.info("=" * 60)
    logger.info("ETO v2 Training Complete")
    logger.info(f"Results saved to: {output_dir}/")
    logger.info("=" * 60)

    return model, history


if __name__ == '__main__':
    main()

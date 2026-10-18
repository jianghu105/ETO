"""
ETO Real MPO-SELEX Training Script

Trains ETO on real MPO-SELEX data with:
  - Abundance-free pathway
  - Continuous-time evolutionary dynamics
  - Evolutionary Potential Score (EPS) supervised by evolutionary_advantage
  - Population-aware sampling
  - Training monitoring (NaN, abundance leakage, loss components)
"""

import os
import sys
import yaml
import argparse
import logging
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.preprocessing import load_real_data, compute_features, prepare_training_data
from data.dataset import SELEXDataset, SELEXDataModule
from data.population_sampler import PopulationSampler, SELEXRealDataset, build_population_dataloader
from models.eto_model import RefactoredETOModel
from training.trainer import Trainer
from training.losses import RefactoredETOLoss
from evaluation.metrics import evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def build_dataloaders(config: dict):
    """Build train/val dataloaders for real data."""
    data_cfg = config['data']
    training_cfg = config['training']
    sampling_cfg = config.get('sampling', {})

    logger.info("Loading real MPO-SELEX data...")
    df = load_real_data(
        data_cfg['real_data_path'],
        max_sequences=data_cfg.get('max_sequences'),
        min_total_count=data_cfg.get('min_total_count', 1),
        seq_length=data_cfg.get('seq_length', 36),
    )

    logger.info("Computing features...")
    features = compute_features(df)

    logger.info("Preparing training data (compact arrays)...")
    arrays = prepare_training_data(features)
    n_samples = len(arrays['pair_idx'])
    logger.info(f"Total samples: {n_samples}")

    use_sampling = sampling_cfg.get('enabled', False)

    if use_sampling:
        logger.info("Using population-aware sampling")
        train_loader = build_population_dataloader(
            arrays, features, config
        )
        val_loader = None
    else:
        from data.dataset import CompactSELEXDataset
        dataset = CompactSELEXDataset(
            arrays,
            count_dropout_prob=training_cfg.get('count_dropout_prob', 0.0),
        )
        train_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=training_cfg.get('batch_size', 256),
            shuffle=True,
            num_workers=training_cfg.get('num_workers', 0),
            pin_memory=True,
            drop_last=False,
        )
        val_loader = None

    return train_loader, val_loader, df, features, arrays


def train(config: dict, output_dir: str):
    """Main training loop."""
    os.makedirs(output_dir, exist_ok=True)

    log_path = os.path.join(output_dir, 'training.log')
    file_handler = logging.FileHandler(log_path, mode='w')
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    logger.addHandler(file_handler)

    seed = config['training'].get('seed', 42)
    set_seed(seed)

    device = config['training'].get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        logger.warning("CUDA not available, falling back to CPU")
    config['training']['device'] = device

    logger.info("=" * 60)
    logger.info("ETO Real MPO-SELEX Training")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Config: {config}")

    train_loader, val_loader, df, features, arrays = build_dataloaders(config)

    logger.info(f"Dataset: {len(df)} sequences, {len(arrays['pair_idx'])} training samples")
    logger.info(f"Batch size: {config['training']['batch_size']}")
    logger.info(f"Batches per epoch: {len(train_loader)}")

    model = RefactoredETOModel(config)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    trainer = Trainer(model, config, device=device)

    logger.info("Starting training...")
    history = trainer.train(
        train_loader,
        val_loader=val_loader,
        output_dir=output_dir,
    )

    import csv
    metrics_path = os.path.join(output_dir, 'metrics.csv')
    fieldnames = ['epoch', 'train_total', 'train_transition',
                  'train_population', 'train_advantage',
                  'train_ea_regression',
                  'train_rollout', 'train_smooth']
    if 'abundance_leakage' in history[0].get('train', {}):
        fieldnames.append('train_abundance_leakage')
    if 'nan_count' in history[0].get('train', {}):
        fieldnames.append('train_nan_count')
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for entry in history:
            row = {'epoch': entry['epoch']}
            for k, v in entry['train'].items():
                row[f'train_{k}'] = v
            writer.writerow(row)

    logger.info(f"Training complete. Results saved to {output_dir}")
    logger.info(f"Best loss: {trainer.best_loss:.4f}")

    logger.info("Running post-training evaluation...")
    model.eval()

    from data.dataset import CompactSELEXDataset
    eval_dataset = CompactSELEXDataset(arrays, count_dropout_prob=0.0)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=training_cfg.get('batch_size', 512),
        shuffle=False, num_workers=0
    )
    eval_metrics, phi, ea = evaluate_model(model, eval_loader, device)
    logger.info("Evaluation metrics:")
    for k, v in eval_metrics.items():
        logger.info(f"  {k}: {v:.4f}")

    import json
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(eval_metrics, f, indent=2)

    logger.info("Running abundance ranking baseline...")
    from evaluation.baselines import AbundanceRankingBaseline
    bl = AbundanceRankingBaseline()
    bl_result = bl.evaluate(
        arrays['log_count_curr'],
        arrays['evolutionary_advantage'],
        arrays['relative_abundance_curr'],
        ea_valid=arrays.get('ea_valid', np.ones(len(arrays['pair_idx']), dtype=bool)),
        pair_indices=arrays['pair_idx']
    )
    baseline_results = {'abundance_ranking': bl_result}
    logger.info(f"  Abundance Ranking: spearman={bl_result.get('spearman', 0):.4f}")

    with open(os.path.join(output_dir, 'baseline_results.json'), 'w') as f:
        json.dump(baseline_results, f, indent=2)

    logger.info(f"All results saved to {output_dir}/")

    return model, history, features, df


def main():
    parser = argparse.ArgumentParser(description='Train ETO on real MPO-SELEX data')
    parser.add_argument('--config', type=str, default='configs/train_real.yaml',
                        help='Path to training config')
    parser.add_argument('--output_dir', type=str, default='experiments/real_v1',
                        help='Output directory')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Override batch size')
    args = parser.parse_args()

    config = load_config(args.config)

    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size

    model, history, features, df = train(config, args.output_dir)

    logger.info("Done.")


if __name__ == '__main__':
    main()

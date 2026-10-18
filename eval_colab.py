"""
Standalone evaluation script for Colab.
Usage:
  !python eval_colab.py
"""
import os
import sys
import json
import yaml
import logging
import torch
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data.preprocessing import load_real_data, compute_features, prepare_training_data
from data.dataset import CompactSELEXDataset
from models.eto_model import RefactoredETOModel
from evaluation.metrics import evaluate_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger('ETO-eval')


def main():
    config_path = 'configs/train_real.yaml'
    output_dir = 'experiments/real_v1'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config['training']['device'] = device
    logger.info(f"Device: {device}")

    # Find checkpoint
    for name in ['checkpoint_best.pt', 'checkpoint.pt']:
        ckpt_path = os.path.join(output_dir, name)
        if os.path.exists(ckpt_path):
            break
    else:
        logger.error(f"No checkpoint found in {output_dir}/")
        return

    logger.info(f"Checkpoint: {ckpt_path}")

    # Load data
    logger.info("Loading data...")
    data_cfg = config['data']
    df = load_real_data(
        data_cfg['real_data_path'],
        max_sequences=data_cfg.get('max_sequences'),
        min_total_count=data_cfg.get('min_total_count', 1),
        seq_length=data_cfg.get('seq_length', 36),
    )
    features = compute_features(df)
    arrays = prepare_training_data(features)

    # Build eval loader (full dataset, no sampling)
    eval_dataset = CompactSELEXDataset(arrays, count_dropout_prob=0.0)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=config['training'].get('batch_size', 512),
        shuffle=False, num_workers=0
    )

    # Load model
    logger.info("Loading model...")
    model = RefactoredETOModel(config)
    model.to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    logger.info(f"Loaded epoch {ckpt['epoch']}, best loss: {ckpt['best_loss']:.4f}")

    # Evaluate
    logger.info("Running evaluation...")
    model.eval()
    eval_metrics, phi, ea = evaluate_model(model, eval_loader, device)

    logger.info("=" * 50)
    logger.info("ETO Evaluation Results:")
    for k, v in eval_metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    logger.info("=" * 50)

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(eval_metrics, f, indent=2)

    # Baselines (abundance ranking only - CNN baseline requires separate training)
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
    logger.info(f"  Abundance Ranking: spearman={bl_result.get('spearman', 0):.4f} "
                f"ndcg@50={bl_result.get('ndcg@50', 0):.4f}")

    baseline_results = {'abundance_ranking': bl_result}
    for name, res in baseline_results.items():
        logger.info(f"  [{name}] spearman={res.get('spearman', 0):.4f} "
                     f"ndcg@50={res.get('ndcg@50', 0):.4f} "
                     f"low_abund={res.get('low_abundance_discovery', 0):.4f}")

    with open(os.path.join(output_dir, 'baseline_results.json'), 'w') as f:
        json.dump(baseline_results, f, indent=2)

    # Save EPS scores (take R3->R5 transition, first 227K entries)
    n_seqs = len(df)
    phi_r3r5 = phi[:n_seqs]
    results_df = pd.DataFrame({
        'sequence': df['sequence'].values,
        'EPS_score': phi_r3r5,
        'rank': pd.Series(phi_r3r5).rank(ascending=False).astype(int).values,
    })
    results_df.sort_values('rank').to_csv(
        os.path.join(output_dir, 'predicted_potential.csv'), index=False)

    logger.info(f"All results saved to {output_dir}/")


if __name__ == '__main__':
    main()

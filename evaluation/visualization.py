import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import seaborn as sns


def plot_training_loss(history: List[Dict], output_dir: str = 'results'):
    os.makedirs(output_dir, exist_ok=True)

    epochs = [h['epoch'] for h in history]
    train_total = [h['train']['total'] for h in history]
    train_trans = [h['train']['transition'] for h in history]
    train_pop = [h['train']['population'] for h in history]
    train_rank = [h['train']['ranking'] for h in history]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].plot(epochs, train_total, 'b-', linewidth=2)
    axes[0, 0].set_title('Total Training Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, train_trans, 'r-', linewidth=2, label='Transition')
    axes[0, 1].plot(epochs, train_pop, 'g-', linewidth=2, label='Population')
    axes[0, 1].plot(epochs, train_rank, 'm-', linewidth=2, label='Ranking')
    axes[0, 1].set_title('Component Losses')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    if history[0].get('val') is not None:
        val_total = [h['val']['total'] for h in history if h['val'] is not None]
        axes[1, 0].plot(epochs[:len(val_total)], train_total[:len(val_total)],
                        'b-', label='Train', linewidth=2)
        axes[1, 0].plot(epochs[:len(val_total)], val_total, 'r-', label='Val', linewidth=2)
        axes[1, 0].set_title('Train vs Validation')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Total Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

    log_total = np.log1p(np.array(train_total))
    axes[1, 1].plot(epochs, log_total, 'b-', linewidth=2)
    axes[1, 1].set_title('Log-Scaled Loss')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('log(1 + Loss)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_loss.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_phi_vs_future_count(phi: np.ndarray, future_count: np.ndarray,
                             output_dir: str = 'results'):
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(phi, future_count, alpha=0.3, s=10, c='steelblue')
    axes[0].set_xlabel('Evolutionary Potential Score (Phi)')
    axes[0].set_ylabel('Future Abundance (count)')
    axes[0].set_title('Phi vs Future Abundance')
    axes[0].grid(True, alpha=0.3)

    from scipy import stats
    corr, pval = stats.spearmanr(phi, future_count)
    axes[0].text(0.05, 0.95, f'Spearman r={corr:.3f}\np={pval:.2e}',
                 transform=axes[0].transAxes, fontsize=12,
                 verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    sorted_idx = np.argsort(-phi)
    top_k_frac = min(50, len(phi) // 10)
    top_phi_mean = np.mean(future_count[sorted_idx[:top_k_frac]])
    bottom_phi_mean = np.mean(future_count[sorted_idx[-top_k_frac:]])

    axes[1].bar(['Top Phi', 'Bottom Phi'], [top_phi_mean, bottom_phi_mean],
                color=['#2196F3', '#FF5722'])
    axes[1].set_ylabel('Mean Future Count')
    axes[1].set_title('Top vs Bottom Phi Sequences')
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phi_vs_count.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_ablation_comparison(metrics: Dict[str, Dict],
                             output_dir: str = 'results'):
    os.makedirs(output_dir, exist_ok=True)

    metric_names = ['spearman_corr', 'ndcg@10', 'top10_overlap']
    model_names = list(metrics.keys())
    n_models = len(model_names)
    n_metrics = len(metric_names)

    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
    if n_metrics == 1:
        axes = [axes]

    colors = ['#2196F3', '#FF9800', '#4CAF50', '#E91E63']

    for j, metric in enumerate(metric_names):
        values = []
        for model_name in model_names:
            values.append(metrics[model_name].get(metric, 0))

        bars = axes[j].bar(model_names, values,
                           color=colors[:n_models])
        axes[j].set_title(metric.replace('_', ' ').title())
        axes[j].set_ylabel(metric)
        axes[j].grid(True, alpha=0.3, axis='y')
        axes[j].tick_params(axis='x', rotation=30)

        for bar, val in zip(bars, values):
            axes[j].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                         f'{val:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ablation_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_round_trajectory(model, dataloader, device: str,
                          output_dir: str = 'results'):
    os.makedirs(output_dir, exist_ok=True)

    round_names = ['R3->R5', 'R5->R9', 'R9->R10', 'R10->R11']
    trans_losses = []

    model.eval()
    with torch.no_grad():
        for pair_idx in range(4):
            pair_losses = []
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                mask = batch['pair_idx'] == pair_idx
                if mask.sum() == 0:
                    continue

                sub_batch = {k: v[mask] for k, v in batch.items()}
                outputs = model(sub_batch)

                loss = ((outputs['z_next_pred'] - outputs['z_t_target']) ** 2).mean()
                pair_losses.append(loss.item())

            trans_losses.append(np.mean(pair_losses) if pair_losses else 0)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(round_names, trans_losses,
                  color=['#4CAF50', '#F44336', '#2196F3', '#4CAF50'])
    ax.set_title('Transition Loss by Round Pair')
    ax.set_ylabel('MSE Transition Loss')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, trans_losses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'round_trajectory.png'),
                dpi=150, bbox_inches='tight')
    plt.close()


def plot_phi_distribution(phi: np.ndarray, output_dir: str = 'results'):
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(phi, bins=50, color='steelblue', alpha=0.7, edgecolor='white')
    ax.set_xlabel('Evolutionary Potential Score (Phi)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Predicted Phi Scores')
    ax.axvline(np.median(phi), color='red', linestyle='--', label=f'Median: {np.median(phi):.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'phi_distribution.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

"""
ETO-Core Training Pipeline
===========================

Masked Huber loss, early stopping, checkpoint save/resume, multi-seed support.
Compatible with Google Colab GPU training.
"""

import os
import json
import time
import copy
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, List
from scipy import stats


class MaskedHuberLoss(nn.Module):
    """Huber loss that respects target validity mask."""

    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        if mask.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        p = pred[mask]
        t = target[mask]
        return nn.functional.huber_loss(p, t, delta=self.delta)


def compute_metrics(pred: np.ndarray, target: np.ndarray,
                    mask: np.ndarray) -> Dict[str, float]:
    """Compute evaluation metrics on valid targets."""
    if mask.sum() < 2:
        return {'spearman': 0.0, 'pearson': 0.0, 'mae': 0.0, 'rmse': 0.0,
                'n_valid': int(mask.sum())}

    p = pred[mask]
    t = target[mask]

    spearman = float(stats.spearmanr(p, t).statistic) if len(p) > 2 else 0.0
    pearson = float(stats.pearsonr(p, t).statistic) if len(p) > 2 else 0.0
    mae = float(np.mean(np.abs(p - t)))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))

    return {
        'spearman': spearman,
        'pearson': pearson,
        'mae': mae,
        'rmse': rmse,
        'n_valid': int(mask.sum()),
    }


def compute_ndcg(pred: np.ndarray, target: np.ndarray,
                 mask: np.ndarray, k: int = 50) -> float:
    """NDCG@K on valid targets."""
    if mask.sum() < 2:
        return 0.0

    p = pred[mask]
    t = target[mask]

    order_p = np.argsort(-p)[:k]
    order_t = np.argsort(-t)
    rank_map = {idx: rank for rank, idx in enumerate(order_t)}

    dcg = 0.0
    for i, idx in enumerate(order_p):
        rel = t[idx]
        rank_in_target = rank_map.get(idx, len(t))
        dcg += (2 ** rel - 1) / np.log2(i + 2)

    ideal = np.sort(t)[::-1][:k]
    idcg = sum((2 ** r - 1) / np.log2(i + 2) for i, r in enumerate(ideal))

    return float(dcg / (idcg + 1e-8))


class Trainer:
    """
    Training loop with:
    - Masked Huber loss
    - Gradient clipping
    - AMP (mixed precision) on GPU
    - Early stopping on validation loss
    - Best checkpoint saving
    - Resume from checkpoint
    - Training log (JSON lines)
    """

    def __init__(self, model: nn.Module, config: Dict, device: str = 'cpu'):
        self.model = model.to(device)
        self.config = config
        self.device = device

        train_cfg = config.get('training', {})
        self.lr = train_cfg.get('lr', 1e-3)
        self.weight_decay = train_cfg.get('weight_decay', 1e-4)
        self.max_grad_norm = train_cfg.get('max_grad_norm', 1.0)
        self.epochs = train_cfg.get('epochs', 100)
        self.patience = train_cfg.get('patience', 15)
        self.huber_delta = train_cfg.get('huber_delta', 1.0)
        self.use_amp = train_cfg.get('use_amp', True) and device != 'cpu'

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=self.epochs)
        self.criterion = MaskedHuberLoss(delta=self.huber_delta)
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        self.best_val_loss = float('inf')
        self.best_state = None
        self.best_epoch = 0
        self.patience_counter = 0
        self.train_log = []
        self.start_epoch = 0

    def train_epoch(self, loader) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}

            self.optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=self.use_amp):
                out = self.model(batch)
                loss = self.criterion(out['prediction'], batch['target'],
                                      batch['target_mask'])

            if batch['target_mask'].sum() == 0:
                continue

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()
            n_batches += 1

        return {'loss': total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def validate(self, loader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_pred = []
        all_target = []
        all_mask = []

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.model(batch)
            loss = self.criterion(out['prediction'], batch['target'],
                                  batch['target_mask'])

            total_loss += loss.item()
            n_batches += 1

            all_pred.append(out['prediction'].cpu().numpy())
            all_target.append(batch['target'].cpu().numpy())
            all_mask.append(batch['target_mask'].cpu().numpy())

        pred = np.concatenate(all_pred)
        target = np.concatenate(all_target)
        mask = np.concatenate(all_mask)

        metrics = compute_metrics(pred, target, mask)
        metrics['loss'] = total_loss / max(n_batches, 1)
        metrics['ndcg@50'] = compute_ndcg(pred, target, mask, k=50)
        return metrics

    @torch.no_grad()
    def predict(self, loader) -> Dict[str, np.ndarray]:
        self.model.eval()
        all_pred = []
        all_target = []
        all_mask = []
        all_transition = []

        for batch in loader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.model(batch)
            all_pred.append(out['prediction'].cpu().numpy())
            all_target.append(batch['target'].cpu().numpy())
            all_mask.append(batch['target_mask'].cpu().numpy())
            all_transition.append(batch['transition_type'].cpu().numpy())

        return {
            'pred': np.concatenate(all_pred),
            'target': np.concatenate(all_target),
            'mask': np.concatenate(all_mask),
            'transition': np.concatenate(all_transition),
        }

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'best_state': self.best_state,
            'metrics': metrics,
            'config': self.config,
        }, path)

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        self.scaler.load_state_dict(ckpt['scaler_state_dict'])
        self.best_val_loss = ckpt['best_val_loss']
        self.best_epoch = ckpt['best_epoch']
        self.best_state = ckpt.get('best_state')
        self.start_epoch = ckpt['epoch'] + 1
        return ckpt.get('metrics', {})

    def train(self, train_loader, val_loader, output_dir: str) -> Dict:
        os.makedirs(output_dir, exist_ok=True)
        log_path = os.path.join(output_dir, 'train_log.jsonl')

        for epoch in range(self.start_epoch, self.epochs):
            t0 = time.time()
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            self.scheduler.step()

            elapsed = time.time() - t0
            entry = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'val_loss': val_metrics['loss'],
                'val_spearman': val_metrics['spearman'],
                'val_pearson': val_metrics['pearson'],
                'val_mae': val_metrics['mae'],
                'val_rmse': val_metrics['rmse'],
                'val_ndcg@50': val_metrics.get('ndcg@50', 0.0),
                'lr': self.optimizer.param_groups[0]['lr'],
                'time': elapsed,
            }
            self.train_log.append(entry)

            with open(log_path, 'a') as f:
                f.write(json.dumps(entry) + '\n')

            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']
                self.best_epoch = epoch
                self.best_state = copy.deepcopy(self.model.state_dict())
                self.patience_counter = 0
                self.save_checkpoint(
                    os.path.join(output_dir, 'checkpoint_best.pt'),
                    epoch, val_metrics)
            else:
                self.patience_counter += 1

            if epoch % 10 == 0 or is_best:
                print(f"  Epoch {epoch:3d} | "
                      f"train_loss={train_metrics['loss']:.4f} "
                      f"val_loss={val_metrics['loss']:.4f} "
                      f"val_spearman={val_metrics['spearman']:.4f} "
                      f"val_ndcg={val_metrics.get('ndcg@50', 0):.4f} "
                      f"{'* best' if is_best else ''}")

            if self.patience_counter >= self.patience:
                print(f"  Early stopping at epoch {epoch} "
                      f"(best: {self.best_epoch})")
                break

        self.save_checkpoint(
            os.path.join(output_dir, 'checkpoint_last.pt'),
            epoch, val_metrics)

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            self.save_checkpoint(
                os.path.join(output_dir, 'checkpoint_best.pt'),
                self.best_epoch, val_metrics)

        return {
            'best_epoch': self.best_epoch,
            'best_val_loss': self.best_val_loss,
            'final_epoch': epoch,
            'train_log': self.train_log,
        }


def evaluate_on_test(model: nn.Module, test_loader, normalizer,
                     device: str = 'cpu') -> Dict:
    """Evaluate model on test set with per-transition breakdown."""
    model.eval()
    trainer = Trainer(model, {'training': {}}, device=device)
    predictions = trainer.predict(test_loader)

    pred = predictions['pred']
    target = predictions['target']
    mask = predictions['mask']
    transition = predictions['transition']

    pred_original = normalizer.inverse_transform(pred)
    target_original = normalizer.inverse_transform(target)

    overall = compute_metrics(pred_original, target_original, mask)
    overall['ndcg@50'] = compute_ndcg(pred_original, target_original, mask, k=50)

    per_transition = {}
    transition_names = {0: 'R3_R5', 1: 'R5_R9', 2: 'R9_R10'}
    for ti in np.unique(transition):
        t_mask = (transition == ti) & mask
        if t_mask.sum() < 2:
            continue
        name = transition_names.get(int(ti), f'transition_{int(ti)}')
        p_sub = pred_original[t_mask]
        t_sub = target_original[t_mask]
        all_valid = np.ones(len(p_sub), dtype=bool)
        m = compute_metrics(p_sub, t_sub, all_valid)
        m['ndcg@50'] = compute_ndcg(p_sub, t_sub, all_valid, k=50)
        per_transition[name] = m

    return {
        'overall': overall,
        'per_transition': per_transition,
        'predictions': predictions,
    }

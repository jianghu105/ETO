import os
import time
import math
import torch
import torch.nn as nn
import logging
from typing import Dict, Optional, List
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from .losses import RefactoredETOLoss, count_nans

logger = logging.getLogger(__name__)


class Trainer:
    """
    Training loop for ETO with real-data monitoring.

    Monitors:
      - NaN / Inf in all output tensors
      - Abundance leakage: |dEPS/d(log_count)|
      - Loss components per epoch
      - Rollout stability
    """

    def __init__(self, model: nn.Module, config: dict, device: str = 'cuda'):
        self.model = model
        self.config = config
        self.device = device
        self.train_cfg = config['training']

        self.model.to(device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.train_cfg.get('lr', 0.001),
            weight_decay=self.train_cfg.get('weight_decay', 0.0001),
        )

        total_steps = self.train_cfg.get('epochs', 100) * 100
        warmup_steps = self.train_cfg.get('warmup_steps', 500)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        self.criterion = RefactoredETOLoss(
            lambda_transition=self.train_cfg.get('lambda_transition', 1.0),
            lambda_population=self.train_cfg.get('lambda_population', 0.5),
            lambda_advantage=self.train_cfg.get('lambda_advantage', 1.0),
            lambda_ea_regression=self.train_cfg.get('lambda_ea_regression', 0.5),
            lambda_rollout=self.train_cfg.get('lambda_rollout', 0.3),
            lambda_smooth=self.train_cfg.get('lambda_smooth', 0.05),
            ranking_margin=self.train_cfg.get('ranking_margin', 0.1),
            ranking_n_pairs=self.train_cfg.get('ranking_n_pairs', 128),
        )

        self.grad_clip = self.train_cfg.get('grad_clip', 1.0)
        self.log_interval = self.train_cfg.get('log_interval', 10)
        self.save_interval = self.train_cfg.get('save_interval', 10)

        self.nan_check = self.train_cfg.get('nan_check', True)
        self.abundance_leakage_check = self.train_cfg.get('abundance_leakage_check', True)

        self.history = []
        self.best_loss = float('inf')
        self.start_epoch = 0

    def _check_nan(self, outputs: Dict, batch_idx: int, epoch: int) -> bool:
        nan_info = count_nans(outputs)
        if nan_info:
            logger.warning(f"Epoch {epoch} batch {batch_idx}: NaN/Inf detected: {nan_info}")
            return True
        return False

    def _check_abundance_leakage(self, batch: Dict, outputs: Dict) -> float:
        """
        Audit: |dEPS / d(log_count_curr)|.
        If EPS truly encodes evolutionary advantage (not abundance),
        this gradient should be near zero.
        """
        eps = outputs.get('eps')
        if eps is None:
            return 0.0
        log_count_curr = batch.get('log_count_curr')
        if log_count_curr is None:
            return 0.0
        grad = torch.autograd.grad(
            eps.sum(), log_count_curr,
            create_graph=False, retain_graph=True,
            allow_unused=True,
        )[0]
        if grad is None:
            return 0.0
        return grad.abs().mean().item()

    def train_epoch(self, dataloader, epoch: int) -> Dict[str, float]:
        self.model.train()
        epoch_losses = {
            'total': 0, 'transition': 0, 'population': 0,
            'advantage': 0, 'ea_regression': 0,
            'rollout': 0, 'smooth': 0,
        }
        n_batches = 0
        nan_count = 0
        leakage_sum = 0.0
        leakage_count = 0

        for batch_idx, batch in enumerate(dataloader):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            self.optimizer.zero_grad()

            outputs = self.model(batch)

            if self.nan_check:
                if self._check_nan(outputs, batch_idx, epoch):
                    nan_count += 1

            losses = self.criterion(outputs, batch)

            if torch.isnan(losses['total']):
                logger.warning(f"Epoch {epoch} batch {batch_idx}: NaN total loss, skipping")
                nan_count += 1
                continue

            losses['total'].backward()

            if self.grad_clip > 0:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            if self.abundance_leakage_check and batch_idx % 50 == 0:
                batch_for_grad = {k: v.clone().requires_grad_(True)
                                  if k == 'log_count_curr' else v
                                  for k, v in batch.items()}
                outputs_for_grad = self.model(batch_for_grad)
                leak = self._check_abundance_leakage(batch_for_grad, outputs_for_grad)
                leakage_sum += leak
                leakage_count += 1

            self.optimizer.step()
            self.scheduler.step()

            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()
            n_batches += 1

            if (batch_idx + 1) % self.log_interval == 0:
                avg = {k: v / max(1, n_batches) for k, v in epoch_losses.items()}
                msg = (
                    f"Epoch {epoch} [{batch_idx+1}/{len(dataloader)}] "
                    f"loss={avg['total']:.4f} trans={avg['transition']:.4f} "
                    f"pop={avg['population']:.4f} adv={avg['advantage']:.4f}"
                )
                if leakage_count > 0:
                    avg_leak = leakage_sum / leakage_count
                    msg += f" leak={avg_leak:.4f}"
                logger.info(msg)

        result = {k: v / max(1, n_batches) for k, v in epoch_losses.items()}
        result['nan_count'] = nan_count
        if leakage_count > 0:
            result['abundance_leakage'] = leakage_sum / leakage_count
        return result

    @torch.no_grad()
    def validate(self, dataloader) -> Dict[str, float]:
        self.model.eval()
        val_losses = {
            'total': 0, 'transition': 0, 'population': 0,
            'advantage': 0, 'ea_regression': 0,
            'rollout': 0, 'smooth': 0,
        }
        n_batches = 0

        for batch in dataloader:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            outputs = self.model(batch)
            losses = self.criterion(outputs, batch)

            for k in val_losses:
                val_losses[k] += losses[k].item()
            n_batches += 1

        return {k: v / max(1, n_batches) for k, v in val_losses.items()}

    def save_checkpoint(self, path: str, epoch: int, is_best: bool = False):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss,
            'history': self.history,
            'config': self.config,
        }
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        torch.save(checkpoint, path)
        if is_best:
            best_path = path.replace('.pt', '_best.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"New best model saved (loss={self.best_loss:.4f})")

    def load_checkpoint(self, path: str):
        if not os.path.exists(path):
            logger.warning(f"Checkpoint not found: {path}")
            return
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_loss = checkpoint['best_loss']
        self.history = checkpoint.get('history', [])
        logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    def train(self, train_loader, val_loader=None, output_dir: str = 'results'):
        epochs = self.train_cfg.get('epochs', 100)
        os.makedirs(output_dir, exist_ok=True)

        logger.info(f"Starting training for {epochs} epochs")
        logger.info(f"Device: {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        for epoch in range(self.start_epoch, epochs):
            t0 = time.time()
            train_losses = self.train_epoch(train_loader, epoch)

            elapsed = time.time() - t0
            log_msg = (
                f"Epoch {epoch}/{epochs-1} ({elapsed:.1f}s) - "
                f"train: total={train_losses['total']:.4f} "
                f"trans={train_losses['transition']:.4f} "
                f"pop={train_losses['population']:.4f} "
                f"adv={train_losses['advantage']:.4f}"
            )
            if 'abundance_leakage' in train_losses:
                log_msg += f" | leakage={train_losses['abundance_leakage']:.4f}"
            if train_losses.get('nan_count', 0) > 0:
                log_msg += f" | NaN batches: {train_losses['nan_count']}"

            if val_loader is not None:
                val_losses = self.validate(val_loader)
                log_msg += (
                    f" | val: total={val_losses['total']:.4f} "
                    f"trans={val_losses['transition']:.4f}"
                )

            logger.info(log_msg)

            self.history.append({
                'epoch': epoch,
                'train': train_losses,
                'val': val_losses if val_loader else None,
            })

            is_best = train_losses['total'] < self.best_loss
            if is_best:
                self.best_loss = train_losses['total']

            ckpt_path = os.path.join(output_dir, 'checkpoint.pt')
            self.save_checkpoint(ckpt_path, epoch, is_best)

            if (epoch + 1) % self.save_interval == 0:
                periodic_path = os.path.join(output_dir, f'checkpoint_epoch{epoch}.pt')
                self.save_checkpoint(periodic_path, epoch)

        logger.info(f"Training complete. Best loss: {self.best_loss:.4f}")
        return self.history

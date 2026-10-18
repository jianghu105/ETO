"""
ETO-Core: Minimal Architecture for Evolutionary Sequence Optimization
=====================================================================

One state representation, one transition operator, one observation decoder.

Three model variants sharing the same data pipeline:
  Model A (StaticBaseline):  sequence → predict target
  Model B (HistoryBaseline): sequence + history → predict target
  Model C (ETO-Core):        sequence + history + context + transition → predict target
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class SequenceEncoder(nn.Module):
    """Multi-kernel 1D CNN sequence encoder. Reused from ETO-Legacy."""

    def __init__(self, seq_length: int = 36, embedding_dim: int = 128,
                 conv_kernels=(3, 5, 7), conv_filters: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.branches = nn.ModuleList()
        for k in conv_kernels:
            pad = k // 2
            self.branches.append(nn.Sequential(
                nn.Conv1d(4, conv_filters, k, padding=pad),
                nn.LayerNorm([conv_filters, seq_length]),
                nn.GELU(),
            ))

        total_filters = conv_filters * len(conv_kernels)
        self.pool_attn = nn.Parameter(torch.randn(1, total_filters, 1) * 0.02)
        self.proj = nn.Sequential(
            nn.Linear(total_filters, embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, embedding_dim),
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, one_hot: torch.Tensor) -> torch.Tensor:
        feats = [branch(one_hot) for branch in self.branches]
        x = torch.cat(feats, dim=1)
        attn = (x * self.pool_attn).sum(dim=1, keepdim=True)
        attn = F.softmax(attn.transpose(1, 2), dim=2).transpose(1, 2)
        x = (x * attn).sum(dim=2)
        x = self.proj(x)
        return self.norm(x)


class HistoryEncoder(nn.Module):
    """Lightweight MLP encoder for history observations."""

    def __init__(self, max_history: int = 5, output_dim: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(max_history + max_history, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, history_counts: torch.Tensor,
                history_mask: torch.Tensor) -> torch.Tensor:
        masked = history_counts * history_mask.float()
        x = torch.cat([masked, history_mask.float()], dim=1)
        return self.net(x)


class ContextEncoder(nn.Module):
    """Encodes selection condition + time as a context embedding."""

    def __init__(self, n_context_types: int = 3, context_embed_dim: int = 8,
                 time_embed_dim: int = 8, output_dim: int = 16,
                 n_fourier_freqs: int = 4):
        super().__init__()
        self.context_embed = nn.Embedding(n_context_types, context_embed_dim)
        self.n_fourier_freqs = n_fourier_freqs
        fourier_dim = n_fourier_freqs * 2
        self.proj = nn.Linear(context_embed_dim + fourier_dim + 1, output_dim)
        self.norm = nn.LayerNorm(output_dim)

    def fourier_time(self, t: torch.Tensor) -> torch.Tensor:
        freqs = torch.arange(1, self.n_fourier_freqs + 1,
                            device=t.device, dtype=torch.float32)
        angles = t.unsqueeze(-1) * freqs.unsqueeze(0) * (2.0 * math.pi / 10.0)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

    def forward(self, transition_type: torch.Tensor,
                delta_t: torch.Tensor) -> torch.Tensor:
        ctx = self.context_embed(transition_type)
        fourier = self.fourier_time(delta_t)
        x = torch.cat([ctx, fourier, delta_t.unsqueeze(-1)], dim=1)
        return self.norm(self.proj(x))


class StateBuilder(nn.Module):
    """Concatenates and projects embeddings into a state vector."""

    def __init__(self, seq_dim: int = 128, history_dim: int = 32,
                 context_dim: int = 16, state_dim: int = 128,
                 dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_dim + history_dim + context_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, state_dim),
            nn.LayerNorm(state_dim),
        )

    def forward(self, seq_emb: torch.Tensor, history_emb: torch.Tensor,
                context_emb: torch.Tensor) -> torch.Tensor:
        x = torch.cat([seq_emb, history_emb, context_emb], dim=1)
        return self.net(x)


class TransitionOperator(nn.Module):
    """
    Conditional transition: z_next = z + MLP(z, context, delta_t).

    Residual structure ensures the operator learns state *changes*,
    not absolute states.
    """

    def __init__(self, state_dim: int = 128, context_dim: int = 16,
                 hidden_dim: int = 256, n_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        layers = []
        in_dim = state_dim + context_dim + 1
        for i in range(n_layers):
            out_dim = hidden_dim if i < n_layers - 1 else state_dim
            layers.extend([
                nn.Linear(in_dim if i == 0 else hidden_dim, out_dim),
                nn.GELU() if i < n_layers - 1 else nn.Identity(),
                nn.Dropout(dropout) if i < n_layers - 1 else nn.Identity(),
            ])
        self.mlp = nn.Sequential(*layers)
        self.state_dim = state_dim

    def forward(self, z: torch.Tensor, context_emb: torch.Tensor,
                delta_t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, context_emb, delta_t.unsqueeze(-1)], dim=1)
        delta_z = self.mlp(x)
        return z + delta_z


class ObservationDecoder(nn.Module):
    """Predicts next-round observable (normalized log-count) from state."""

    def __init__(self, state_dim: int = 128, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state).squeeze(-1)


class ETCoreModel(nn.Module):
    """
    ETO-Core: Full model with sequence, history, context, and transition.

    Model C from the task specification.
    """

    def __init__(self, seq_length: int = 36, state_dim: int = 128,
                 history_dim: int = 32, context_dim: int = 16,
                 max_history: int = 5, dropout: float = 0.1):
        super().__init__()
        self.seq_encoder = SequenceEncoder(
            seq_length=seq_length, embedding_dim=128, dropout=dropout)
        self.history_encoder = HistoryEncoder(
            max_history=max_history, output_dim=history_dim, dropout=dropout)
        self.context_encoder = ContextEncoder(
            output_dim=context_dim)
        self.state_builder = StateBuilder(
            seq_dim=128, history_dim=history_dim, context_dim=context_dim,
            state_dim=state_dim, dropout=dropout)
        self.transition_op = TransitionOperator(
            state_dim=state_dim, context_dim=context_dim, dropout=dropout)
        self.obs_decoder = ObservationDecoder(state_dim=state_dim)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        seq_emb = self.seq_encoder(batch['one_hot'])
        hist_emb = self.history_encoder(batch['history_counts'],
                                         batch['history_mask'])
        ctx_emb = self.context_encoder(batch['transition_type'],
                                        batch['delta_t'])

        z = self.state_builder(seq_emb, hist_emb, ctx_emb)
        z_next = self.transition_op(z, ctx_emb, batch['delta_t'])
        pred = self.obs_decoder(z_next)

        return {
            'prediction': pred,
            'z': z,
            'z_next': z_next,
        }


class StaticBaseline(nn.Module):
    """
    Model A: Sequence-only baseline.

    Uses only sequence features to predict the target.
    No history, no context, no transition.
    """

    def __init__(self, seq_length: int = 36, dropout: float = 0.1):
        super().__init__()
        self.seq_encoder = SequenceEncoder(
            seq_length=seq_length, embedding_dim=128, dropout=dropout)
        self.head = nn.Sequential(
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        seq_emb = self.seq_encoder(batch['one_hot'])
        pred = self.head(seq_emb).squeeze(-1)
        return {'prediction': pred}


class HistoryBaseline(nn.Module):
    """
    Model B: Sequence + history baseline.

    Uses sequence and history but no explicit transition operator.
    Tests whether history provides value beyond static sequence features.
    """

    def __init__(self, seq_length: int = 36, history_dim: int = 32,
                 context_dim: int = 16, max_history: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        self.seq_encoder = SequenceEncoder(
            seq_length=seq_length, embedding_dim=128, dropout=dropout)
        self.history_encoder = HistoryEncoder(
            max_history=max_history, output_dim=history_dim, dropout=dropout)
        self.context_encoder = ContextEncoder(output_dim=context_dim)
        self.head = nn.Sequential(
            nn.Linear(128 + history_dim + context_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        seq_emb = self.seq_encoder(batch['one_hot'])
        hist_emb = self.history_encoder(batch['history_counts'],
                                         batch['history_mask'])
        ctx_emb = self.context_encoder(batch['transition_type'],
                                        batch['delta_t'])
        x = torch.cat([seq_emb, hist_emb, ctx_emb], dim=1)
        pred = self.head(x).squeeze(-1)
        return {'prediction': pred}


def build_model(model_type: str, **kwargs) -> nn.Module:
    """Factory for building model variants."""
    models = {
        'static': StaticBaseline,
        'history': HistoryBaseline,
        'etocore': ETCoreModel,
    }
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(models.keys())}")
    return models[model_type](**kwargs)

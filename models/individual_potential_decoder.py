import torch
import torch.nn as nn
from typing import Dict


class IndividualPotentialDecoder(nn.Module):
    """
    Individual-level potential decoder.

    Instead of mapping Z → N scores (which depends on fixed population size),
    this decoder computes EPS for each individual independently:

        EPS_i = f(Z_future, x_i)

    where:
        Z_future: population latent (evolutionary environment)
        x_i: individual dynamic state (seq_emb + velocity + structure)

    This is set-based: works for any number of individuals.

    Architecture:
        Input: concat(Z_future, seq_emb, velocity, [structure_emb])
        Dimension: 256 + 128 + 1 + [16] = 385 or 401
        MLP → 1 scalar EPS_i
    """

    def __init__(self, population_dim: int = 256,
                 seq_embedding_dim: int = 128,
                 velocity_dim: int = 1,
                 structure_dim: int = 0,
                 hidden_dim: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        self.population_dim = population_dim
        self.seq_embedding_dim = seq_embedding_dim
        self.velocity_dim = velocity_dim
        self.structure_dim = structure_dim

        input_dim = population_dim + seq_embedding_dim + velocity_dim + structure_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, z_future: torch.Tensor,
                seq_emb: torch.Tensor,
                velocity: torch.Tensor,
                structure_emb: torch.Tensor = None) -> torch.Tensor:
        """
        Compute EPS for each individual.

        Args:
            z_future: (B, population_dim) - future population state
            seq_emb: (B, seq_embedding_dim) - individual sequence embedding
            velocity: (B, velocity_dim) or (B,) - individual velocity
            structure_emb: (B, structure_dim) or None

        Returns:
            eps: (B,) - evolutionary potential score for each individual
        """
        B = z_future.shape[0]

        if velocity.dim() == 1:
            velocity = velocity.unsqueeze(-1)

        parts = [z_future, seq_emb, velocity]
        if structure_emb is not None and self.structure_dim > 0:
            parts.append(structure_emb)

        combined = torch.cat(parts, dim=-1)
        eps = self.mlp(combined).squeeze(-1)
        eps = torch.nan_to_num(eps, nan=0.0, posinf=5.0, neginf=-5.0)
        return eps

    def forward_batch(self, z_future: torch.Tensor,
                      individuals: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Batch-mode: compute EPS for N individuals sharing the same Z_future.

        Args:
            z_future: (B, population_dim) - shared population state
            individuals: dict with keys:
                'seq_emb': (B*N, seq_embedding_dim) or (B, N, seq_embedding_dim)
                'velocity': (B*N, velocity_dim) or (B, N, velocity_dim)
                'structure_emb': (B*N, structure_dim) or None

        Returns:
            eps: (B*N,) or (B, N) - EPS for each individual
        """
        seq_emb = individuals['seq_emb']
        velocity = individuals['velocity']
        structure_emb = individuals.get('structure_emb', None)

        is_nested = seq_emb.dim() == 3
        if is_nested:
            B, N, D = seq_emb.shape
            z_expanded = z_future.unsqueeze(1).expand(-1, N, -1)
            z_flat = z_expanded.reshape(B * N, -1)
            seq_flat = seq_emb.reshape(B * N, -1)
            vel_flat = velocity.reshape(B * N, -1)
            struct_flat = structure_emb.reshape(B * N, -1) if structure_emb is not None else None
        else:
            z_flat = z_future
            seq_flat = seq_emb
            vel_flat = velocity
            struct_flat = structure_emb

        eps_flat = self.forward(z_flat, seq_flat, vel_flat, struct_flat)

        if is_nested:
            eps_flat = eps_flat.view(B, N)

        return eps_flat


class SetPopulationDecoder(nn.Module):
    """
    Set-based population decoder for reconstruction.

    Instead of outputting fixed N values, this decoder:
      1. Generates a population-level summary from Z
      2. Can reconstruct per-round statistics (mean, variance)
      3. Does not depend on specific population size

    Output: per-round population statistics (mean_log_count, log_variance)
    """

    def __init__(self, population_dim: int = 256, hidden_dim: int = 256,
                 n_rounds: int = 5):
        super().__init__()
        self.population_dim = population_dim
        self.n_rounds = n_rounds

        self.stats_decoder = nn.Sequential(
            nn.Linear(population_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_rounds * 2),
        )

    def forward(self, z_future: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_future: (B, population_dim)

        Returns:
            stats: (B, n_rounds * 2) - [mean_log_count_R1, log_var_R1, ...]
        """
        stats = self.stats_decoder(z_future)
        return stats

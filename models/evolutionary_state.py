import torch
import torch.nn as nn
from typing import Tuple, Dict


class IndividualStateBuilder(nn.Module):
    """
    Legacy state builder - kept for backward compatibility.
    Builds combined state: [seq_emb, log_count, velocity]
    """
    def __init__(self, seq_embedding_dim: int = 128):
        super().__init__()
        self.seq_embedding_dim = seq_embedding_dim
        self.state_dim = seq_embedding_dim + 2  # + log_count + velocity

    def forward(self, seq_emb: torch.Tensor, log_count: torch.Tensor,
                velocity: torch.Tensor) -> torch.Tensor:
        if log_count.dim() == 1:
            log_count = log_count.unsqueeze(-1)
        if velocity.dim() == 1:
            velocity = velocity.unsqueeze(-1)

        x = torch.cat([seq_emb, log_count, velocity], dim=-1)
        return x


class AbundanceFreeStateBuilder(nn.Module):
    """
    Refactored state builder with abundance-free pathway.

    Splits individual state into two branches:
      - dynamic_state: [seq_emb, velocity, structure_emb] (abundance-independent)
      - observation_state: [log_count] (observation-only)

    The population encoder primarily uses dynamic_state,
    so Z_t represents evolutionary dynamics, not abundance snapshot.
    """

    def __init__(self, seq_embedding_dim: int = 128,
                 velocity_dim: int = 1,
                 structure_dim: int = 0):
        super().__init__()
        self.seq_embedding_dim = seq_embedding_dim
        self.velocity_dim = velocity_dim
        self.structure_dim = structure_dim

        self.dynamic_dim = seq_embedding_dim + velocity_dim + structure_dim
        self.observation_dim = 1  # log_count only

    def forward(self, seq_emb: torch.Tensor,
                velocity: torch.Tensor,
                log_count: torch.Tensor,
                structure_emb: torch.Tensor = None
                ) -> Dict[str, torch.Tensor]:
        """
        Args:
            seq_emb: (B, seq_embedding_dim)
            velocity: (B, velocity_dim) or (B,)
            log_count: (B,) or (B, 1)
            structure_emb: (B, structure_dim) or None

        Returns:
            dict with:
              'dynamic_state': (B, dynamic_dim)
              'observation_state': (B, 1)
              'full_state': (B, dynamic_dim + 1) - for compatibility
        """
        if velocity.dim() == 1:
            velocity = velocity.unsqueeze(-1)
        if log_count.dim() == 1:
            log_count = log_count.unsqueeze(-1)

        parts = [seq_emb, velocity]
        if structure_emb is not None and self.structure_dim > 0:
            parts.append(structure_emb)

        dynamic_state = torch.cat(parts, dim=-1)
        observation_state = log_count
        full_state = torch.cat([dynamic_state, observation_state], dim=-1)

        return {
            'dynamic_state': dynamic_state,
            'observation_state': observation_state,
            'full_state': full_state,
        }

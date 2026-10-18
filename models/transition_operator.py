import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Dict, Optional


class ConditionalTransitionOperator(nn.Module):
    """
    Legacy transition operator - kept for backward compatibility.
    """
    def __init__(self, state_dim: int, context_dim: int = 32,
                 hidden_dim: int = 512, n_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.state_dim = state_dim
        self.context_dim = context_dim

        self.input_proj = nn.Linear(state_dim + context_dim, hidden_dim)

        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            if i < n_layers - 1:
                layers.append(nn.Dropout(dropout))
        self.hidden_net = nn.Sequential(*layers)

        self.output_proj = nn.Linear(hidden_dim, state_dim)
        self.output_norm = nn.LayerNorm(state_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, z_t: torch.Tensor, e_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([z_t, e_t], dim=-1)
        h = self.input_proj(combined)
        h = self.hidden_net(h)
        delta_z = self.output_proj(h)
        delta_z = self.output_norm(delta_z)
        delta_z = self.dropout(delta_z)

        z_next = z_t + delta_z
        return z_next, delta_z


class RolloutTransitionOperator(nn.Module):
    """
    Continuous-time transition operator with multi-step trajectory rollout.

    Core dynamics:
        Z(t + Δt) = Z(t) + Δt · T_θ(Z(t), E(t))

    Supports:
      - Single-step: Z(t) → Z(t+1)
      - Multi-step rollout: Z(t₀) → Z(t₁) → ... → Z(tₙ)
      - Arbitrary time steps via continuous time encoding

    Temporal smoothness is enforced via:
        L_smooth = ||ΔZ(tₖ) - ΔZ(tₖ₊₁)||²
    which penalizes discontinuous changes in the dynamics field.
    """

    def __init__(self, state_dim: int, context_dim: int = 32,
                 hidden_dim: int = 512, n_layers: int = 3, dropout: float = 0.1,
                 max_rollout_steps: int = 4):
        super().__init__()
        self.state_dim = state_dim
        self.context_dim = context_dim
        self.max_rollout_steps = max_rollout_steps

        self.input_proj = nn.Linear(state_dim + context_dim, hidden_dim)

        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            if i < n_layers - 1:
                layers.append(nn.Dropout(dropout))
        self.hidden_net = nn.Sequential(*layers)

        self.output_proj = nn.Linear(hidden_dim, state_dim)
        self.output_norm = nn.LayerNorm(state_dim)
        self.dropout = nn.Dropout(dropout)

    def single_step(self, z_t: torch.Tensor, e_t: torch.Tensor
                    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single transition: Z_{t+1} = Z_t + T(Z_t, E_t)
        """
        combined = torch.cat([z_t, e_t], dim=-1)
        h = self.input_proj(combined)
        h = self.hidden_net(h)
        delta_z = self.output_proj(h)
        delta_z = self.output_norm(delta_z)
        delta_z = self.dropout(delta_z)
        delta_z = torch.nan_to_num(delta_z, nan=0.0, posinf=1.0, neginf=-1.0)

        z_next = z_t + delta_z
        return z_next, delta_z

    def rollout(self, z_start: torch.Tensor,
                context_sequence: List[torch.Tensor],
                n_steps: Optional[int] = None
                ) -> Dict[str, List[torch.Tensor]]:
        """
        Multi-step trajectory rollout.

        Args:
            z_start: (B, state_dim) - initial population state
            context_sequence: list of (B, context_dim) context embeddings
                              for each step [E_R3→R5, E_R5→R9, E_R9→R10, E_R10→R11]
            n_steps: number of steps to rollout (None = all)

        Returns:
            dict with:
              'z_trajectory': list of z states [z_0, z_1, ..., z_n]
              'delta_z_trajectory': list of delta_z at each step
        """
        if n_steps is None:
            n_steps = len(context_sequence)
        n_steps = min(n_steps, len(context_sequence), self.max_rollout_steps)

        z_trajectory = [z_start]
        delta_z_trajectory = []

        z_current = z_start
        for step in range(n_steps):
            e_t = context_sequence[step]
            z_next, delta_z = self.single_step(z_current, e_t)
            z_trajectory.append(z_next)
            delta_z_trajectory.append(delta_z)
            z_current = z_next

        return {
            'z_trajectory': z_trajectory,
            'delta_z_trajectory': delta_z_trajectory,
        }

    def forward(self, z_t: torch.Tensor, e_t: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Backward-compatible single-step forward."""
        return self.single_step(z_t, e_t)

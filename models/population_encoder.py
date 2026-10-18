import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict


class EvolutionaryAttentionPool(nn.Module):
    """
    Legacy attention pool - kept for backward compatibility.
    """
    def __init__(self, state_dim: int, population_dim: int = 256,
                 n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.state_dim = state_dim
        self.population_dim = population_dim
        self.n_heads = n_heads

        self.query_proj = nn.Linear(state_dim, population_dim)
        self.key_proj = nn.Linear(state_dim, population_dim)
        self.value_proj = nn.Linear(state_dim, population_dim)

        self.context_vector = nn.Parameter(torch.randn(n_heads, population_dim // n_heads))
        nn.init.xavier_uniform_(self.context_vector)

        self.output_proj = nn.Linear(population_dim, population_dim)
        self.norm = nn.LayerNorm(population_dim)
        self.dropout = nn.Dropout(dropout)

        self.global_token = nn.Parameter(torch.randn(1, 1, state_dim) * 0.02)

    def forward(self, states: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        B, N, D = states.shape

        global_tok = self.global_token.expand(B, -1, -1)
        states_aug = torch.cat([global_tok, states], dim=1)

        Q = self.query_proj(states_aug)
        K = self.key_proj(states)
        V = self.value_proj(states)

        head_dim = self.population_dim // self.n_heads
        Q = Q.view(B, -1, self.n_heads, head_dim).transpose(1, 2)
        K = K.view(B, -1, self.n_heads, head_dim).transpose(1, 2)
        V = V.view(B, -1, self.n_heads, head_dim).transpose(1, 2)

        scale = math.sqrt(head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale

        if mask is not None:
            mask_aug = F.pad(mask, (1, 0), value=True)
            mask_aug = mask_aug.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(~mask_aug, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, N + 1, self.population_dim)

        z_t = context[:, 0, :]
        z_t = self.output_proj(z_t)
        z_t = self.norm(z_t)
        return z_t


class PopulationEncoder(nn.Module):
    """
    Legacy population encoder - kept for backward compatibility.
    """
    def __init__(self, state_dim: int, population_dim: int = 256,
                 n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention_pool = EvolutionaryAttentionPool(
            state_dim, population_dim, n_heads, dropout
        )

    def forward(self, states: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        return self.attention_pool(states, mask)


class AbundanceFreePopulationEncoder(nn.Module):
    """
    Population encoder that primarily uses dynamic_state (abundance-free).

    Z_t represents evolutionary dynamics, not abundance snapshot.

    Architecture:
      - Primary: attention pool over dynamic_states (seq_emb + velocity + structure)
      - Auxiliary: observation encoder for log_count (optional, gated fusion)

    The gate controls how much observation information flows into Z_t.
    During training, the gate can be regularized to stay small,
    forcing Z_t to rely on dynamics rather than abundance.
    """

    def __init__(self, dynamic_dim: int, population_dim: int = 256,
                 observation_dim: int = 1, n_heads: int = 8,
                 dropout: float = 0.1, use_observation: bool = True):
        super().__init__()
        self.population_dim = population_dim
        self.use_observation = use_observation

        self.dynamic_pool = EvolutionaryAttentionPool(
            dynamic_dim, population_dim, n_heads, dropout
        )

        if use_observation:
            self.observation_encoder = nn.Sequential(
                nn.Linear(observation_dim, 64),
                nn.GELU(),
                nn.Linear(64, population_dim),
            )
            self.gate = nn.Sequential(
                nn.Linear(population_dim * 2, population_dim),
                nn.Sigmoid(),
            )
            self.output_proj = nn.Linear(population_dim, population_dim)
            self.output_norm = nn.LayerNorm(population_dim)

    def forward(self, dynamic_states: torch.Tensor,
                observation_states: torch.Tensor = None,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            dynamic_states: (B, N, dynamic_dim) - abundance-free states
            observation_states: (B, N, observation_dim) - log_count
            mask: (B, N) - padding mask

        Returns:
            Z_t: (B, population_dim)
        """
        z_dynamic = self.dynamic_pool(dynamic_states, mask)

        if not self.use_observation or observation_states is None:
            return z_dynamic

        B, N, _ = observation_states.shape
        obs_mean = observation_states.mean(dim=1)
        z_obs = self.observation_encoder(obs_mean)

        gate_weight = self.gate(torch.cat([z_dynamic, z_obs], dim=-1))
        z_t = z_dynamic + gate_weight * z_obs
        z_t = self.output_proj(z_t)
        z_t = self.output_norm(z_t)

        return z_t

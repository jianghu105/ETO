import math
import torch
import torch.nn as nn
import torch.nn.functional as F


CONTEXT_TYPES = {
    "R3_R5": 0,      # normal
    "R5_R9": 1,      # counter_selection
    "R9_R10": 2,     # recovery
    "R10_R11": 3,    # normal
}

ROUND_INDICES = {
    "R3_R5": 0,
    "R5_R9": 1,
    "R9_R10": 2,
    "R10_R11": 3,
}

ROUND_TIME_MAP = {
    "R3_R5": 0.0,
    "R5_R9": 1.0,
    "R9_R10": 2.0,
    "R10_R11": 3.0,
}


class ContextEncoder(nn.Module):
    """
    Legacy context encoder - kept for backward compatibility.
    """
    def __init__(self, n_context_types: int = 4, context_dim: int = 32):
        super().__init__()
        self.context_dim = context_dim
        self.embedding = nn.Embedding(n_context_types, context_dim)
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, context_dim),
            nn.GELU(),
            nn.LayerNorm(context_dim),
        )

    def forward(self, context_id: torch.Tensor) -> torch.Tensor:
        e = self.embedding(context_id)
        e = self.mlp(e)
        return e


class ContinuousTimeEncoder(nn.Module):
    """
    Fourier feature encoding for continuous evolutionary time.

    Maps scalar t → MLP([sin(ω₁t), cos(ω₁t), ..., sin(ωₖt), cos(ωₖt)])

    This encoding ensures that:
      1. Nearby time points produce similar embeddings (smooth interpolation)
      2. Unseen future times can be handled (extrapolation beyond training range)
      3. No NaN from out-of-vocabulary discrete indices

    In the ETO framework, t represents the evolutionary stage:
        t=0: R3→R5 (initial positive selection)
        t=1: R5→R9 (counter-selection)
        t=2: R9→R10 (recovery)
        t=3: R10→R11 (final enrichment)

    The Fourier basis allows the model to learn continuous dynamics
    Z(t+Δt) = Z(t) + Δt·T(Z(t), E(t)) rather than memorizing discrete transitions.
    """

    def __init__(self, n_frequencies: int = 8, output_dim: int = 16):
        super().__init__()
        self.n_frequencies = n_frequencies
        self.output_dim = output_dim

        freqs = torch.arange(1, n_frequencies + 1, dtype=torch.float32)
        self.register_buffer('freqs', freqs)

        self.proj = nn.Linear(2 * n_frequencies, output_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (B,) float tensor — continuous time values (e.g. 0.0, 1.0, 2.0, 3.0)

        Returns:
            (B, output_dim) Fourier time embedding
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        t = t.float().unsqueeze(-1)

        angles = self.freqs.unsqueeze(0) * t * (2.0 * math.pi / 4.0)
        features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        return self.proj(features)


class HybridContextEncoder(nn.Module):
    """
    Hybrid context encoder: context_type + continuous_time.

    Uses Fourier feature encoding for time instead of discrete round embedding.
    This allows the model to:
      - Interpolate between known rounds (e.g., t=1.5 between R5→R9 and R9→R10)
      - Extrapolate to unseen future rounds (e.g., R11→R12 at t=4.0)
      - Avoid NaN from unseen discrete round indices

    E_t = MLP(context_embedding(type), fourier_embedding(t))
    """

    def __init__(self, n_context_types: int = 4, n_rounds: int = 10,
                 context_embed_dim: int = 16, round_embed_dim: int = 16,
                 output_dim: int = 32, n_fourier_freqs: int = 8,
                 use_continuous_time: bool = True):
        super().__init__()
        self.context_embed_dim = context_embed_dim
        self.output_dim = output_dim
        self.use_continuous_time = use_continuous_time

        self.context_embedding = nn.Embedding(n_context_types, context_embed_dim)

        if use_continuous_time:
            self.time_encoder = ContinuousTimeEncoder(
                n_frequencies=n_fourier_freqs,
                output_dim=round_embed_dim,
            )
        else:
            self.round_embedding = nn.Embedding(n_rounds, round_embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(context_embed_dim + round_embed_dim, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, context_id: torch.Tensor,
                round_idx: torch.Tensor = None,
                continuous_time: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            context_id: (B,) LongTensor — selection type (0-3)
            round_idx: (B,) LongTensor — discrete round index (fallback)
            continuous_time: (B,) FloatTensor — continuous time values

        Returns:
            E_t: (B, output_dim)
        """
        ctx_emb = self.context_embedding(context_id)

        if self.use_continuous_time:
            if continuous_time is not None:
                t = continuous_time
            elif round_idx is not None:
                t = round_idx.float()
            else:
                t = context_id.float()
            time_emb = self.time_encoder(t)
        else:
            if round_idx is None:
                round_idx = context_id.clone()
            time_emb = self.round_embedding(round_idx)

        combined = torch.cat([ctx_emb, time_emb], dim=-1)
        e_t = self.mlp(combined)
        return e_t

import torch
import torch.nn as nn
import torch.nn.functional as F


class PopulationDecoder(nn.Module):
    def __init__(self, population_dim: int = 256, hidden_dim: int = 256,
                 n_rounds: int = 5):
        super().__init__()
        self.population_dim = population_dim
        self.n_rounds = n_rounds

        self.mlp = nn.Sequential(
            nn.Linear(population_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_rounds),
        )

    def forward(self, z_future: torch.Tensor) -> torch.Tensor:
        # z_future: (B, population_dim)
        # output: (B, n_rounds) - predicted log abundance distribution
        logits = self.mlp(z_future)
        return logits


class PotentialDecoder(nn.Module):
    def __init__(self, seq_embedding_dim: int = 128, population_dim: int = 256,
                 hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(seq_embedding_dim + population_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, seq_emb: torch.Tensor,
                z_future: torch.Tensor) -> torch.Tensor:
        # seq_emb: (B, seq_embedding_dim)
        # z_future: (B, population_dim)
        combined = torch.cat([seq_emb, z_future], dim=-1)
        phi = self.mlp(combined)  # (B, 1)
        return phi.squeeze(-1)  # (B,)

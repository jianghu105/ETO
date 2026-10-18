import torch
import torch.nn as nn
from typing import Optional


class StructureEncoder(nn.Module):
    """
    Optional structure feature encoder.

    Inputs (all optional, zero-filled if absent):
      - gc_content: scalar in [0, 1]
      - mfe: minimum free energy (normalized scalar)
      - paired_prob: mean pairing probability (scalar in [0, 1])

    Output: 16-dim structure embedding.

    When no structure features are available, returns zeros.
    """

    def __init__(self, structure_dim: int = 16, dropout: float = 0.1):
        super().__init__()
        self.structure_dim = structure_dim
        self.input_dim = 3  # gc_content, mfe, paired_prob

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, structure_dim),
            nn.LayerNorm(structure_dim),
        )

    def forward(self, structure_features: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            structure_features: (B, 3) tensor [gc_content, mfe, paired_prob]
                                or None (returns zeros)

        Returns:
            structure_emb: (B, structure_dim)
        """
        if structure_features is None:
            return None

        B = structure_features.shape[0]
        device = structure_features.device

        has_structure = structure_features.abs().sum() > 0

        if not has_structure:
            return torch.zeros(B, self.structure_dim, device=device)

        return self.encoder(structure_features)

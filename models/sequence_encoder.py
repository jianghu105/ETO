import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 seq_length: int = 36):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=padding)
        self.norm = nn.LayerNorm([out_channels, seq_length])
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        return x


class AttentionPool1d(nn.Module):
    def __init__(self, channels: int, seq_length: int = 36):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(channels, channels // 2),
            nn.GELU(),
            nn.Linear(channels // 2, 1),
        )
        self.seq_length = seq_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, L)
        weights = self.attn(x.transpose(1, 2))  # (B, L, 1)
        weights = F.softmax(weights, dim=1)
        pooled = (x * weights.transpose(1, 2)).sum(dim=-1)  # (B, C)
        return pooled


class SequenceEncoder(nn.Module):
    def __init__(self, seq_length: int = 36, embedding_dim: int = 128,
                 conv_kernels: list = [3, 5, 7], conv_filters: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.seq_length = seq_length
        self.embedding_dim = embedding_dim

        self.conv_blocks = nn.ModuleList()
        for k in conv_kernels:
            self.conv_blocks.append(
                ConvBlock(4, conv_filters, k, seq_length)
            )

        total_filters = conv_filters * len(conv_kernels)
        self.projection = nn.Linear(total_filters, embedding_dim)
        self.pool = AttentionPool1d(total_filters, seq_length)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, L)
        conv_outputs = []
        for block in self.conv_blocks:
            conv_outputs.append(block(x))

        merged = torch.cat(conv_outputs, dim=1)  # (B, total_filters, L)
        pooled = self.pool(merged)  # (B, total_filters)
        projected = self.projection(pooled)  # (B, embedding_dim)
        projected = self.norm(projected)
        projected = self.dropout(projected)
        return projected

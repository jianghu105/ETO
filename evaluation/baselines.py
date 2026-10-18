"""
Baseline methods for ETO comparison.

Baseline 1: Current abundance ranking
  score = log1p(count)

Baseline 2: Sequence-only CNN
  CNN(one_hot) -> score

Baseline 3: Sequence + abundance temporal model
  LSTM/Transformer over (sequence, historical_abundance) -> score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional

from evaluation.metrics import unified_evaluation


class AbundanceRankingBaseline:
    """
    Baseline 1: Score = current abundance (log count).

    Tests whether simply ranking by current count is sufficient
    to predict evolutionary advantage.
    """

    def __init__(self):
        self.name = "abundance_ranking"

    def predict(self, log_counts: np.ndarray) -> np.ndarray:
        return log_counts

    def evaluate(self, log_counts: np.ndarray,
                 evolutionary_advantage: np.ndarray,
                 current_abundance: np.ndarray = None,
                 ea_valid: np.ndarray = None,
                 pair_indices: np.ndarray = None) -> Dict[str, float]:
        scores = self.predict(log_counts)
        return unified_evaluation(scores, evolutionary_advantage, current_abundance,
                                  ea_valid=ea_valid, pair_indices=pair_indices)


class SequenceOnlyCNN(nn.Module):
    """
    Baseline 2: CNN(sequence) -> score.

    Uses the same sequence encoder as ETO but predicts
    evolutionary advantage directly from sequence, without
    population context or temporal dynamics.
    """

    def __init__(self, seq_length: int = 36, embedding_dim: int = 128):
        super().__init__()
        self.name = "seq_only_cnn"

        self.conv1 = nn.Conv1d(4, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, 1),
        )

    def forward(self, one_hot: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(one_hot))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.bn(x)
        x = self.pool(x).squeeze(-1)
        x = self.fc(x)
        return x.squeeze(-1)


class TemporalSequenceModel(nn.Module):
    """
    Baseline 3: Sequence + abundance temporal model.

    LSTM over (sequence_embedding, historical_abundance) -> score.
    Uses the full round trajectory as input.
    """

    def __init__(self, seq_length: int = 36, n_rounds: int = 5,
                 hidden_dim: int = 128):
        super().__init__()
        self.name = "temporal_sequence"
        self.n_rounds = n_rounds

        self.conv1 = nn.Conv1d(4, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.seq_proj = nn.Linear(64, hidden_dim)

        self.lstm = nn.LSTM(
            input_size=hidden_dim + 1,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, one_hot: torch.Tensor,
                log_count_trajectory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            one_hot: (B, 4, L)
            log_count_trajectory: (B, n_rounds) - log1p counts at each round
        """
        x = F.relu(self.conv1(one_hot))
        x = F.relu(self.conv2(x))
        seq_emb = self.pool(x).squeeze(-1)
        seq_emb = self.seq_proj(seq_emb)

        B = one_hot.shape[0]
        T = log_count_trajectory.shape[1]

        seq_expanded = seq_emb.unsqueeze(1).expand(-1, T, -1)
        counts_expanded = log_count_trajectory.unsqueeze(-1)
        temporal_input = torch.cat([seq_expanded, counts_expanded], dim=-1)

        lstm_out, _ = self.lstm(temporal_input)
        last_hidden = lstm_out[:, -1, :]
        score = self.head(last_hidden).squeeze(-1)
        return score


def train_baseline_cnn(model: nn.Module,
                       samples: List[Dict],
                       config: dict,
                       device: str = 'cuda',
                       epochs: int = 30) -> nn.Module:
    """Train sequence-only CNN baseline."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    one_hots = torch.stack([s['one_hot'] for s in samples])
    ea_targets = torch.stack([s['evolutionary_advantage'] for s in samples])

    dataset = torch.utils.data.TensorDataset(one_hots, ea_targets)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config['training']['batch_size'],
        shuffle=True, num_workers=0,
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        for batch_oh, batch_ea in loader:
            batch_oh = batch_oh.to(device)
            batch_ea = batch_ea.to(device)

            optimizer.zero_grad()
            pred = model(batch_oh)
            loss = F.mse_loss(pred, batch_ea)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / max(1, n_batches)
            print(f"  [SeqOnlyCNN] Epoch {epoch+1}/{epochs} loss={avg_loss:.6f}")

    return model


def train_baseline_temporal(model: nn.Module,
                            samples: List[Dict],
                            features: Dict,
                            config: dict,
                            device: str = 'cuda',
                            epochs: int = 30) -> nn.Module:
    """Train sequence + abundance temporal model baseline."""
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    from data.preprocessing import ROUND_NAMES

    n = len(samples)
    log_count_traj = np.zeros((n, len(ROUND_NAMES)), dtype=np.float32)
    for r_idx, r in enumerate(ROUND_NAMES):
        for i in range(n):
            pair_idx = samples[i]['pair_idx'].item()
            from data.preprocessing import ROUND_PAIRS
            r_prev, r_curr = ROUND_PAIRS[pair_idx]
            log_count_traj[i, r_idx] = features['log_counts'][r][samples[i]['seq_idx'].item()]

    one_hots = torch.stack([s['one_hot'] for s in samples])
    log_traj = torch.from_numpy(log_count_traj)
    ea_targets = torch.stack([s['evolutionary_advantage'] for s in samples])

    dataset = torch.utils.data.TensorDataset(one_hots, log_traj, ea_targets)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=config['training']['batch_size'],
        shuffle=True, num_workers=0,
    )

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        n_batches = 0
        for batch_oh, batch_traj, batch_ea in loader:
            batch_oh = batch_oh.to(device)
            batch_traj = batch_traj.to(device)
            batch_ea = batch_ea.to(device)

            optimizer.zero_grad()
            pred = model(batch_oh, batch_traj)
            loss = F.mse_loss(pred, batch_ea)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / max(1, n_batches)
            print(f"  [Temporal] Epoch {epoch+1}/{epochs} loss={avg_loss:.6f}")

    return model


def evaluate_baselines(samples: List[Dict],
                       features: Dict,
                       config: dict,
                       device: str = 'cuda') -> Dict[str, Dict[str, float]]:
    """
    Run all baselines and return unified metrics.

    Returns:
        {baseline_name: {metric_name: value}}
    """
    from data.preprocessing import ROUND_NAMES

    results = {}

    n = len(arrays['pair_idx'])
    ea_all = arrays['evolutionary_advantage']
    log_count_curr = arrays['log_count_curr']
    rel_abund = arrays['relative_abundance_curr']
    ea_valid = arrays.get('ea_valid', np.ones(n, dtype=bool))
    pair_indices = arrays['pair_idx']

    print("=" * 60)
    print("Baseline 1: Current Abundance Ranking")
    print("=" * 60)
    bl1 = AbundanceRankingBaseline()
    results['abundance_ranking'] = bl1.evaluate(log_count_curr, ea_all, rel_abund,
                                                 ea_valid=ea_valid, pair_indices=pair_indices)
    for k, v in results['abundance_ranking'].items():
        print(f"  {k}: {v:.4f}")

    print("\n" + "=" * 60)
    print("Baseline 2: Sequence-only CNN")
    print("=" * 60)
    seq_cfg = config.get('model', {}).get('sequence_encoder', {})
    cnn_model = SequenceOnlyCNN(
        seq_length=config['data']['seq_length'],
        embedding_dim=seq_cfg.get('embedding_dim', 128),
    )
    cnn_model = train_baseline_cnn(cnn_model, samples, config, device, epochs=30)

    cnn_model.eval()
    cnn_scores = []
    bs = config['training']['batch_size']
    with torch.no_grad():
        for i in range(0, n, bs):
            batch_oh = torch.stack([s['one_hot'] for s in samples[i:i+bs]]).to(device)
            cnn_scores.append(cnn_model(batch_oh).cpu().numpy())
    cnn_scores = np.concatenate(cnn_scores)
    results['seq_only_cnn'] = unified_evaluation(cnn_scores, ea_all, rel_abund,
                                                   ea_valid=ea_valid, pair_indices=pair_indices)
    for k, v in results['seq_only_cnn'].items():
        print(f"  {k}: {v:.4f}")

    print("\n" + "=" * 60)
    print("Baseline 3: Sequence + Abundance Temporal Model")
    print("=" * 60)
    temporal_model = TemporalSequenceModel(
        seq_length=config['data']['seq_length'],
        n_rounds=len(ROUND_NAMES),
    )
    temporal_model = train_baseline_temporal(
        temporal_model, samples, features, config, device, epochs=30
    )

    temporal_model.eval()
    temporal_scores = []
    with torch.no_grad():
        log_traj = np.zeros((n, len(ROUND_NAMES)), dtype=np.float32)
        for r_idx, r in enumerate(ROUND_NAMES):
            for i in range(n):
                pair_idx = samples[i]['pair_idx'].item()
                from data.preprocessing import ROUND_PAIRS
                log_traj[i, r_idx] = features['log_counts'][r][samples[i]['seq_idx'].item()]
        for i in range(0, n, bs):
            batch_oh = torch.stack([s['one_hot'] for s in samples[i:i+bs]]).to(device)
            batch_traj = torch.from_numpy(log_traj[i:i+bs]).to(device)
            temporal_scores.append(temporal_model(batch_oh, batch_traj).cpu().numpy())
    temporal_scores = np.concatenate(temporal_scores)
    results['temporal_sequence'] = unified_evaluation(temporal_scores, ea_all, rel_abund,
                                                        ea_valid=ea_valid, pair_indices=pair_indices)
    for k, v in results['temporal_sequence'].items():
        print(f"  {k}: {v:.4f}")

    return results

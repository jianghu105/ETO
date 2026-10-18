import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class TransitionLoss(nn.Module):
    """
    MSE loss on population state transition.
    L_trans = MSE(Z_pred, Z_true)
    """
    def __init__(self):
        super().__init__()

    def forward(self, z_next_pred: torch.Tensor,
                z_t_target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(z_next_pred, z_t_target.detach())


class PopulationReconstructionLoss(nn.Module):
    """
    KL divergence loss for population reconstruction.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pop_pred: torch.Tensor,
                log_count_curr: torch.Tensor,
                log_count_prev: torch.Tensor) -> torch.Tensor:
        target = torch.stack([log_count_prev, log_count_curr], dim=-1)
        target_soft = F.softmax(target, dim=-1)
        pred_soft = F.softmax(pop_pred[:, :2], dim=-1)

        kl = F.kl_div(
            pred_soft.log().clamp(min=-20),
            target_soft,
            reduction='batchmean'
        )
        return kl


class EARegressionLoss(nn.Module):
    """
    Huber regression loss on EA values.

    Directly regresses EPS toward the observational EA target,
    complementing the ranking loss with absolute calibration.
    Only computed on samples with valid EA labels.
    """

    def __init__(self, huber_delta: float = 1.0):
        super().__init__()
        self.huber_delta = huber_delta

    def forward(self, eps_pred: torch.Tensor,
                evolutionary_advantage: torch.Tensor,
                ea_valid: torch.Tensor = None) -> torch.Tensor:
        if ea_valid is not None:
            valid_idx = ea_valid.nonzero(as_tuple=True)[0]
            if valid_idx.shape[0] == 0:
                return torch.tensor(0.0, device=eps_pred.device, requires_grad=True)
            eps_pred = eps_pred[valid_idx]
            evolutionary_advantage = evolutionary_advantage[valid_idx]

        if eps_pred.shape[0] == 0:
            return torch.tensor(0.0, device=eps_pred.device, requires_grad=True)

        return F.huber_loss(eps_pred, evolutionary_advantage.detach(),
                            delta=self.huber_delta)


class EvolutionPotentialRankingLoss(nn.Module):
    """
    Ranking loss supervised by evolutionary_advantage (not raw abundance).

    EPS should rank sequences by their latent evolutionary potential:
      Phi = "the latent potential of a sequence to maintain or gain
             evolutionary advantage under a given selection context"

    Loss: pairwise margin ranking
      If EA_i > EA_j, then EPS_i should > EPS_j

    Respects ea_valid_mask — only valid samples participate in ranking.
    """

    def __init__(self, margin: float = 0.1, n_pairs: int = 128):
        super().__init__()
        self.margin = margin
        self.n_pairs = n_pairs

    def forward(self, eps_pred: torch.Tensor,
                evolutionary_advantage: torch.Tensor,
                ea_valid: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            eps_pred: (B,) - predicted Evolutionary Potential Scores
            evolutionary_advantage: (B,) - observational EA targets
            ea_valid: (B,) bool mask — True where EA is defined

        Returns:
            loss: scalar
        """
        B = eps_pred.shape[0]

        if ea_valid is not None:
            valid_idx = ea_valid.nonzero(as_tuple=True)[0]
            n_valid = valid_idx.shape[0]
            if n_valid < 2:
                return torch.tensor(0.0, device=eps_pred.device, requires_grad=True)
            eps_pred = eps_pred[valid_idx]
            evolutionary_advantage = evolutionary_advantage[valid_idx]
            B = n_valid

        if B < 2:
            return torch.tensor(0.0, device=eps_pred.device, requires_grad=True)

        n_pairs = min(self.n_pairs, B * (B - 1) // 2)

        idx_a = torch.randint(0, B, (n_pairs,), device=eps_pred.device)
        idx_b = torch.randint(0, B, (n_pairs,), device=eps_pred.device)

        mask = idx_a != idx_b
        if mask.sum() == 0:
            return torch.tensor(0.0, device=eps_pred.device, requires_grad=True)
        idx_a = idx_a[mask]
        idx_b = idx_b[mask]

        eps_a = eps_pred[idx_a]
        eps_b = eps_pred[idx_b]
        ea_a = evolutionary_advantage[idx_a]
        ea_b = evolutionary_advantage[idx_b]

        ea_diff = ea_a - ea_b
        eps_diff = eps_a - eps_b

        loss = torch.relu(-ea_diff.sign() * eps_diff + self.margin)
        return loss.mean()


class RolloutLoss(nn.Module):
    """
    Multi-step trajectory rollout loss.

    Accumulates MSE + cosine distance over all predicted steps:
      L_rollout = sum_k [MSE(Z_pred_k, Z_true_k) + alpha * (1 - cos(Z_pred_k, Z_true_k))]
    """

    def __init__(self, cosine_weight: float = 0.1):
        super().__init__()
        self.cosine_weight = cosine_weight

    def forward(self, z_trajectory: List[torch.Tensor],
                z_targets: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            z_trajectory: list of predicted z states [z_0, z_1, ..., z_n]
            z_targets: list of target z states [z_1_true, z_2_true, ..., z_n_true]

        Returns:
            loss: scalar
        """
        assert len(z_trajectory) == len(z_targets) + 1

        total_loss = torch.tensor(0.0, device=z_trajectory[0].device)
        for k in range(len(z_targets)):
            z_pred_k = z_trajectory[k + 1]
            z_true_k = z_targets[k].detach()

            mse = F.mse_loss(z_pred_k, z_true_k)

            cos_sim = F.cosine_similarity(z_pred_k, z_true_k, dim=-1).mean()
            cos_loss = 1.0 - cos_sim

            total_loss = total_loss + mse + self.cosine_weight * cos_loss

        return total_loss / len(z_targets)


class TemporalSmoothnessLoss(nn.Module):
    """
    Penalizes discontinuous transitions between adjacent time steps.

    L_smooth = mean(||delta_z_{k+1} - delta_z_k||^2)

    Encourages the transition operator to produce smoothly varying
    dynamics as a function of continuous time. This is key to
    generalizing across unseen round transitions.
    """

    def __init__(self):
        super().__init__()

    def forward(self, delta_z_trajectory: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            delta_z_trajectory: list of delta_z at each step [d_0, d_1, ..., d_n]

        Returns:
            loss: scalar
        """
        if len(delta_z_trajectory) < 2:
            return torch.tensor(0.0, device=delta_z_trajectory[0].device,
                                requires_grad=True)

        total = torch.tensor(0.0, device=delta_z_trajectory[0].device)
        for k in range(len(delta_z_trajectory) - 1):
            diff = delta_z_trajectory[k + 1] - delta_z_trajectory[k]
            total = total + (diff ** 2).mean()

        return total / (len(delta_z_trajectory) - 1)


class ETOLoss(nn.Module):
    """
    Legacy loss - kept for backward compatibility.
    """
    def __init__(self, lambda_population: float = 1.0,
                 lambda_ranking: float = 0.5,
                 round_weights: Optional[Dict[str, float]] = None):
        super().__init__()
        self.transition_loss = TransitionLoss()
        self.population_loss = PopulationReconstructionLoss()
        self.ranking_loss = EvolutionPotentialRankingLoss()

        self.lambda_population = lambda_population
        self.lambda_ranking = lambda_ranking
        self.round_weights = round_weights or {
            'R3_R5': 1.0, 'R5_R9': 0.2, 'R9_R10': 1.0, 'R10_R11': 1.0
        }

    def forward(self, outputs: Dict[str, torch.Tensor],
                batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        pair_idx = batch['pair_idx']
        round_names = ['R3_R5', 'R5_R9', 'R9_R10', 'R10_R11']

        weights = torch.ones(pair_idx.shape[0], device=pair_idx.device)
        for i, name in enumerate(round_names):
            mask = pair_idx == i
            weights[mask] = self.round_weights.get(name, 1.0)

        l_trans_raw = F.mse_loss(
            outputs['z_next_pred'], outputs['z_t_target'].detach(),
            reduction='none'
        ).mean(dim=-1)
        l_trans = (l_trans_raw * weights).mean()

        l_pop = self.population_loss(
            outputs['pop_pred'],
            batch['log_count_curr'],
            batch['log_count_prev'],
        )

        l_rank = self.ranking_loss(
            outputs['phi'],
            batch.get('evolutionary_advantage', outputs['phi'].detach()),
        )

        total = l_trans + self.lambda_population * l_pop + self.lambda_ranking * l_rank

        return {
            'total': total,
            'transition': l_trans,
            'population': l_pop,
            'ranking': l_rank,
        }


class RefactoredETOLoss(nn.Module):
    """
    Refactored loss with evolutionary advantage target and temporal smoothness.

    EPS (Evolutionary Potential Score) is supervised by:
      - EA Huber regression (absolute calibration)
      - EA ranking (relative ordering)
      - Both masked by ea_valid (R10→R11 excluded from EA supervision)

    Total loss:
      L = lambda_transition * L_transition
        + lambda_population * L_population
        + lambda_advantage * L_advantage_rank
        + lambda_ea_regression * L_ea_regression
        + lambda_rollout * L_rollout
        + lambda_smooth * L_smooth
    """

    def __init__(self, lambda_transition: float = 1.0,
                 lambda_population: float = 0.5,
                 lambda_advantage: float = 1.0,
                 lambda_ea_regression: float = 0.5,
                 lambda_rollout: float = 0.3,
                 lambda_smooth: float = 0.05,
                 ranking_margin: float = 0.1,
                 ranking_n_pairs: int = 128,
                 huber_delta: float = 1.0):
        super().__init__()
        self.transition_loss_fn = TransitionLoss()
        self.population_loss_fn = PopulationReconstructionLoss()
        self.advantage_rank_loss_fn = EvolutionPotentialRankingLoss(
            margin=ranking_margin, n_pairs=ranking_n_pairs
        )
        self.ea_regression_loss_fn = EARegressionLoss(huber_delta=huber_delta)
        self.rollout_loss_fn = RolloutLoss()
        self.smoothness_loss_fn = TemporalSmoothnessLoss()

        self.lambda_transition = lambda_transition
        self.lambda_population = lambda_population
        self.lambda_advantage = lambda_advantage
        self.lambda_ea_regression = lambda_ea_regression
        self.lambda_rollout = lambda_rollout
        self.lambda_smooth = lambda_smooth

    def forward(self, outputs: Dict[str, torch.Tensor],
                batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: model outputs dict with keys:
                'z_next_pred', 'z_t_target', 'pop_pred', 'eps'
                optional: 'z_trajectory', 'z_rollout_targets'
            batch: data batch dict with keys:
                'log_count_prev', 'log_count_curr', 'evolutionary_advantage'
                optional: 'ea_valid'

        Returns:
            loss dict
        """
        l_trans = self.transition_loss_fn(
            outputs['z_next_pred'],
            outputs['z_t_target'],
        )

        l_pop = self.population_loss_fn(
            outputs['pop_pred'],
            batch['log_count_curr'],
            batch['log_count_prev'],
        )

        advantage_target = batch['evolutionary_advantage']
        ea_valid = batch.get('ea_valid', None)

        l_advantage = self.advantage_rank_loss_fn(
            outputs['eps'],
            advantage_target,
            ea_valid,
        )

        l_ea_reg = self.ea_regression_loss_fn(
            outputs['eps'],
            advantage_target,
            ea_valid,
        )

        l_rollout = torch.tensor(0.0, device=l_trans.device)
        l_smooth = torch.tensor(0.0, device=l_trans.device)
        if 'z_trajectory' in outputs and 'z_rollout_targets' in outputs:
            l_rollout = self.rollout_loss_fn(
                outputs['z_trajectory'],
                outputs['z_rollout_targets'],
            )
        if 'delta_z_trajectory' in outputs and len(outputs['delta_z_trajectory']) >= 2:
            l_smooth = self.smoothness_loss_fn(outputs['delta_z_trajectory'])

        total = (self.lambda_transition * l_trans +
                 self.lambda_population * l_pop +
                 self.lambda_advantage * l_advantage +
                 self.lambda_ea_regression * l_ea_reg +
                 self.lambda_rollout * l_rollout +
                 self.lambda_smooth * l_smooth)

        return {
            'total': total,
            'transition': l_trans,
            'population': l_pop,
            'advantage': l_advantage,
            'ea_regression': l_ea_reg,
            'rollout': l_rollout,
            'smooth': l_smooth,
        }


def count_nans(tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, int]:
    """Count NaN and Inf values in a dict of tensors for monitoring."""
    counts = {}
    for key, val in tensor_dict.items():
        if isinstance(val, torch.Tensor):
            n_nan = torch.isnan(val).sum().item()
            n_inf = torch.isinf(val).sum().item()
            if n_nan > 0 or n_inf > 0:
                counts[key] = {'nan': int(n_nan), 'inf': int(n_inf)}
        elif isinstance(val, list):
            for i, v in enumerate(val):
                if isinstance(v, torch.Tensor):
                    n_nan = torch.isnan(v).sum().item()
                    n_inf = torch.isinf(v).sum().item()
                    if n_nan > 0 or n_inf > 0:
                        counts[f"{key}[{i}]"] = {'nan': int(n_nan), 'inf': int(n_inf)}
    return counts

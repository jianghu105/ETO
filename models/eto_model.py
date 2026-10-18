import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List

from .sequence_encoder import SequenceEncoder
from .structure_encoder import StructureEncoder
from .evolutionary_state import IndividualStateBuilder, AbundanceFreeStateBuilder
from .population_encoder import PopulationEncoder, AbundanceFreePopulationEncoder
from .context_encoder import ContextEncoder, HybridContextEncoder, ContinuousTimeEncoder, ROUND_TIME_MAP
from .transition_operator import ConditionalTransitionOperator, RolloutTransitionOperator
from .potential_decoder import PopulationDecoder, PotentialDecoder
from .individual_potential_decoder import IndividualPotentialDecoder, SetPopulationDecoder


class ETOModel(nn.Module):
    """
    Legacy ETO model - kept for backward compatibility.
    """
    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config['model']

        self.seq_encoder = SequenceEncoder(
            seq_length=config['data']['seq_length'],
            embedding_dim=model_cfg['sequence_encoder']['embedding_dim'],
            conv_kernels=model_cfg['sequence_encoder']['conv_kernels'],
            conv_filters=model_cfg['sequence_encoder']['conv_filters'],
            dropout=model_cfg['sequence_encoder']['dropout'],
        )

        seq_emb_dim = model_cfg['sequence_encoder']['embedding_dim']
        self.state_builder = IndividualStateBuilder(seq_emb_dim)
        state_dim = self.state_builder.state_dim

        pop_cfg = model_cfg['population_encoder']
        self.pop_encoder = PopulationEncoder(
            state_dim=state_dim,
            population_dim=pop_cfg['population_dim'],
            n_heads=pop_cfg['n_heads'],
            dropout=pop_cfg['dropout'],
        )

        ctx_cfg = model_cfg['context_encoder']
        self.context_encoder = ContextEncoder(
            n_context_types=ctx_cfg['n_context_types'],
            context_dim=ctx_cfg['context_dim'],
        )

        trans_cfg = model_cfg['transition_operator']
        self.transition_op = ConditionalTransitionOperator(
            state_dim=pop_cfg['population_dim'],
            context_dim=ctx_cfg['context_dim'],
            hidden_dim=trans_cfg['hidden_dim'],
            n_layers=trans_cfg['n_layers'],
            dropout=trans_cfg['dropout'],
        )

        self.pop_decoder = PopulationDecoder(
            population_dim=pop_cfg['population_dim'],
            hidden_dim=pop_cfg['population_dim'],
            n_rounds=len(config['data']['rounds']),
        )

        pot_cfg = model_cfg['potential_decoder']
        self.potential_decoder = PotentialDecoder(
            seq_embedding_dim=seq_emb_dim,
            population_dim=pop_cfg['population_dim'],
            hidden_dim=pot_cfg['hidden_dim'],
            dropout=pot_cfg['dropout'],
        )

        self.population_dim = pop_cfg['population_dim']
        self.seq_embedding_dim = seq_emb_dim

    def encode_sequences(self, one_hot: torch.Tensor) -> torch.Tensor:
        return self.seq_encoder(one_hot)

    def build_states(self, seq_emb: torch.Tensor,
                     log_count: torch.Tensor,
                     velocity: torch.Tensor) -> torch.Tensor:
        return self.state_builder(seq_emb, log_count, velocity)

    def encode_population(self, states: torch.Tensor,
                          mask: torch.Tensor = None) -> torch.Tensor:
        return self.pop_encoder(states, mask)

    def encode_context(self, context_id: torch.Tensor) -> torch.Tensor:
        return self.context_encoder(context_id)

    def transition(self, z_t: torch.Tensor,
                   e_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.transition_op(z_t, e_t)

    def decode_population(self, z_future: torch.Tensor) -> torch.Tensor:
        return self.pop_decoder(z_future)

    def decode_potential(self, seq_emb: torch.Tensor,
                         z_future: torch.Tensor) -> torch.Tensor:
        return self.potential_decoder(seq_emb, z_future)

    def forward(self, batch: Dict[str, torch.Tensor],
                return_intermediates: bool = False) -> Dict[str, torch.Tensor]:
        one_hot = batch['one_hot']
        log_count_prev = batch['log_count_prev']
        log_count_curr = batch['log_count_curr']
        context_id = batch['context_id']

        seq_emb = self.encode_sequences(one_hot)

        velocity_prev = batch['x_prev'][:, 1] if batch['x_prev'].shape[-1] > 1 \
            else torch.zeros_like(log_count_prev)
        state_prev = self.build_states(seq_emb, log_count_prev, velocity_prev)

        velocity_curr = batch['x_curr'][:, 1] if batch['x_curr'].shape[-1] > 1 \
            else torch.zeros_like(log_count_curr)
        state_curr = self.build_states(seq_emb, log_count_curr, velocity_curr)

        z_t = self.encode_population(state_prev.unsqueeze(1))
        e_t = self.encode_context(context_id)
        z_next_pred, delta_z = self.transition(z_t, e_t)

        z_t_target = self.encode_population(state_curr.unsqueeze(1))

        pop_pred = self.decode_population(z_next_pred)
        phi = self.decode_potential(seq_emb, z_next_pred)

        result = {
            'z_next_pred': z_next_pred,
            'z_t_target': z_t_target,
            'delta_z': delta_z,
            'pop_pred': pop_pred,
            'phi': phi,
            'seq_emb': seq_emb,
        }

        if return_intermediates:
            result['z_t'] = z_t
            result['e_t'] = e_t
            result['state_prev'] = state_prev
            result['state_curr'] = state_curr

        return result


class RefactoredETOModel(nn.Module):
    """
    Refactored ETO model with continuous-time evolutionary dynamics.

    ETO learns a continuous evolutionary transition operator that estimates
    latent evolutionary advantage under selective environments.

    Core equation:
        Z(t + dt) = Z(t) + dt * T_theta(Z(t), E(t))

    where:
        t       — continuous evolutionary time tau (configurable per round)
        dt      — evolutionary time step (delta_tau)
        Z(t)    — population latent evolutionary state (abundance-free)
        E(t)    — context embedding (selection type + Fourier time)
        T_theta — learned conditional transition operator

    Individual Evolutionary Potential Score (EPS / Phi):
        Phi_i(t,c) = f_theta(Z(t), x_i, E(t))

        Phi represents:
        "the latent potential of a sequence to maintain or gain evolutionary
         advantage under a given selection context"

        Phi is NOT current abundance.
        Phi is NOT direct binding affinity.

    Components:
      1. Structure feature interface (GC content, MFE proxy, paired prob)
      2. Abundance-free pathway (dynamic_state vs observation_state)
      3. Continuous-time context encoder (Fourier features + context type)
      4. Individual potential decoder (set-based, variable population size)
      5. Multi-step rollout with temporal smoothness regularization

    Data flow:
        Input: one_hot, structure_features, velocity, log_count, context_id, continuous_time
            |
        Sequence Encoder -> seq_emb (128d)
        Structure Encoder -> struct_emb (16d) [optional]
            |
        AbundanceFreeStateBuilder -> {dynamic_state, observation_state}
            |
        AbundanceFreePopulationEncoder(dynamic_states) -> Z_t
            |
        HybridContextEncoder(context_id, continuous_time) -> E_t
            |
        RolloutTransitionOperator(Z_t, E_t) -> Z(t+dt)
            |
        IndividualPotentialDecoder(Z(t+dt), seq_emb, velocity, struct_emb) -> EPS_i
        SetPopulationDecoder(Z(t+dt)) -> population_stats
            |
        Output: {z_next_pred, z_t_target, pop_pred, eps, ...}
    """

    def __init__(self, config: dict):
        super().__init__()
        model_cfg = config['model']

        seq_cfg = model_cfg['sequence_encoder']
        self.seq_encoder = SequenceEncoder(
            seq_length=config['data']['seq_length'],
            embedding_dim=seq_cfg['embedding_dim'],
            conv_kernels=seq_cfg['conv_kernels'],
            conv_filters=seq_cfg['conv_filters'],
            dropout=seq_cfg['dropout'],
        )

        seq_emb_dim = seq_cfg['embedding_dim']

        struct_cfg = model_cfg.get('structure_encoder', {})
        self.structure_dim = struct_cfg.get('structure_dim', 16)
        self.use_structure = struct_cfg.get('enabled', True)
        if self.use_structure:
            self.structure_encoder = StructureEncoder(
                structure_dim=self.structure_dim,
                dropout=struct_cfg.get('dropout', 0.1),
            )
        else:
            self.structure_dim = 0

        state_cfg = model_cfg.get('evolutionary_state', {})
        velocity_dim = state_cfg.get('velocity_dim', 1)
        self.state_builder = AbundanceFreeStateBuilder(
            seq_embedding_dim=seq_emb_dim,
            velocity_dim=velocity_dim,
            structure_dim=self.structure_dim,
        )
        dynamic_dim = self.state_builder.dynamic_dim

        pop_cfg = model_cfg['population_encoder']
        self.pop_encoder = AbundanceFreePopulationEncoder(
            dynamic_dim=dynamic_dim,
            population_dim=pop_cfg['population_dim'],
            observation_dim=1,
            n_heads=pop_cfg['n_heads'],
            dropout=pop_cfg['dropout'],
            use_observation=pop_cfg.get('use_observation', True),
        )

        ctx_cfg = model_cfg['context_encoder']
        self.context_encoder = HybridContextEncoder(
            n_context_types=ctx_cfg['n_context_types'],
            n_rounds=ctx_cfg.get('n_rounds', 10),
            context_embed_dim=ctx_cfg.get('context_embed_dim', 16),
            round_embed_dim=ctx_cfg.get('round_embed_dim', 16),
            output_dim=ctx_cfg['context_dim'],
            n_fourier_freqs=ctx_cfg.get('n_fourier_freqs', 8),
            use_continuous_time=ctx_cfg.get('use_continuous_time', True),
        )

        trans_cfg = model_cfg['transition_operator']
        self.transition_op = RolloutTransitionOperator(
            state_dim=pop_cfg['population_dim'],
            context_dim=ctx_cfg['context_dim'],
            hidden_dim=trans_cfg['hidden_dim'],
            n_layers=trans_cfg['n_layers'],
            dropout=trans_cfg['dropout'],
            max_rollout_steps=trans_cfg.get('max_rollout_steps', 4),
        )

        self.pop_decoder = SetPopulationDecoder(
            population_dim=pop_cfg['population_dim'],
            hidden_dim=pop_cfg['population_dim'],
            n_rounds=len(config['data']['rounds']),
        )

        pot_cfg = model_cfg.get('individual_potential_decoder', {})
        self.potential_decoder = IndividualPotentialDecoder(
            population_dim=pop_cfg['population_dim'],
            seq_embedding_dim=seq_emb_dim,
            velocity_dim=velocity_dim,
            structure_dim=self.structure_dim,
            hidden_dim=pot_cfg.get('hidden_dim', 512),
            dropout=pot_cfg.get('dropout', 0.1),
        )

        self.population_dim = pop_cfg['population_dim']
        self.seq_embedding_dim = seq_emb_dim

    def encode_sequences(self, one_hot: torch.Tensor) -> torch.Tensor:
        return self.seq_encoder(one_hot)

    def encode_structure(self, structure_features: torch.Tensor = None
                         ) -> torch.Tensor:
        if not self.use_structure or structure_features is None:
            return None
        return self.structure_encoder(structure_features)

    def build_states(self, seq_emb: torch.Tensor,
                     velocity: torch.Tensor,
                     log_count: torch.Tensor,
                     structure_emb: torch.Tensor = None
                     ) -> Dict[str, torch.Tensor]:
        return self.state_builder(seq_emb, velocity, log_count, structure_emb)

    def encode_population(self, dynamic_states: torch.Tensor,
                          observation_states: torch.Tensor = None,
                          mask: torch.Tensor = None) -> torch.Tensor:
        return self.pop_encoder(dynamic_states, observation_states, mask)

    def encode_context(self, context_id: torch.Tensor,
                       round_idx: torch.Tensor = None,
                       continuous_time: torch.Tensor = None) -> torch.Tensor:
        return self.context_encoder(context_id, round_idx, continuous_time)

    def transition(self, z_t: torch.Tensor,
                   e_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.transition_op(z_t, e_t)

    def rollout(self, z_start: torch.Tensor,
                context_sequence: List[torch.Tensor],
                n_steps: int = None) -> Dict:
        return self.transition_op.rollout(z_start, context_sequence, n_steps)

    def decode_population(self, z_future: torch.Tensor) -> torch.Tensor:
        return self.pop_decoder(z_future)

    def decode_potential(self, z_future: torch.Tensor,
                         seq_emb: torch.Tensor,
                         velocity: torch.Tensor,
                         structure_emb: torch.Tensor = None) -> torch.Tensor:
        return self.potential_decoder(z_future, seq_emb, velocity, structure_emb)

    def forward(self, batch: Dict[str, torch.Tensor],
                return_intermediates: bool = False,
                rollout_steps: int = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass with new architecture.

        Args:
            batch: dict with keys:
                'one_hot': (B, 4, L)
                'structure_features': (B, 3) [optional]
                'x_prev': (B, 2) - [log_count_prev, velocity_prev]
                'x_curr': (B, 2) - [log_count_curr, velocity_curr]
                'log_count_prev': (B,)
                'log_count_curr': (B,)
                'context_id': (B,)
                'round_idx': (B,) [optional]
            return_intermediates: whether to return intermediate states
            rollout_steps: if set, perform multi-step rollout

        Returns:
            outputs dict
        """
        one_hot = batch['one_hot']
        log_count_prev = batch['log_count_prev']
        log_count_curr = batch['log_count_curr']
        context_id = batch['context_id']
        round_idx = batch.get('round_idx', None)
        continuous_time = batch.get('continuous_time', None)

        seq_emb = self.encode_sequences(one_hot)

        struct_emb = None
        if 'structure_features' in batch and self.use_structure:
            struct_emb = self.encode_structure(batch['structure_features'])

        velocity_prev = batch['x_prev'][:, 1] if batch['x_prev'].shape[-1] > 1 \
            else torch.zeros_like(log_count_prev)
        states_prev = self.build_states(seq_emb, velocity_prev, log_count_prev, struct_emb)

        velocity_curr = batch['x_curr'][:, 1] if batch['x_curr'].shape[-1] > 1 \
            else torch.zeros_like(log_count_curr)
        states_curr = self.build_states(seq_emb, velocity_curr, log_count_curr, struct_emb)

        dynamic_prev = states_prev['dynamic_state'].unsqueeze(1)
        obs_prev = states_prev['observation_state'].unsqueeze(1)
        z_t = self.encode_population(dynamic_prev, obs_prev)

        e_t = self.encode_context(context_id, round_idx, continuous_time)
        z_next_pred, delta_z = self.transition(z_t, e_t)

        dynamic_curr = states_curr['dynamic_state'].unsqueeze(1)
        z_t_target = self.encode_population(dynamic_curr)

        pop_pred = self.decode_population(z_next_pred)

        eps = self.decode_potential(z_next_pred, seq_emb, velocity_prev, struct_emb)

        result = {
            'z_next_pred': z_next_pred,
            'z_t_target': z_t_target,
            'delta_z': delta_z,
            'pop_pred': pop_pred,
            'eps': eps,
            'seq_emb': seq_emb,
            'z_t': z_t,
        }

        if rollout_steps is not None and rollout_steps > 1:
            context_sequence = [e_t]
            for step in range(1, rollout_steps):
                step_context = context_id + step
                step_context = step_context.clamp(max=3)
                step_round = round_idx + step if round_idx is not None else None
                step_time = None
                if continuous_time is not None:
                    step_time = continuous_time + step
                elif step_round is not None:
                    step_time = step_round.float()
                e_step = self.encode_context(step_context, step_round, step_time)
                context_sequence.append(e_step)

            rollout_result = self.rollout(z_t, context_sequence, rollout_steps)
            result['z_trajectory'] = rollout_result['z_trajectory']
            result['delta_z_trajectory'] = rollout_result['delta_z_trajectory']
            result['z_rollout_targets'] = [z_t_target] * rollout_steps

        if return_intermediates:
            result['e_t'] = e_t
            result['states_prev'] = states_prev
            result['states_curr'] = states_curr
            result['struct_emb'] = struct_emb

        return result

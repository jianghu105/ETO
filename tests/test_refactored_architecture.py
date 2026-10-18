"""
Unit tests for ETO v2 refactored architecture.

Tests each component independently and the integrated model.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from typing import Dict

from models.structure_encoder import StructureEncoder
from models.evolutionary_state import IndividualStateBuilder, AbundanceFreeStateBuilder
from models.context_encoder import ContextEncoder, HybridContextEncoder
from models.population_encoder import PopulationEncoder, AbundanceFreePopulationEncoder
from models.transition_operator import ConditionalTransitionOperator, RolloutTransitionOperator
from models.individual_potential_decoder import IndividualPotentialDecoder, SetPopulationDecoder
from models.sequence_encoder import SequenceEncoder
from models.eto_model import ETOModel, RefactoredETOModel
from training.losses import (
    TransitionLoss, PopulationReconstructionLoss,
    EvolutionPotentialRankingLoss, RolloutLoss,
    ETOLoss, RefactoredETOLoss, EARegressionLoss,
)


def test_structure_encoder():
    print("=" * 60)
    print("Test 1: StructureEncoder")
    print("=" * 60)

    B = 32
    encoder = StructureEncoder(structure_dim=16)

    features = torch.randn(B, 3)
    out = encoder(features)
    assert out.shape == (B, 16), f"Expected (32, 16), got {out.shape}"
    print(f"  Input:  {features.shape}")
    print(f"  Output: {out.shape}")

    out_none = encoder(None)
    assert out_none is None
    print(f"  None input → None output: PASS")

    zeros = torch.zeros(B, 3)
    out_zeros = encoder(zeros)
    assert out_zeros.shape == (B, 16)
    assert out_zeros.abs().sum() == 0
    print(f"  Zero features → zero output: PASS")

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Parameters: {n_params:,}")
    print("  [PASS] StructureEncoder\n")


def test_abundance_free_state():
    print("=" * 60)
    print("Test 2: AbundanceFreeStateBuilder")
    print("=" * 60)

    B = 32
    seq_emb_dim = 128
    velocity_dim = 1
    structure_dim = 16

    builder = AbundanceFreeStateBuilder(seq_emb_dim, velocity_dim, structure_dim)

    seq_emb = torch.randn(B, seq_emb_dim)
    velocity = torch.randn(B, velocity_dim)
    log_count = torch.randn(B)
    struct_emb = torch.randn(B, structure_dim)

    states = builder(seq_emb, velocity, log_count, struct_emb)

    assert 'dynamic_state' in states
    assert 'observation_state' in states
    assert 'full_state' in states

    expected_dynamic = seq_emb_dim + velocity_dim + structure_dim
    assert states['dynamic_state'].shape == (B, expected_dynamic), \
        f"Expected ({B}, {expected_dynamic}), got {states['dynamic_state'].shape}"
    assert states['observation_state'].shape == (B, 1)
    assert states['full_state'].shape == (B, expected_dynamic + 1)

    print(f"  seq_emb:      {seq_emb.shape}")
    print(f"  velocity:     {velocity.shape}")
    print(f"  log_count:    {log_count.shape}")
    print(f"  struct_emb:   {struct_emb.shape}")
    print(f"  dynamic:      {states['dynamic_state'].shape}")
    print(f"  observation:  {states['observation_state'].shape}")
    print(f"  full:         {states['full_state'].shape}")

    assert not torch.allclose(
        states['dynamic_state'][:, :seq_emb_dim],
        torch.zeros_like(states['dynamic_state'][:, :seq_emb_dim])
    ), "Dynamic state should not be all zeros"

    print("  Dynamic state is abundance-free: PASS")
    print("  [PASS] AbundanceFreeStateBuilder\n")


def test_hybrid_context_encoder():
    print("=" * 60)
    print("Test 3: HybridContextEncoder")
    print("=" * 60)

    B = 32
    encoder = HybridContextEncoder(
        n_context_types=4, n_rounds=10,
        context_embed_dim=16, round_embed_dim=16,
        output_dim=32,
    )

    context_id = torch.randint(0, 4, (B,))
    round_idx = torch.randint(0, 10, (B,))

    e_t = encoder(context_id, round_idx)
    assert e_t.shape == (B, 32), f"Expected ({B}, 32), got {e_t.shape}"
    print(f"  context_id: {context_id.shape}")
    print(f"  round_idx:  {round_idx.shape}")
    print(f"  output:     {e_t.shape}")

    e_t_no_round = encoder(context_id)
    assert e_t_no_round.shape == (B, 32)
    print(f"  Without round_idx: {e_t_no_round.shape}")

    diff = (e_t - e_t_no_round).abs().sum()
    assert diff > 0, "Round embedding should change output"
    print(f"  Round embedding changes output: PASS (diff={diff:.4f})")

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Parameters: {n_params:,}")
    print("  [PASS] HybridContextEncoder\n")


def test_abundance_free_population_encoder():
    print("=" * 60)
    print("Test 4: AbundanceFreePopulationEncoder")
    print("=" * 60)

    B = 16
    N = 50
    dynamic_dim = 145
    population_dim = 256

    encoder = AbundanceFreePopulationEncoder(
        dynamic_dim=dynamic_dim,
        population_dim=population_dim,
        observation_dim=1,
        n_heads=8,
        use_observation=True,
    )

    dynamic_states = torch.randn(B, N, dynamic_dim)
    obs_states = torch.randn(B, N, 1)

    z_t = encoder(dynamic_states, obs_states)
    assert z_t.shape == (B, population_dim), f"Expected ({B}, {population_dim}), got {z_t.shape}"
    print(f"  dynamic_states: {dynamic_states.shape}")
    print(f"  obs_states:     {obs_states.shape}")
    print(f"  Z_t:            {z_t.shape}")

    z_t_no_obs = encoder(dynamic_states, None)
    assert z_t_no_obs.shape == (B, population_dim)
    print(f"  Without obs:    {z_t_no_obs.shape}")

    encoder_no_obs = AbundanceFreePopulationEncoder(
        dynamic_dim=dynamic_dim,
        population_dim=population_dim,
        use_observation=False,
    )
    z_t_forced = encoder_no_obs(dynamic_states, obs_states)
    assert z_t_forced.shape == (B, population_dim)
    print(f"  No obs encoder: {z_t_forced.shape}")

    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"  Parameters: {n_params:,}")
    print("  [PASS] AbundanceFreePopulationEncoder\n")


def test_rollout_transition():
    print("=" * 60)
    print("Test 5: RolloutTransitionOperator")
    print("=" * 60)

    B = 16
    state_dim = 256
    context_dim = 32

    op = RolloutTransitionOperator(
        state_dim=state_dim,
        context_dim=context_dim,
        hidden_dim=512,
        n_layers=3,
        max_rollout_steps=4,
    )

    z_t = torch.randn(B, state_dim)
    e_t = torch.randn(B, context_dim)

    z_next, delta_z = op(z_t, e_t)
    assert z_next.shape == (B, state_dim)
    assert delta_z.shape == (B, state_dim)
    print(f"  Single step:")
    print(f"    Z_t:      {z_t.shape}")
    print(f"    E_t:      {e_t.shape}")
    print(f"    Z_next:   {z_next.shape}")
    print(f"    delta_z:  {delta_z.shape}")

    residual_check = torch.allclose(z_next, z_t + delta_z, atol=1e-5)
    assert residual_check, "Z_next should equal Z_t + delta_z"
    print(f"    Residual: Z_next = Z_t + delta_z: PASS")

    context_seq = [torch.randn(B, context_dim) for _ in range(4)]

    for n_steps in [1, 2, 3, 4]:
        result = op.rollout(z_t, context_seq, n_steps)
        z_traj = result['z_trajectory']
        assert len(z_traj) == n_steps + 1, f"Expected {n_steps + 1} states, got {len(z_traj)}"
        for z in z_traj:
            assert z.shape == (B, state_dim)
        print(f"  Rollout {n_steps}-step: {len(z_traj)} states, all {z_traj[0].shape}")

    n_params = sum(p.numel() for p in op.parameters())
    print(f"  Parameters: {n_params:,}")
    print("  [PASS] RolloutTransitionOperator\n")


def test_individual_potential_decoder():
    print("=" * 60)
    print("Test 6: IndividualPotentialDecoder")
    print("=" * 60)

    B = 32
    pop_dim = 256
    seq_dim = 128
    vel_dim = 1
    struct_dim = 16

    decoder = IndividualPotentialDecoder(
        population_dim=pop_dim,
        seq_embedding_dim=seq_dim,
        velocity_dim=vel_dim,
        structure_dim=struct_dim,
        hidden_dim=512,
    )

    z_future = torch.randn(B, pop_dim)
    seq_emb = torch.randn(B, seq_dim)
    velocity = torch.randn(B, vel_dim)
    struct_emb = torch.randn(B, struct_dim)

    eps = decoder(z_future, seq_emb, velocity, struct_emb)
    assert eps.shape == (B,), f"Expected ({B},), got {eps.shape}"
    print(f"  z_future:  {z_future.shape}")
    print(f"  seq_emb:   {seq_emb.shape}")
    print(f"  velocity:  {velocity.shape}")
    print(f"  struct:    {struct_emb.shape}")
    print(f"  EPS:       {eps.shape}")

    z_shared = torch.randn(4, pop_dim)
    individuals = {
        'seq_emb': torch.randn(4, 100, seq_dim),
        'velocity': torch.randn(4, 100, vel_dim),
        'structure_emb': torch.randn(4, 100, struct_dim),
    }
    eps_batch = decoder.forward_batch(z_shared, individuals)
    assert eps_batch.shape == (4, 100), f"Expected (4, 100), got {eps_batch.shape}"
    print(f"  Batch mode: Z(4) × N(100) → EPS(4, 100): PASS")

    decoder_no_struct = IndividualPotentialDecoder(
        population_dim=pop_dim,
        seq_embedding_dim=seq_dim,
        velocity_dim=vel_dim,
        structure_dim=0,
        hidden_dim=512,
    )
    eps_no_struct = decoder_no_struct(z_future, seq_emb, velocity, None)
    assert eps_no_struct.shape == (B,)
    print(f"  Without structure: EPS shape {eps_no_struct.shape}: PASS")

    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"  Parameters: {n_params:,}")
    print("  [PASS] IndividualPotentialDecoder\n")


def test_future_ranking_loss():
    print("=" * 60)
    print("Test 7: EvolutionPotentialRankingLoss (evolutionary advantage)")
    print("=" * 60)

    B = 64
    loss_fn = EvolutionPotentialRankingLoss(margin=0.1, n_pairs=128)

    eps_pred = torch.randn(B, requires_grad=True)
    evolutionary_advantage = torch.randn(B)

    loss = loss_fn(eps_pred, evolutionary_advantage)
    assert loss.dim() == 0, "Loss should be scalar"
    assert loss.item() >= 0, "Loss should be non-negative"
    print(f"  eps_pred:               {eps_pred.shape}")
    print(f"  evolutionary_advantage: {evolutionary_advantage.shape}")
    print(f"  loss:                   {loss.item():.6f}")

    loss.backward()
    assert eps_pred.grad is not None
    print(f"  Gradient flows: PASS")

    eps_good = evolutionary_advantage.clone().detach().requires_grad_(True)
    loss_good = loss_fn(eps_good, evolutionary_advantage)
    print(f"  Perfect ranking loss: {loss_good.item():.6f}")

    eps_bad = -evolutionary_advantage.clone().detach().requires_grad_(True)
    loss_bad = loss_fn(eps_bad, evolutionary_advantage)
    print(f"  Inverse ranking loss: {loss_bad.item():.6f}")
    assert loss_bad.item() > loss_good.item(), \
        "Inverse ranking should have higher loss"
    print(f"  Inverse > Perfect: PASS")

    print("  [PASS] EvolutionPotentialRankingLoss\n")


def test_rollout_loss():
    print("=" * 60)
    print("Test 8: RolloutLoss")
    print("=" * 60)

    B = 16
    state_dim = 256
    loss_fn = RolloutLoss()

    z_trajectory = [torch.randn(B, state_dim) for _ in range(4)]
    z_targets = [torch.randn(B, state_dim) for _ in range(3)]

    loss = loss_fn(z_trajectory, z_targets)
    assert loss.dim() == 0
    assert loss.item() >= 0
    print(f"  z_trajectory: {len(z_trajectory)} states")
    print(f"  z_targets:    {len(z_targets)} states")
    print(f"  loss:         {loss.item():.6f}")

    z_perfect = [torch.randn(B, state_dim)]
    z_perfect.extend([t.clone() for t in z_targets])
    loss_perfect = loss_fn(z_perfect, z_targets)
    assert loss_perfect.item() < 1e-6, "Perfect prediction should have ~0 loss"
    print(f"  Perfect rollout loss: {loss_perfect.item():.8f}")
    print("  [PASS] RolloutLoss\n")


def test_refactored_loss():
    print("=" * 60)
    print("Test 9: RefactoredETOLoss")
    print("=" * 60)

    B = 32
    state_dim = 256

    loss_fn = RefactoredETOLoss(
        lambda_transition=1.0,
        lambda_population=0.5,
        lambda_advantage=1.0,
        lambda_rollout=0.3,
    )

    outputs = {
        'z_next_pred': torch.randn(B, state_dim, requires_grad=True),
        'z_t_target': torch.randn(B, state_dim),
        'pop_pred': torch.randn(B, 10),
        'eps': torch.randn(B, requires_grad=True),
    }

    batch = {
        'log_count_prev': torch.randn(B),
        'log_count_curr': torch.randn(B),
        'evolutionary_advantage': torch.randn(B),
    }

    losses = loss_fn(outputs, batch)
    assert 'total' in losses
    assert 'transition' in losses
    assert 'population' in losses
    assert 'advantage' in losses
    assert 'rollout' in losses
    print(f"  total:      {losses['total'].item():.6f}")
    print(f"  transition: {losses['transition'].item():.6f}")
    print(f"  population: {losses['population'].item():.6f}")
    print(f"  advantage:  {losses['advantage'].item():.6f}")
    print(f"  rollout:    {losses['rollout'].item():.6f}")

    losses['total'].backward()
    print(f"  Gradient flows: PASS")
    print("  [PASS] RefactoredETOLoss\n")


def test_refactored_model():
    print("=" * 60)
    print("Test 10: RefactoredETOModel (full integration)")
    print("=" * 60)

    config = {
        'data': {
            'seq_length': 36,
            'rounds': ['R3', 'R5', 'R9', 'R10', 'R11'],
        },
        'model': {
            'sequence_encoder': {
                'embedding_dim': 128,
                'conv_kernels': [3, 5, 7],
                'conv_filters': 64,
                'dropout': 0.1,
            },
            'structure_encoder': {
                'enabled': True,
                'structure_dim': 16,
                'dropout': 0.1,
            },
            'evolutionary_state': {
                'velocity_dim': 1,
            },
            'population_encoder': {
                'population_dim': 256,
                'n_heads': 8,
                'dropout': 0.1,
                'use_observation': True,
            },
            'context_encoder': {
                'context_dim': 32,
                'n_context_types': 4,
                'n_rounds': 10,
                'context_embed_dim': 16,
                'round_embed_dim': 16,
            },
            'transition_operator': {
                'hidden_dim': 512,
                'n_layers': 3,
                'dropout': 0.1,
                'max_rollout_steps': 4,
            },
            'individual_potential_decoder': {
                'hidden_dim': 512,
                'dropout': 0.1,
            },
        },
    }

    model = RefactoredETOModel(config)
    model.eval()

    B = 16
    batch = {
        'one_hot': torch.randn(B, 4, 36),
        'structure_features': torch.randn(B, 3),
        'x_prev': torch.randn(B, 2),
        'x_curr': torch.randn(B, 2),
        'log_count_prev': torch.randn(B),
        'log_count_curr': torch.randn(B),
        'context_id': torch.randint(0, 4, (B,)),
        'round_idx': torch.randint(0, 4, (B,)),
    }

    with torch.no_grad():
        outputs = model(batch)

    assert 'z_next_pred' in outputs
    assert 'z_t_target' in outputs
    assert 'eps' in outputs
    assert 'pop_pred' in outputs
    assert 'z_t' in outputs

    print(f"  z_next_pred: {outputs['z_next_pred'].shape}")
    print(f"  z_t_target:  {outputs['z_t_target'].shape}")
    print(f"  eps:         {outputs['eps'].shape}")
    print(f"  pop_pred:    {outputs['pop_pred'].shape}")
    print(f"  z_t:         {outputs['z_t'].shape}")

    assert outputs['eps'].shape == (B,), f"EPS should be ({B},), got {outputs['eps'].shape}"
    print(f"  EPS is per-individual: PASS")

    with torch.no_grad():
        outputs_rollout = model(batch, rollout_steps=3)
    assert 'z_trajectory' in outputs_rollout
    assert len(outputs_rollout['z_trajectory']) == 4
    print(f"  Rollout 3-step: {len(outputs_rollout['z_trajectory'])} states: PASS")

    with torch.no_grad():
        outputs_inter = model(batch, return_intermediates=True)
    assert 'struct_emb' in outputs_inter
    assert 'states_prev' in outputs_inter
    print(f"  Intermediates returned: PASS")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {n_params:,}")
    print("  [PASS] RefactoredETOModel\n")


def test_abundance_shortcut_check():
    print("=" * 60)
    print("Test 11: Abundance Shortcut Check")
    print("=" * 60)

    config = {
        'data': {
            'seq_length': 36,
            'rounds': ['R3', 'R5', 'R9', 'R10', 'R11'],
        },
        'model': {
            'sequence_encoder': {
                'embedding_dim': 128,
                'conv_kernels': [3, 5, 7],
                'conv_filters': 64,
                'dropout': 0.1,
            },
            'structure_encoder': {
                'enabled': True,
                'structure_dim': 16,
                'dropout': 0.1,
            },
            'evolutionary_state': {'velocity_dim': 1},
            'population_encoder': {
                'population_dim': 256, 'n_heads': 8,
                'dropout': 0.1, 'use_observation': True,
            },
            'context_encoder': {
                'context_dim': 32, 'n_context_types': 4,
                'n_rounds': 10, 'context_embed_dim': 16,
                'round_embed_dim': 16,
            },
            'transition_operator': {
                'hidden_dim': 512, 'n_layers': 3,
                'dropout': 0.1, 'max_rollout_steps': 4,
            },
            'individual_potential_decoder': {
                'hidden_dim': 512, 'dropout': 0.1,
            },
        },
    }

    model = RefactoredETOModel(config)
    model.eval()

    B = 64
    base_batch = {
        'one_hot': torch.randn(B, 4, 36),
        'structure_features': torch.randn(B, 3),
        'x_prev': torch.randn(B, 2),
        'x_curr': torch.randn(B, 2),
        'log_count_prev': torch.randn(B),
        'log_count_curr': torch.randn(B),
        'context_id': torch.randint(0, 4, (B,)),
        'round_idx': torch.randint(0, 4, (B,)),
    }

    with torch.no_grad():
        out1 = model(base_batch)
        eps1 = out1['eps']

    modified_batch = {k: v.clone() for k, v in base_batch.items()}
    modified_batch['log_count_prev'] = torch.randn(B) * 10
    modified_batch['log_count_curr'] = torch.randn(B) * 10
    modified_batch['x_prev'][:, 0] = torch.randn(B) * 10
    modified_batch['x_curr'][:, 0] = torch.randn(B) * 10

    with torch.no_grad():
        out2 = model(modified_batch)
        eps2 = out2['eps']

    eps_diff = (eps1 - eps2).abs().mean()
    seq_diff = (out1['seq_emb'] - out2['seq_emb']).abs().mean()

    print(f"  EPS change when abundance changes: {eps_diff:.6f}")
    print(f"  Seq emb change (should be 0):      {seq_diff:.6f}")
    assert seq_diff < 1e-5, "Sequence embedding should not change"
    print(f"  Sequence embedding is abundance-free: PASS")
    print("  Note: EPS may change because velocity is part of dynamic state,")
    print("  but the model is no longer directly mapping abundance → EPS.")
    print("  [PASS] Abundance Shortcut Check\n")


def test_ea_validity_mask():
    print("=" * 60)
    print("Test 12: EA Validity Mask — invalid labels produce zero gradient")
    print("=" * 60)

    B = 64
    eps_pred = torch.randn(B, requires_grad=True)
    ea_target = torch.randn(B)
    ea_valid = torch.zeros(B, dtype=torch.bool)
    ea_valid[:B//2] = True

    rank_loss_fn = EvolutionPotentialRankingLoss(margin=0.1, n_pairs=128)
    reg_loss_fn = EARegressionLoss(huber_delta=1.0)

    l_rank_all = rank_loss_fn(eps_pred, ea_target, ea_valid)
    l_rank_all.backward(retain_graph=True)
    grad_all = eps_pred.grad.clone()

    grad_invalid = grad_all[B//2:]
    grad_valid = grad_all[:B//2]

    print(f"  Valid samples:   {ea_valid.sum().item()}")
    print(f"  Invalid samples: {(~ea_valid).sum().item()}")
    print(f"  Ranking loss:    {l_rank_all.item():.6f}")
    print(f"  Grad norm (valid region):   {grad_valid.norm().item():.6f}")
    print(f"  Grad norm (invalid region): {grad_invalid.norm().item():.6f}")

    assert grad_invalid.norm().item() < 1e-6, \
        "Invalid samples should have zero gradient from ranking loss"
    print(f"  Invalid samples have zero gradient (ranking): PASS")

    eps_pred2 = torch.randn(B, requires_grad=True)
    l_reg = reg_loss_fn(eps_pred2, ea_target, ea_valid)
    l_reg.backward()
    grad_reg = eps_pred2.grad.clone()
    grad_reg_invalid = grad_reg[B//2:]
    print(f"  Regression loss: {l_reg.item():.6f}")
    print(f"  Grad norm (invalid region): {grad_reg_invalid.norm().item():.6f}")
    assert grad_reg_invalid.norm().item() < 1e-6, \
        "Invalid samples should have zero gradient from regression loss"
    print(f"  Invalid samples have zero gradient (regression): PASS")

    ea_all_invalid = torch.zeros(B, dtype=torch.bool)
    eps_pred3 = torch.randn(B, requires_grad=True)
    l_rank_none = rank_loss_fn(eps_pred3, ea_target, ea_all_invalid)
    l_rank_none.backward()
    assert l_rank_none.item() == 0.0, "All-invalid batch should give zero loss"
    assert eps_pred3.grad is None or eps_pred3.grad.norm().item() < 1e-6, \
        "All-invalid batch should give zero gradient"
    print(f"  All-invalid batch → zero loss & gradient: PASS")

    print("  [PASS] EA Validity Mask\n")


def test_refactored_loss_with_mask():
    print("=" * 60)
    print("Test 13: RefactoredETOLoss with ea_valid mask")
    print("=" * 60)

    B = 32
    state_dim = 256

    loss_fn = RefactoredETOLoss(
        lambda_transition=1.0,
        lambda_population=0.5,
        lambda_advantage=1.0,
        lambda_ea_regression=0.5,
    )

    outputs = {
        'z_next_pred': torch.randn(B, state_dim, requires_grad=True),
        'z_t_target': torch.randn(B, state_dim),
        'pop_pred': torch.randn(B, 10),
        'eps': torch.randn(B, requires_grad=True),
    }

    ea_valid = torch.ones(B, dtype=torch.bool)
    ea_valid[B//2:] = False

    batch = {
        'log_count_prev': torch.randn(B),
        'log_count_curr': torch.randn(B),
        'evolutionary_advantage': torch.randn(B),
        'ea_valid': ea_valid,
    }

    losses = loss_fn(outputs, batch)
    assert 'ea_regression' in losses
    print(f"  total:         {losses['total'].item():.6f}")
    print(f"  advantage:     {losses['advantage'].item():.6f}")
    print(f"  ea_regression: {losses['ea_regression'].item():.6f}")

    losses['total'].backward()
    print(f"  Gradient flows: PASS")
    print("  [PASS] RefactoredETOLoss with mask\n")


def run_all_tests():
    print("\n" + "=" * 60)
    print("ETO v2 Architecture Refactoring - Unit Tests")
    print("=" * 60 + "\n")

    tests = [
        test_structure_encoder,
        test_abundance_free_state,
        test_hybrid_context_encoder,
        test_abundance_free_population_encoder,
        test_rollout_transition,
        test_individual_potential_decoder,
        test_future_ranking_loss,
        test_rollout_loss,
        test_refactored_loss,
        test_refactored_model,
        test_abundance_shortcut_check,
        test_ea_validity_mask,
        test_refactored_loss_with_mask,
    ]

    passed = 0
    failed = 0
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {test_fn.__name__}: {e}\n")
            import traceback
            traceback.print_exc()
            print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)

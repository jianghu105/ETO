"""
Search Strategies
=================

Four strategies under the same budget constraint:
1. Random Search — random candidate selection
2. Static Prediction Ranking — rank by sequence-only model prediction
3. Genetic Algorithm — evolutionary optimization with crossover/mutation
4. ETO-Guided Search — use ETO-Core ensemble predictions + uncertainty + novelty
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from .candidate_generator import CandidatePool, single_point_mutations
from .scoring import SearchScorer, ensemble_predict, compute_novelty


def prepare_candidate_batch(sequences: List[str],
                            transition_type: int = 0,
                            delta_t: float = 2.0,
                            seq_length: int = 36) -> Dict[str, torch.Tensor]:
    """
    Build a batch dict for model inference from candidate sequences.
    Uses default history (zeros) and context for new candidates.
    """
    from data.preprocessing import one_hot_encode, compute_structure_features

    n = len(sequences)
    max_history = 5

    one_hot = np.zeros((n, 4, seq_length), dtype=np.float32)
    structure_features = np.zeros((n, 3), dtype=np.float32)
    history_counts = np.zeros((n, max_history), dtype=np.float32)
    history_mask = np.zeros((n, max_history), dtype=bool)
    transition_types = np.full(n, transition_type, dtype=np.int64)
    delta_ts = np.full(n, delta_t, dtype=np.float32)

    for i, seq in enumerate(sequences):
        one_hot[i] = one_hot_encode(seq, seq_length)
        structure_features[i] = compute_structure_features(seq)

    return {
        'one_hot': torch.from_numpy(one_hot),
        'structure_features': torch.from_numpy(structure_features),
        'history_counts': torch.from_numpy(history_counts),
        'history_mask': torch.from_numpy(history_mask),
        'transition_type': torch.from_numpy(transition_types),
        'delta_t': torch.from_numpy(delta_ts),
        'round_from': torch.zeros(n, dtype=torch.long),
        'round_to': torch.ones(n, dtype=torch.long),
        'target': torch.zeros(n, dtype=torch.float32),
        'target_mask': torch.ones(n, dtype=torch.bool),
        'family_id': torch.zeros(n, dtype=torch.long),
        'seq_idx': torch.arange(n, dtype=torch.long),
    }


class SearchResult:
    """Stores results from a search run."""

    def __init__(self, strategy_name: str, budget: int):
        self.strategy_name = strategy_name
        self.budget = budget
        self.selected = []
        self.scores = []
        self.uncertainties = []
        self.novelty_scores = []
        self.provenance = []

    def add(self, sequence: str, score: float, uncertainty: float = 0.0,
            novelty: float = 0.0, provenance: Dict = None):
        self.selected.append(sequence)
        self.scores.append(score)
        self.uncertainties.append(uncertainty)
        self.novelty_scores.append(novelty)
        self.provenance.append(provenance or {})

    def get_families(self, pool: CandidatePool) -> List[int]:
        fams = []
        for seq in self.selected:
            prov = pool.get_provenance(seq)
            fams.append(prov.get('family_id', -1))
        return fams

    def summary(self, pool: CandidatePool) -> Dict:
        n_unique_families = len(set(self.get_families(pool)))
        return {
            'strategy': self.strategy_name,
            'budget': self.budget,
            'n_selected': len(self.selected),
            'n_unique_families': n_unique_families,
            'mean_score': float(np.mean(self.scores)) if self.scores else 0,
            'mean_uncertainty': float(np.mean(self.uncertainties)),
            'mean_novelty': float(np.mean(self.novelty_scores)),
        }


def random_search(pool: CandidatePool, budget: int, seed: int = 42
                  ) -> SearchResult:
    """Select candidates uniformly at random."""
    rng = np.random.RandomState(seed)
    result = SearchResult('random', budget)

    available = pool.get_unselected()
    if len(available) <= budget:
        selected = available
    else:
        indices = rng.choice(len(available), size=budget, replace=False)
        selected = [available[i] for i in indices]

    for seq in selected:
        pool.select(seq)
        result.add(seq, score=0.0, provenance=pool.get_provenance(seq))

    return result


def static_ranking_search(models: List, pool: CandidatePool,
                          budget: int, device: str = 'cpu',
                          batch_size: int = 512) -> SearchResult:
    """Rank candidates by static model prediction (no uncertainty/novelty)."""
    result = SearchResult('static_ranking', budget)
    available = pool.get_unselected()

    all_scores = []
    for i in range(0, len(available), batch_size):
        batch_seqs = available[i:i + batch_size]
        batch = prepare_candidate_batch(batch_seqs)
        mean_pred, std_pred = ensemble_predict(models, batch, device)
        all_scores.extend(mean_pred.tolist())

    ranked = np.argsort(-np.array(all_scores))

    for rank_idx in ranked[:budget]:
        seq = available[rank_idx]
        pool.select(seq)
        result.add(seq, score=all_scores[rank_idx],
                   provenance=pool.get_provenance(seq))

    return result


def eto_guided_search(models: List, pool: CandidatePool,
                      budget: int, device: str = 'cpu',
                      beta: float = 0.5, gamma: float = 0.3,
                      batch_size: int = 512,
                      mode: str = 'full') -> SearchResult:
    """
    ETO-guided search with ensemble predictions + uncertainty + novelty.

    Greedy: at each step, score all available candidates, pick the best.
    For efficiency, batch inference is used.
    """
    result = SearchResult(f'eto_guided_{mode}', budget)
    scorer = SearchScorer(beta=beta, gamma=gamma, mode=mode)

    selected_seqs = []

    for step in range(budget):
        available = pool.get_unselected()
        if not available:
            break

        all_mean = []
        all_std = []
        for i in range(0, len(available), batch_size):
            batch_seqs = available[i:i + batch_size]
            batch = prepare_candidate_batch(batch_seqs)
            mean_pred, std_pred = ensemble_predict(models, batch, device)
            all_mean.extend(mean_pred.tolist())
            all_std.extend(std_pred.tolist())

        mean_pred = np.array(all_mean)
        std_pred = np.array(all_std)

        scores = scorer.score(mean_pred, std_pred, available, selected_seqs)

        best_idx = np.argmax(scores)
        best_seq = available[best_idx]
        best_score = scores[best_idx]
        best_std = std_pred[best_idx]
        best_novelty = compute_novelty(best_seq, selected_seqs)

        pool.select(best_seq)
        selected_seqs.append(best_seq)
        result.add(best_seq, score=best_score, uncertainty=best_std,
                   novelty=best_novelty, provenance=pool.get_provenance(best_seq))

    return result


def genetic_algorithm_search(models: List, pool: CandidatePool,
                             budget: int, device: str = 'cpu',
                             population_size: int = 100,
                             n_generations: int = 50,
                             mutation_rate: int = 1,
                             batch_size: int = 512) -> SearchResult:
    """
    Simple genetic algorithm search.

    Fitness = model prediction. Selection = top-k. Crossover = single-point.
    Mutation = random single-point mutation.
    """
    rng = np.random.RandomState(42)
    result = SearchResult('genetic_algorithm', budget)

    available = pool.get_unselected()
    if len(available) < population_size:
        population = available[:]
    else:
        init_idx = rng.choice(len(available), population_size, replace=False)
        population = [available[i] for i in init_idx]

    def evaluate_fitness(seqs):
        fitness = []
        for i in range(0, len(seqs), batch_size):
            batch = prepare_candidate_batch(seqs[i:i + batch_size])
            mean_pred, _ = ensemble_predict(models, batch, device)
            fitness.extend(mean_pred.tolist())
        return np.array(fitness)

    for gen in range(n_generations):
        fitness = evaluate_fitness(population)

        top_k = max(2, population_size // 4)
        elite_idx = np.argsort(-fitness)[:top_k]
        elite = [population[i] for i in elite_idx]

        new_pop = elite[:]
        while len(new_pop) < population_size:
            p1, p2 = rng.choice(len(elite), 2, replace=False)
            parent1, parent2 = elite[p1], elite[p2]

            crossover = rng.randint(1, len(parent1))
            child = parent1[:crossover] + parent2[crossover:]

            if rng.random() < 0.3:
                pos = rng.randint(len(child))
                orig = child[pos]
                new_base = rng.choice([b for b in 'ACGT' if b != orig])
                child = child[:pos] + new_base + child[pos + 1:]

            new_pop.append(child)

        population = new_pop[:population_size]

    final_fitness = evaluate_fitness(population)
    ranked = np.argsort(-final_fitness)

    for idx in ranked[:budget]:
        seq = population[idx]
        if seq in pool.candidates:
            pool.select(seq)
            result.add(seq, score=final_fitness[idx],
                       provenance=pool.get_provenance(seq))
        else:
            result.add(seq, score=final_fitness[idx],
                       provenance={'source': 'ga_novel'})

    return result

"""
Search Module Tests
"""

import sys
import os
import numpy as np
import torch
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search.candidate_generator import (
    single_point_mutations, multi_point_mutations, CandidatePool,
)
from search.scoring import compute_novelty, SearchScorer
from search.strategies import (
    prepare_candidate_batch, random_search, SearchResult,
)


class TestCandidateGeneration:

    def test_single_point_mutations(self):
        seq = 'ACGT'
        mutants = single_point_mutations(seq)
        assert len(mutants) == 4 * 3
        for m in mutants:
            assert len(m['sequence']) == 4
            assert m['parent'] == seq
            assert m['n_mutations'] == 1

    def test_multi_point_mutations(self):
        seq = 'ACGTACGTACGT'
        mutants = multi_point_mutations(seq, n_mutations=2, max_candidates=10)
        assert len(mutants) <= 10
        for m in mutants:
            assert m['n_mutations'] == 2

    def test_candidate_pool(self):
        seqs = ['ACGTACGT', 'TGCATGCA', 'AAAACCCC']
        families = np.array([0, 1, 2])
        pool = CandidatePool(seqs, families)
        n = pool.generate_mutants(n_parents=3, mutations_per_parent=1,
                                  max_per_parent=5)
        assert n > 0
        assert pool.n_total > 0
        assert pool.n_available == pool.n_total


class TestScoring:

    def test_novelty_empty(self):
        assert compute_novelty('ACGT', []) == 1.0

    def test_novelty_identical(self):
        nov = compute_novelty('ACGTACGT', ['ACGTACGT'], k=4)
        assert nov == 0.0

    def test_scorer_prediction_only(self):
        scorer = SearchScorer(mode='prediction')
        mean = np.array([1.0, 2.0, 3.0])
        std = np.array([0.1, 0.2, 0.3])
        scores = scorer.score(mean, std, ['A', 'B', 'C'], [])
        np.testing.assert_array_equal(scores, mean)

    def test_scorer_full(self):
        scorer = SearchScorer(beta=1.0, gamma=1.0, mode='full')
        mean = np.array([1.0, 2.0])
        std = np.array([0.5, 0.5])
        scores = scorer.score(mean, std, ['ACGT', 'TTTT'], [])
        assert scores[0] > mean[0]


class TestSearchStrategies:

    def test_random_search(self):
        seqs = ['ACGTACGT', 'TGCATGCA', 'AAAACCCC', 'GGGGTTTT']
        families = np.array([0, 1, 2, 3])
        pool = CandidatePool(seqs, families)
        pool.generate_mutants(n_parents=4, max_per_parent=5)

        result = random_search(pool, budget=3)
        assert len(result.selected) == 3 or len(result.selected) == pool.n_total

    def test_prepare_batch(self):
        seqs = ['ACGTACGT' * 4 + 'ACGT', 'TGCATGCA' * 4 + 'TGCA']
        batch = prepare_candidate_batch(seqs[:1])
        assert batch['one_hot'].shape == (1, 4, 36)

    def test_search_result_summary(self):
        result = SearchResult('test', budget=5)
        result.add('ACGT', score=1.0, novelty=0.5)
        result.add('TGGG', score=2.0, novelty=0.8)
        seqs = ['ACGTACGT', 'TGCATGCA']
        families = np.array([0, 1])
        pool = CandidatePool(seqs, families)
        summary = result.summary(pool)
        assert summary['n_selected'] == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])

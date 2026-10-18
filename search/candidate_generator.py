"""
Candidate Generation for Budget-Constrained Search
===================================================

Generates novel candidate sequences via constrained mutation from existing
sequences. Tracks provenance (parent, mutation position, family).
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

BASES = ['A', 'C', 'G', 'T']


def single_point_mutations(sequence: str, max_candidates: int = None
                           ) -> List[Dict]:
    """
    Generate all single-point mutants of a sequence.

    Returns list of dicts with:
      - sequence: mutant sequence
      - parent: parent sequence
      - position: mutation position
      - original_base: base before mutation
      - new_base: base after mutation
    """
    mutants = []
    for pos in range(len(sequence)):
        orig = sequence[pos]
        for base in BASES:
            if base == orig:
                continue
            mutant = sequence[:pos] + base + sequence[pos + 1:]
            mutants.append({
                'sequence': mutant,
                'parent': sequence,
                'position': pos,
                'original_base': orig,
                'new_base': base,
                'n_mutations': 1,
            })
            if max_candidates and len(mutants) >= max_candidates:
                return mutants
    return mutants


def multi_point_mutations(sequence: str, n_mutations: int = 2,
                          max_candidates: int = None) -> List[Dict]:
    """
    Generate multi-point mutants (n_mutations positions changed).
    Uses random sampling for efficiency.
    """
    rng = np.random.RandomState(42)
    mutants = []
    seq_len = len(sequence)

    attempts = 0
    max_attempts = max_candidates * 10 if max_candidates else 10000

    while len(mutants) < (max_candidates or 100) and attempts < max_attempts:
        positions = rng.choice(seq_len, size=n_mutations, replace=False)
        seq_list = list(sequence)
        changes = []
        for pos in positions:
            orig = seq_list[pos]
            new_base = rng.choice([b for b in BASES if b != orig])
            seq_list[pos] = new_base
            changes.append((pos, orig, new_base))

        mutant = ''.join(seq_list)
        mutants.append({
            'sequence': mutant,
            'parent': sequence,
            'positions': [c[0] for c in changes],
            'changes': changes,
            'n_mutations': n_mutations,
        })
        attempts += 1

    return mutants


class CandidatePool:
    """
    Manages the pool of candidate sequences for search.

    Tracks:
    - Source sequences (existing, not used for training test set)
    - Generated mutants
    - Already-selected candidates (for diversity)
    - Provenance information
    """

    def __init__(self, source_sequences: List[str],
                 source_family_ids: np.ndarray,
                 exclude_sequences: set = None):
        self.source_seqs = list(source_sequences)
        self.source_families = source_family_ids
        self.exclude = exclude_sequences or set()

        self.candidates = {}
        self.selected = set()
        self.family_map = defaultdict(list)

        for i, seq in enumerate(self.source_seqs):
            if seq not in self.exclude:
                self.family_map[source_family_ids[i]].append(seq)

    def generate_mutants(self, n_parents: int = 100,
                         mutations_per_parent: int = 1,
                         max_per_parent: int = 20,
                         seed: int = 42) -> int:
        """
        Generate mutant candidates from source sequences.

        Returns number of new candidates added.
        """
        rng = np.random.RandomState(seed)
        families = list(self.family_map.keys())
        n_added = 0

        selected_parents = rng.choice(
            len(self.source_seqs),
            size=min(n_parents, len(self.source_seqs)),
            replace=False
        )

        for idx in selected_parents:
            parent = self.source_seqs[idx]
            if parent in self.exclude:
                continue

            if mutations_per_parent == 1:
                mutants = single_point_mutations(parent, max_per_parent)
            else:
                mutants = multi_point_mutations(
                    parent, mutations_per_parent, max_per_parent)

            for m in mutants:
                seq = m['sequence']
                if seq not in self.candidates and seq not in self.exclude:
                    m['family_id'] = int(self.source_families[idx])
                    m['source'] = 'mutant'
                    self.candidates[seq] = m
                    n_added += 1

        return n_added

    def get_unselected(self) -> List[str]:
        """Return all candidates not yet selected."""
        return [seq for seq in self.candidates
                if seq not in self.selected]

    def select(self, sequence: str):
        """Mark a candidate as selected."""
        self.selected.add(sequence)

    def get_family_diversity(self, sequences: List[str]) -> int:
        """Count unique families in a set of sequences."""
        families = set()
        for seq in sequences:
            if seq in self.candidates:
                families.add(self.candidates[seq]['family_id'])
        return len(families)

    def get_provenance(self, sequence: str) -> Dict:
        """Get provenance info for a candidate."""
        return self.candidates.get(sequence, {})

    @property
    def n_total(self) -> int:
        return len(self.candidates)

    @property
    def n_selected(self) -> int:
        return len(self.selected)

    @property
    def n_available(self) -> int:
        return self.n_total - self.n_selected

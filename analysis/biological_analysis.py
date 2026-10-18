"""
Top-K Biological Analysis for ETO Predictions

Analyzes high-EPS vs low-EPS sequences for biological characteristics.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def compute_gc_content(sequences: List[str]) -> np.ndarray:
    """Compute GC content for each sequence."""
    gc_contents = []
    for seq in sequences:
        gc = (seq.count('G') + seq.count('C')) / len(seq)
        gc_contents.append(gc)
    return np.array(gc_contents)


def compute_nucleotide_freq(sequences: List[str]) -> Dict[str, float]:
    """Compute average nucleotide frequencies."""
    all_chars = ''.join(sequences)
    total = len(all_chars)
    freq = {
        'A': all_chars.count('A') / total,
        'T': all_chars.count('T') / total,
        'G': all_chars.count('G') / total,
        'C': all_chars.count('C') / total,
    }
    return freq


def compute_kmer_freq(sequences: List[str], k: int = 3) -> Dict[str, float]:
    """Compute k-mer frequencies."""
    kmer_counts = Counter()
    total_kmers = 0
    for seq in sequences:
        for i in range(len(seq) - k + 1):
            kmer = seq[i:i+k]
            kmer_counts[kmer] += 1
            total_kmers += 1

    return {kmer: count / total_kmers for kmer, count in kmer_counts.items()}


def analyze_top_k(predictions_path: str, data_path: str, output_dir: str,
                  top_k_list: List[int] = [100, 500, 1000]):
    """
    Analyze top-K sequences by EPS score.

    Args:
        predictions_path: Path to predicted_potential.csv
        data_path: Path to original MPO xlsx file
        output_dir: Output directory for results
        top_k_list: List of K values to analyze
    """
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("ETO Top-K Biological Analysis")
    print("=" * 70)

    # Load predictions
    print("\nLoading predictions...")
    pred_df = pd.read_csv(predictions_path)
    pred_df = pred_df.sort_values('rank')
    print(f"Total sequences: {len(pred_df)}")

    # Load original data for abundance info
    print("Loading original data...")
    raw_df = pd.read_excel(data_path)

    # Find sequence column
    seq_col = None
    for col in raw_df.columns:
        if '序列' in col or 'sequence' in col.lower():
            seq_col = col
            break
    if seq_col is None:
        seq_col = raw_df.columns[0]

    print(f"Sequence column: {seq_col}")

    # Find round columns
    round_cols = {}
    for col in raw_df.columns:
        for r in ['R3', 'R5', 'R9', 'R10', 'R11']:
            if r in col:
                round_cols[r] = col
                break

    print(f"Round columns: {round_cols}")

    # Merge predictions with original data
    merged = pred_df.merge(raw_df, left_on='sequence', right_on=seq_col, how='left')
    print(f"Merged sequences: {len(merged)}")

    # Analysis for different K values
    results = {}

    for K in top_k_list:
        print(f"\n{'=' * 70}")
        print(f"Analysis: Top {K} vs Bottom {K}")
        print('=' * 70)

        top_df = merged.head(K)
        bottom_df = merged.tail(K)

        top_seqs = top_df['sequence'].tolist()
        bottom_seqs = bottom_df['sequence'].tolist()

        # GC content
        top_gc = compute_gc_content(top_seqs)
        bottom_gc = compute_gc_content(bottom_seqs)

        print(f"\nGC Content:")
        print(f"  Top-{K}:    {top_gc.mean():.3f} ± {top_gc.std():.3f}")
        print(f"  Bottom-{K}: {bottom_gc.mean():.3f} ± {bottom_gc.std():.3f}")
        print(f"  Difference: {top_gc.mean() - bottom_gc.mean():+.3f}")

        # Nucleotide frequencies
        top_freq = compute_nucleotide_freq(top_seqs)
        bottom_freq = compute_nucleotide_freq(bottom_seqs)

        print(f"\nNucleotide Frequencies:")
        print(f"  Top-{K}:    A={top_freq['A']:.3f} T={top_freq['T']:.3f} G={top_freq['G']:.3f} C={top_freq['C']:.3f}")
        print(f"  Bottom-{K}: A={bottom_freq['A']:.3f} T={bottom_freq['T']:.3f} G={bottom_freq['G']:.3f} C={bottom_freq['C']:.3f}")

        # Abundance analysis
        print(f"\nAbundance Analysis (R3 → R5 → R9 → R10 → R11):")
        for label, df_subset in [('Top', top_df), ('Bottom', bottom_df)]:
            counts = []
            for r in ['R3', 'R5', 'R9', 'R10', 'R11']:
                if r in round_cols:
                    mean_count = df_subset[round_cols[r]].sum() / len(df_subset)
                    counts.append(mean_count)
                    print(f"  {label}-{K} {r}: mean count = {mean_count:.2f}")

        # EPS score distribution
        print(f"\nEPS Score Distribution:")
        print(f"  Top-{K}:    {top_df['EPS_score'].mean():.3f} ± {top_df['EPS_score'].std():.3f}")
        print(f"  Bottom-{K}: {bottom_df['EPS_score'].mean():.3f} ± {bottom_df['EPS_score'].std():.3f}")

        results[K] = {
            'top_gc_mean': float(top_gc.mean()),
            'top_gc_std': float(top_gc.std()),
            'bottom_gc_mean': float(bottom_gc.mean()),
            'bottom_gc_std': float(bottom_gc.std()),
            'top_freq': top_freq,
            'bottom_freq': bottom_freq,
        }

    # Visualization
    print(f"\n{'=' * 70}")
    print("Generating visualizations...")
    print('=' * 70)

    # Plot 1: GC content distribution
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, K in enumerate(top_k_list):
        top_seqs = merged.head(K)['sequence'].tolist()
        bottom_seqs = merged.tail(K)['sequence'].tolist()
        top_gc = compute_gc_content(top_seqs)
        bottom_gc = compute_gc_content(bottom_seqs)

        axes[i].hist(top_gc, bins=30, alpha=0.6, label=f'Top-{K}', color='red', density=True)
        axes[i].hist(bottom_gc, bins=30, alpha=0.6, label=f'Bottom-{K}', color='blue', density=True)
        axes[i].set_xlabel('GC Content')
        axes[i].set_ylabel('Density')
        axes[i].set_title(f'K={K}')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gc_content_distribution.png'), dpi=150)
    print("Saved: gc_content_distribution.png")

    # Plot 2: EPS score distribution
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(merged['EPS_score'], bins=100, color='green', alpha=0.7, edgecolor='black')
    ax.axvline(merged.head(1000)['EPS_score'].min(), color='red', linestyle='--', label='Top-1000 threshold')
    ax.axvline(merged.tail(1000)['EPS_score'].max(), color='blue', linestyle='--', label='Bottom-1000 threshold')
    ax.set_xlabel('EPS Score')
    ax.set_ylabel('Count')
    ax.set_title('EPS Score Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'eps_score_distribution.png'), dpi=150)
    print("Saved: eps_score_distribution.png")

    # Plot 3: Top-K abundance enrichment
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for i, K in enumerate(top_k_list):
        top_df = merged.head(K)
        rounds = ['R3', 'R5', 'R9', 'R10', 'R11']
        means = []
        for r in rounds:
            if r in round_cols:
                means.append(top_df[round_cols[r]].mean())

        axes[i].plot(rounds, means, 'o-', linewidth=2, markersize=8, color='red')
        axes[i].set_xlabel('Round')
        axes[i].set_ylabel('Mean Count')
        axes[i].set_title(f'Top-{K} Enrichment')
        axes[i].grid(True, alpha=0.3)
        axes[i].set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_k_enrichment.png'), dpi=150)
    print("Saved: top_k_enrichment.png")

    # Save summary
    summary_df = pd.DataFrame(results).T
    summary_df.to_csv(os.path.join(output_dir, 'biological_analysis_summary.csv'))
    print("Saved: biological_analysis_summary.csv")

    print(f"\n{'=' * 70}")
    print(f"Analysis complete! Results saved to {output_dir}/")
    print('=' * 70)

    return results


if __name__ == '__main__':
    predictions_path = 'experiments/real_v1/predicted_potential.csv'
    data_path = 'dataset/20230531AptDMC双端-MPO.xlsx'
    output_dir = 'experiments/real_v1/biological_analysis'

    analyze_top_k(predictions_path, data_path, output_dir, top_k_list=[100, 500, 1000])

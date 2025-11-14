"""
Topological Data Analysis for NFL defensive formations.

This module computes persistent homology on point clouds representing
defensive player positions using Vietoris-Rips filtration.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from pathlib import Path
from tqdm import tqdm

# TDA libraries
from ripser import ripser
import persim

# For visualization
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


class DefensiveCoverageTDA:
    """
    Compute and analyze persistent homology of defensive formations.

    This class applies Vietoris-Rips filtration to point clouds of defensive
    player positions and extracts topological features.
    """

    def __init__(self, max_dimension: int = 2, max_edge_length: Optional[float] = None):
        """
        Initialize TDA analyzer.

        Args:
            max_dimension: Maximum homology dimension to compute (0, 1, or 2)
                          H0 = connected components (clusters)
                          H1 = loops/holes (coverage gaps)
                          H2 = voids (rarely used for 2D data)
            max_edge_length: Maximum edge length in Rips complex (default: auto)
        """
        self.max_dimension = max_dimension
        self.max_edge_length = max_edge_length
        self.persistence_diagrams = {}

    def compute_persistence(
        self,
        point_cloud: np.ndarray,
        play_id: Optional[Tuple[int, int]] = None
    ) -> Dict:
        """
        Compute persistent homology for a single defensive formation.

        Uses Vietoris-Rips filtration to build a simplicial complex and
        computes persistent homology in dimensions 0, 1, (and optionally 2).

        Args:
            point_cloud: Array of shape (n_defenders, 2) with (x, y) positions
            play_id: Optional tuple (gameId, playId) for tracking

        Returns:
            Dictionary containing:
                - 'dgms': List of persistence diagrams [H0, H1, ...]
                - 'num_points': Number of points in cloud
                - 'play_id': Play identifier (if provided)
        """
        if point_cloud.shape[0] < 2:
            # Need at least 2 points for meaningful homology
            return {
                'dgms': [np.array([]), np.array([])],
                'num_points': point_cloud.shape[0],
                'play_id': play_id
            }

        # Compute Vietoris-Rips persistent homology
        result = ripser(
            point_cloud,
            maxdim=self.max_dimension,
            thresh=self.max_edge_length if self.max_edge_length else np.inf
        )

        persistence_result = {
            'dgms': result['dgms'],
            'num_points': point_cloud.shape[0],
            'play_id': play_id,
            'cocycles': result.get('cocycles', None)  # Sometimes useful for interpretation
        }

        if play_id is not None:
            self.persistence_diagrams[play_id] = persistence_result

        return persistence_result

    def compute_persistence_batch(
        self,
        point_clouds: Dict[Tuple[int, int], np.ndarray],
        verbose: bool = True
    ) -> Dict[Tuple[int, int], Dict]:
        """
        Compute persistent homology for multiple defensive formations.

        Args:
            point_clouds: Dictionary mapping (gameId, playId) to point clouds
            verbose: Whether to show progress bar

        Returns:
            Dictionary mapping play IDs to persistence results
        """
        results = {}

        iterator = tqdm(point_clouds.items(), desc="Computing persistence") if verbose else point_clouds.items()

        for play_id, point_cloud in iterator:
            results[play_id] = self.compute_persistence(point_cloud, play_id)

        self.persistence_diagrams = results
        return results

    def extract_features(
        self,
        persistence_result: Dict,
        dimension: int = 1
    ) -> Dict[str, float]:
        """
        Extract topological features from a persistence diagram.

        Args:
            persistence_result: Output from compute_persistence()
            dimension: Homology dimension (0 or 1)

        Returns:
            Dictionary of features:
                - num_features: Total number of topological features
                - max_persistence: Longest-lived feature
                - avg_persistence: Average persistence
                - total_persistence: Sum of all persistences
                - num_significant: Number of features with persistence > threshold
        """
        dgm = persistence_result['dgms'][dimension]

        if len(dgm) == 0:
            return {
                'num_features': 0,
                'max_persistence': 0.0,
                'avg_persistence': 0.0,
                'total_persistence': 0.0,
                'num_significant': 0,
                'persistence_entropy': 0.0
            }

        # Remove infinite points (for H0, the last component lives forever)
        finite_dgm = dgm[np.isfinite(dgm).all(axis=1)]

        if len(finite_dgm) == 0:
            return {
                'num_features': 0,
                'max_persistence': 0.0,
                'avg_persistence': 0.0,
                'total_persistence': 0.0,
                'num_significant': 0,
                'persistence_entropy': 0.0
            }

        # Compute persistence (death - birth)
        persistences = finite_dgm[:, 1] - finite_dgm[:, 0]

        # Features
        features = {
            'num_features': len(finite_dgm),
            'max_persistence': float(np.max(persistences)),
            'avg_persistence': float(np.mean(persistences)),
            'total_persistence': float(np.sum(persistences)),
            'num_significant': int(np.sum(persistences > 1.0)),  # Threshold = 1 yard
        }

        # Persistence entropy (diversity of feature scales)
        if len(persistences) > 0:
            normalized_pers = persistences / np.sum(persistences)
            # Avoid log(0)
            normalized_pers = normalized_pers[normalized_pers > 0]
            features['persistence_entropy'] = float(-np.sum(normalized_pers * np.log(normalized_pers)))
        else:
            features['persistence_entropy'] = 0.0

        return features

    def extract_features_batch(
        self,
        persistence_results: Optional[Dict[Tuple[int, int], Dict]] = None,
        dimensions: List[int] = [0, 1]
    ) -> pd.DataFrame:
        """
        Extract topological features for all plays.

        Args:
            persistence_results: Dictionary of persistence results (uses self.persistence_diagrams if None)
            dimensions: Which homology dimensions to extract features for

        Returns:
            DataFrame with one row per play and columns for each feature
        """
        if persistence_results is None:
            persistence_results = self.persistence_diagrams

        if not persistence_results:
            raise ValueError("No persistence diagrams computed yet. Run compute_persistence_batch first.")

        rows = []

        for play_id, result in persistence_results.items():
            row = {
                'gameId': play_id[0],
                'playId': play_id[1],
                'num_defenders': result['num_points']
            }

            for dim in dimensions:
                features = self.extract_features(result, dimension=dim)
                # Prefix features with dimension
                for key, value in features.items():
                    row[f'H{dim}_{key}'] = value

            rows.append(row)

        return pd.DataFrame(rows)

    def plot_persistence_diagram(
        self,
        persistence_result: Dict,
        dimensions: List[int] = [0, 1],
        ax: Optional[plt.Axes] = None,
        title: str = "Persistence Diagram",
        show_inf_line: bool = True
    ) -> plt.Axes:
        """
        Plot persistence diagram for a single play.

        Args:
            persistence_result: Output from compute_persistence()
            dimensions: Which dimensions to plot
            ax: Matplotlib axes (creates new if None)
            title: Plot title
            show_inf_line: Whether to show line indicating infinite persistence

        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))

        colors = ['blue', 'red', 'green']
        labels = ['$H_0$ (Components)', '$H_1$ (Loops)', '$H_2$ (Voids)']
        markers = ['o', 's', '^']

        max_val = 0

        for dim in dimensions:
            if dim >= len(persistence_result['dgms']):
                continue

            dgm = persistence_result['dgms'][dim]

            if len(dgm) == 0:
                continue

            # Separate finite and infinite points
            finite_mask = np.isfinite(dgm).all(axis=1)
            finite_dgm = dgm[finite_mask]

            if len(finite_dgm) > 0:
                ax.scatter(
                    finite_dgm[:, 0],
                    finite_dgm[:, 1],
                    c=colors[dim],
                    marker=markers[dim],
                    s=50,
                    alpha=0.6,
                    label=labels[dim],
                    edgecolors='black',
                    linewidth=0.5
                )
                max_val = max(max_val, np.max(finite_dgm))

            # Plot infinite points (if any)
            if show_inf_line and np.any(~finite_mask):
                inf_births = dgm[~finite_mask, 0]
                if len(inf_births) > 0:
                    y_inf = max_val * 1.2 if max_val > 0 else 10
                    ax.scatter(
                        inf_births,
                        [y_inf] * len(inf_births),
                        c=colors[dim],
                        marker=markers[dim],
                        s=50,
                        alpha=0.6,
                        edgecolors='black',
                        linewidth=2
                    )

        # Diagonal line (birth = death)
        if max_val > 0:
            ax.plot([0, max_val * 1.1], [0, max_val * 1.1], 'k--', alpha=0.3, linewidth=1, label='Birth = Death')

        ax.set_xlabel('Birth', fontsize=12)
        ax.set_ylabel('Death', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')

        return ax

    def plot_barcode(
        self,
        persistence_result: Dict,
        dimension: int = 1,
        ax: Optional[plt.Axes] = None,
        title: str = "Persistence Barcode"
    ) -> plt.Axes:
        """
        Plot barcode (alternative to persistence diagram).

        Args:
            persistence_result: Output from compute_persistence()
            dimension: Which dimension to plot
            ax: Matplotlib axes (creates new if None)
            title: Plot title

        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        dgm = persistence_result['dgms'][dimension]

        if len(dgm) == 0:
            ax.text(0.5, 0.5, 'No features in this dimension',
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(title)
            return ax

        # Sort by birth time
        sorted_indices = np.argsort(dgm[:, 0])
        dgm_sorted = dgm[sorted_indices]

        # Plot bars
        for i, (birth, death) in enumerate(dgm_sorted):
            if np.isinf(death):
                # Infinite bar - use arrow
                max_birth = np.max(dgm_sorted[np.isfinite(dgm_sorted[:, 1]), 1])
                death_plot = max_birth * 1.2
                ax.plot([birth, death_plot], [i, i], 'b-', linewidth=2)
                ax.arrow(death_plot, i, max_birth * 0.1, 0,
                        head_width=0.3, head_length=max_birth * 0.05, fc='b', ec='b')
            else:
                ax.plot([birth, death], [i, i], 'b-', linewidth=2)

        ax.set_xlabel('Distance', fontsize=12)
        ax.set_ylabel('Feature Index', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)

        return ax

    def save_persistence_diagrams(self, output_path: str) -> None:
        """Save all computed persistence diagrams to disk."""
        np.save(output_path, self.persistence_diagrams, allow_pickle=True)
        print(f"Saved {len(self.persistence_diagrams)} persistence diagrams to {output_path}")

    def load_persistence_diagrams(self, input_path: str) -> None:
        """Load persistence diagrams from disk."""
        self.persistence_diagrams = np.load(input_path, allow_pickle=True).item()
        print(f"Loaded {len(self.persistence_diagrams)} persistence diagrams from {input_path}")


def analyze_formation_topology(
    point_cloud: np.ndarray,
    play_id: Optional[Tuple[int, int]] = None,
    plot: bool = True
) -> Dict:
    """
    Convenience function to analyze a single formation.

    Args:
        point_cloud: Array of shape (n_defenders, 2)
        play_id: Optional play identifier
        plot: Whether to create visualization

    Returns:
        Dictionary with persistence results and features
    """
    tda = DefensiveCoverageTDA(max_dimension=1)
    result = tda.compute_persistence(point_cloud, play_id)

    # Extract features
    h0_features = tda.extract_features(result, dimension=0)
    h1_features = tda.extract_features(result, dimension=1)

    analysis = {
        'persistence': result,
        'H0_features': h0_features,
        'H1_features': h1_features
    }

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Persistence diagram
        tda.plot_persistence_diagram(
            result,
            dimensions=[0, 1],
            ax=axes[0],
            title=f"Persistence Diagram - Play {play_id}" if play_id else "Persistence Diagram"
        )

        # Barcode for H1 (loops/holes)
        tda.plot_barcode(
            result,
            dimension=1,
            ax=axes[1],
            title=f"$H_1$ Barcode - Play {play_id}" if play_id else "$H_1$ Barcode (Coverage Gaps)"
        )

        plt.tight_layout()
        plt.show()

    return analysis


if __name__ == "__main__":
    # Example usage
    print("TDA Analysis Module for NFL Defensive Formations")
    print("=" * 60)

    # Create example point cloud (simulated defensive formation)
    np.random.seed(42)
    defenders = np.random.rand(11, 2) * 20 + 40  # 11 defenders, field coords

    print("\nExample: Analyzing a simulated defensive formation")
    print(f"Point cloud shape: {defenders.shape}")

    result = analyze_formation_topology(defenders, play_id=(123, 456), plot=True)

    print("\nH0 Features (Connected Components):")
    for key, value in result['H0_features'].items():
        print(f"  {key}: {value:.3f}")

    print("\nH1 Features (Loops/Holes - Coverage Gaps):")
    for key, value in result['H1_features'].items():
        print(f"  {key}: {value:.3f}")

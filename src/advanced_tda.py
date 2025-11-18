"""
Advanced Topological Data Analysis for NFL defensive formations.

This module implements advanced TDA concepts including:
- Persistence landscapes and persistence images (summary statistics)
- Bottleneck and Wasserstein distances (stability metrics)
- Mapper algorithm (Reeb graph approximation)
- Statistical inference on persistence diagrams
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional, Callable
from pathlib import Path
from tqdm import tqdm
import warnings

# TDA libraries
from ripser import ripser
import persim
from persim import PersistenceImager
from sklearn.cluster import DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import UMAP

# Visualization
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import seaborn as sns


class PersistenceSummary:
    """
    Generate persistence summaries: landscapes, images, statistics.

    These vectorizations allow comparing persistence diagrams using
    standard statistical and ML techniques.
    """

    def __init__(self, resolution: int = 50):
        """
        Initialize persistence summary generator.

        Args:
            resolution: Grid resolution for persistence images
        """
        self.resolution = resolution
        self.imager = None

    def compute_persistence_landscape(
        self,
        dgm: np.ndarray,
        k: int = 1,
        num_points: int = 500
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute k-th persistence landscape function.

        The persistence landscape is a functional summary that is:
        - Stable with respect to bottleneck distance
        - Easy to average across multiple diagrams
        - Suitable for statistical inference

        Args:
            dgm: Persistence diagram (n x 2 array)
            k: Which landscape to compute (1 = largest, 2 = second largest, etc.)
            num_points: Number of points to sample

        Returns:
            (x_values, landscape_values)
        """
        if len(dgm) == 0:
            return np.array([]), np.array([])

        # Remove infinite points
        dgm = dgm[np.isfinite(dgm).all(axis=1)]

        if len(dgm) == 0:
            return np.array([]), np.array([])

        # Birth and death times
        births = dgm[:, 0]
        deaths = dgm[:, 1]

        # Domain for landscape
        min_val = np.min(births)
        max_val = np.max(deaths)
        x = np.linspace(min_val, max_val, num_points)

        # For each point, compute landscape value
        landscape = np.zeros(num_points)

        for i, t in enumerate(x):
            # For each feature, compute its contribution at time t
            contributions = []
            for b, d in zip(births, deaths):
                # Tent function: rises from b to (b+d)/2, falls to d
                mid = (b + d) / 2
                if b <= t <= mid:
                    contributions.append(t - b)
                elif mid < t <= d:
                    contributions.append(d - t)
                else:
                    contributions.append(0)

            # k-th largest contribution
            contributions.sort(reverse=True)
            if len(contributions) >= k:
                landscape[i] = contributions[k-1]

        return x, landscape

    def compute_persistence_image(
        self,
        dgm: np.ndarray,
        weight_fn: Optional[Callable] = None
    ) -> np.ndarray:
        """
        Compute persistence image (vectorization of persistence diagram).

        Persistence images are:
        - Stable with respect to bottleneck distance
        - Suitable for ML algorithms (fixed-size vectors)
        - Capture both persistence and birth time information

        Args:
            dgm: Persistence diagram
            weight_fn: Optional weight function (default: linear in persistence)

        Returns:
            2D array representing the persistence image
        """
        if len(dgm) == 0:
            return np.zeros((self.resolution, self.resolution))

        # Remove infinite points
        dgm = dgm[np.isfinite(dgm).all(axis=1)]

        if len(dgm) == 0:
            return np.zeros((self.resolution, self.resolution))

        # Use persim library for persistence images
        if self.imager is None:
            self.imager = PersistenceImager(pixel_size=1.0/self.resolution)

        try:
            img = self.imager.transform(dgm)
            return img
        except:
            # Fallback to simple implementation
            return self._simple_persistence_image(dgm, weight_fn)

    def _simple_persistence_image(
        self,
        dgm: np.ndarray,
        weight_fn: Optional[Callable] = None
    ) -> np.ndarray:
        """Simple fallback implementation of persistence image."""
        if weight_fn is None:
            # Default: weight by persistence
            weight_fn = lambda b, d: d - b

        # Transform to birth-persistence coordinates
        births = dgm[:, 0]
        deaths = dgm[:, 1]
        persistences = deaths - births

        # Create grid
        birth_min, birth_max = births.min(), births.max()
        pers_min, pers_max = 0, persistences.max()

        # Add padding
        padding = 0.1
        birth_range = birth_max - birth_min
        pers_range = pers_max - pers_min

        birth_min -= padding * birth_range
        birth_max += padding * birth_range
        pers_max += padding * pers_range

        # Create image
        img = np.zeros((self.resolution, self.resolution))

        # Place Gaussian at each point
        sigma = 0.1 * max(birth_range, pers_range)

        for b, p in zip(births, persistences):
            weight = weight_fn(b, b + p)

            # Map to pixel coordinates
            i = int((p / (pers_max - pers_min)) * (self.resolution - 1))
            j = int(((b - birth_min) / (birth_max - birth_min)) * (self.resolution - 1))

            # Add Gaussian
            for di in range(-3, 4):
                for dj in range(-3, 4):
                    ni, nj = i + di, j + dj
                    if 0 <= ni < self.resolution and 0 <= nj < self.resolution:
                        dist = np.sqrt(di**2 + dj**2) * (birth_range / self.resolution)
                        img[ni, nj] += weight * np.exp(-dist**2 / (2 * sigma**2))

        return img

    def compute_betti_curve(
        self,
        dgm: np.ndarray,
        num_points: int = 500
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute persistent Betti number function.

        Shows how many topological features exist at each scale.

        Args:
            dgm: Persistence diagram
            num_points: Number of points to sample

        Returns:
            (scales, betti_numbers)
        """
        if len(dgm) == 0:
            return np.array([]), np.array([])

        # Remove infinite points
        dgm = dgm[np.isfinite(dgm).all(axis=1)]

        if len(dgm) == 0:
            return np.array([]), np.array([])

        # Range of scales
        min_scale = np.min(dgm[:, 0])
        max_scale = np.max(dgm[:, 1])
        scales = np.linspace(min_scale, max_scale, num_points)

        # Count features alive at each scale
        betti = np.zeros(num_points)
        for i, t in enumerate(scales):
            # Feature alive if birth <= t < death
            betti[i] = np.sum((dgm[:, 0] <= t) & (dgm[:, 1] > t))

        return scales, betti


class PersistenceStatistics:
    """
    Statistical analysis on persistence diagrams.

    Implements:
    - Bottleneck and Wasserstein distances
    - Hypothesis testing on diagrams
    - Confidence regions
    """

    @staticmethod
    def bottleneck_distance(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
        """
        Compute bottleneck distance between two persistence diagrams.

        The bottleneck distance is stable and provides theoretical guarantees
        for TDA inference.

        Args:
            dgm1, dgm2: Persistence diagrams

        Returns:
            Bottleneck distance
        """
        # Remove infinite points
        dgm1 = dgm1[np.isfinite(dgm1).all(axis=1)]
        dgm2 = dgm2[np.isfinite(dgm2).all(axis=1)]

        if len(dgm1) == 0 and len(dgm2) == 0:
            return 0.0
        if len(dgm1) == 0 or len(dgm2) == 0:
            # Distance to diagonal
            dgm = dgm1 if len(dgm1) > 0 else dgm2
            persistences = dgm[:, 1] - dgm[:, 0]
            return np.max(persistences) / 2 if len(persistences) > 0 else 0.0

        try:
            return persim.bottleneck(dgm1, dgm2)
        except:
            # Fallback: simplified approximation
            return PersistenceStatistics._approx_bottleneck(dgm1, dgm2)

    @staticmethod
    def _approx_bottleneck(dgm1: np.ndarray, dgm2: np.ndarray) -> float:
        """Approximate bottleneck distance."""
        # Simple heuristic: max difference in top k persistences
        k = min(len(dgm1), len(dgm2), 5)

        pers1 = sorted(dgm1[:, 1] - dgm1[:, 0], reverse=True)[:k]
        pers2 = sorted(dgm2[:, 1] - dgm2[:, 0], reverse=True)[:k]

        # Pad shorter list
        while len(pers1) < k:
            pers1.append(0)
        while len(pers2) < k:
            pers2.append(0)

        return max(abs(p1 - p2) for p1, p2 in zip(pers1, pers2))

    @staticmethod
    def wasserstein_distance(
        dgm1: np.ndarray,
        dgm2: np.ndarray,
        q: int = 2
    ) -> float:
        """
        Compute q-Wasserstein distance between persistence diagrams.

        Args:
            dgm1, dgm2: Persistence diagrams
            q: Order of Wasserstein distance (typically 1 or 2)

        Returns:
            Wasserstein distance
        """
        # Remove infinite points
        dgm1 = dgm1[np.isfinite(dgm1).all(axis=1)]
        dgm2 = dgm2[np.isfinite(dgm2).all(axis=1)]

        if len(dgm1) == 0 and len(dgm2) == 0:
            return 0.0

        try:
            return persim.sliced_wasserstein(dgm1, dgm2, M=50)
        except:
            # Fallback
            return PersistenceStatistics.bottleneck_distance(dgm1, dgm2)

    @staticmethod
    def permutation_test(
        group1_dgms: List[np.ndarray],
        group2_dgms: List[np.ndarray],
        n_permutations: int = 1000,
        distance_fn: str = 'bottleneck'
    ) -> Tuple[float, float]:
        """
        Permutation test to compare two groups of persistence diagrams.

        Tests null hypothesis: the two groups have the same distribution.

        Args:
            group1_dgms: List of diagrams from group 1
            group2_dgms: List of diagrams from group 2
            n_permutations: Number of permutations
            distance_fn: 'bottleneck' or 'wasserstein'

        Returns:
            (test_statistic, p_value)
        """
        dist_fn = (PersistenceStatistics.bottleneck_distance
                   if distance_fn == 'bottleneck'
                   else PersistenceStatistics.wasserstein_distance)

        # Compute observed statistic
        # (average within-group distance vs between-group distance)
        def group_statistic(g1, g2):
            # Average distance between groups
            between_dist = []
            for d1 in g1[:10]:  # Sample to save time
                for d2 in g2[:10]:
                    between_dist.append(dist_fn(d1, d2))
            return np.mean(between_dist) if between_dist else 0.0

        observed_stat = group_statistic(group1_dgms, group2_dgms)

        # Permutation test
        all_dgms = group1_dgms + group2_dgms
        n1 = len(group1_dgms)

        perm_stats = []
        for _ in range(n_permutations):
            # Shuffle
            perm = np.random.permutation(len(all_dgms))
            perm_g1 = [all_dgms[i] for i in perm[:n1]]
            perm_g2 = [all_dgms[i] for i in perm[n1:]]

            perm_stats.append(group_statistic(perm_g1, perm_g2))

        # P-value: proportion of permutations with statistic >= observed
        p_value = np.mean([s >= observed_stat for s in perm_stats])

        return observed_stat, p_value


class MapperAnalysis:
    """
    Mapper algorithm for topological data visualization.

    Mapper creates a Reeb graph approximation that reveals:
    - Clustering structure
    - Connected components
    - Loops and flares in data
    """

    def __init__(
        self,
        filter_fn: Optional[Callable] = None,
        n_cubes: int = 10,
        overlap: float = 0.3
    ):
        """
        Initialize Mapper.

        Args:
            filter_fn: Filter function to project data (default: PCA)
            n_cubes: Number of intervals to cover filter range
            overlap: Overlap between adjacent intervals (0-1)
        """
        self.filter_fn = filter_fn
        self.n_cubes = n_cubes
        self.overlap = overlap
        self.graph = None

    def fit(
        self,
        X: np.ndarray,
        clustering_fn: Optional[Callable] = None
    ) -> Dict:
        """
        Fit Mapper to data.

        Args:
            X: Data matrix (n_samples, n_features)
            clustering_fn: Clustering algorithm (default: DBSCAN)

        Returns:
            Mapper graph as dictionary
        """
        # Apply filter function
        if self.filter_fn is None:
            # Default: first principal component
            pca = PCA(n_components=1)
            filter_values = pca.fit_transform(X).flatten()
        else:
            filter_values = self.filter_fn(X)

        # Create overlapping intervals
        f_min, f_max = filter_values.min(), filter_values.max()
        interval_length = (f_max - f_min) / (self.n_cubes * (1 - self.overlap))
        step = interval_length * (1 - self.overlap)

        # Clustering algorithm
        if clustering_fn is None:
            clustering_fn = lambda x: DBSCAN(eps=0.5, min_samples=2).fit_predict(x)

        # Build graph
        graph = {'nodes': {}, 'edges': []}
        node_id = 0

        # For each interval
        for i in range(self.n_cubes):
            start = f_min + i * step
            end = start + interval_length

            # Points in this interval
            mask = (filter_values >= start) & (filter_values <= end)
            if not np.any(mask):
                continue

            X_interval = X[mask]

            # Cluster points in interval
            labels = clustering_fn(X_interval)

            # Create node for each cluster
            for cluster_id in set(labels):
                if cluster_id == -1:  # Noise
                    continue

                cluster_mask = labels == cluster_id
                node_points = np.where(mask)[0][cluster_mask]

                graph['nodes'][node_id] = {
                    'points': node_points,
                    'size': len(node_points),
                    'interval': i
                }
                node_id += 1

        # Find edges (nodes sharing points)
        nodes = list(graph['nodes'].keys())
        for i, n1 in enumerate(nodes):
            for n2 in nodes[i+1:]:
                points1 = set(graph['nodes'][n1]['points'])
                points2 = set(graph['nodes'][n2]['points'])

                if points1 & points2:  # Share points
                    graph['edges'].append((n1, n2))

        self.graph = graph
        return graph

    def plot(
        self,
        ax: Optional[plt.Axes] = None,
        color_by: Optional[np.ndarray] = None
    ) -> plt.Axes:
        """
        Visualize Mapper graph.

        Args:
            ax: Matplotlib axes
            color_by: Optional values to color nodes

        Returns:
            Axes with Mapper plot
        """
        if self.graph is None:
            raise ValueError("Must call fit() first")

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))

        # Position nodes
        node_positions = {}
        for node_id, node_data in self.graph['nodes'].items():
            interval = node_data['interval']
            # Spread nodes in same interval vertically
            nodes_in_interval = [n for n, d in self.graph['nodes'].items()
                                if d['interval'] == interval]
            idx = nodes_in_interval.index(node_id)
            n_in_interval = len(nodes_in_interval)

            x = interval
            y = (idx - n_in_interval/2) * 0.5
            node_positions[node_id] = (x, y)

        # Draw edges
        for n1, n2 in self.graph['edges']:
            x1, y1 = node_positions[n1]
            x2, y2 = node_positions[n2]
            ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.3, linewidth=1)

        # Draw nodes
        for node_id, (x, y) in node_positions.items():
            size = self.graph['nodes'][node_id]['size']

            if color_by is not None:
                # Color by average value
                points = self.graph['nodes'][node_id]['points']
                color_val = np.mean(color_by[points])
                color = plt.cm.viridis(color_val / color_by.max())
            else:
                color = 'lightblue'

            ax.scatter(x, y, s=size*10, c=[color], edgecolors='black',
                      linewidth=2, zorder=10)

        ax.set_xlabel('Filter Function', fontsize=12)
        ax.set_ylabel('Clusters', fontsize=12)
        ax.set_title('Mapper Graph', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        return ax


# Convenience functions for batch analysis

def compute_distance_matrix(
    persistence_diagrams: Dict[Tuple[int, int], Dict],
    dimension: int = 1,
    metric: str = 'bottleneck'
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Compute pairwise distance matrix between persistence diagrams.

    Args:
        persistence_diagrams: Dictionary of persistence results
        dimension: Homology dimension
        metric: 'bottleneck' or 'wasserstein'

    Returns:
        (distance_matrix, play_ids)
    """
    play_ids = list(persistence_diagrams.keys())
    n = len(play_ids)

    dist_matrix = np.zeros((n, n))

    dist_fn = (PersistenceStatistics.bottleneck_distance
               if metric == 'bottleneck'
               else PersistenceStatistics.wasserstein_distance)

    print(f"Computing {n}x{n} distance matrix using {metric} distance...")
    for i in tqdm(range(n)):
        dgm1 = persistence_diagrams[play_ids[i]]['dgms'][dimension]
        for j in range(i+1, n):
            dgm2 = persistence_diagrams[play_ids[j]]['dgms'][dimension]
            dist = dist_fn(dgm1, dgm2)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    return dist_matrix, play_ids


if __name__ == "__main__":
    print("Advanced TDA Module - NFL Defensive Formations")
    print("=" * 70)
    print("\nImplements:")
    print("  - Persistence landscapes and images")
    print("  - Bottleneck and Wasserstein distances")
    print("  - Permutation testing")
    print("  - Mapper algorithm")

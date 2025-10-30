"""
Visualization functions for NFL defensive formations and TDA results.

This module provides:
- Field plot visualizations of defensive formations
- Persistence diagram plotting
- Validation plots for preprocessing
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from pathlib import Path


class NFLFieldVisualizer:
    """Visualize NFL defensive formations on field diagrams."""

    # NFL field dimensions
    FIELD_LENGTH = 120  # yards (including endzones)
    FIELD_WIDTH = 53.3  # yards

    def __init__(self, figsize: Tuple[int, int] = (12, 6)):
        """
        Initialize visualizer.

        Args:
            figsize: Figure size for plots
        """
        self.figsize = figsize

    def draw_field(self, ax: plt.Axes, show_yardlines: bool = True) -> None:
        """
        Draw NFL field background.

        Args:
            ax: Matplotlib axes to draw on
            show_yardlines: Whether to show yard line markers
        """
        # Field boundaries
        ax.set_xlim(0, self.FIELD_LENGTH)
        ax.set_ylim(0, self.FIELD_WIDTH)

        # Green field
        ax.add_patch(patches.Rectangle(
            (0, 0), self.FIELD_LENGTH, self.FIELD_WIDTH,
            facecolor='#2d5f3a', edgecolor='white', linewidth=2
        ))

        # Yard lines
        if show_yardlines:
            for yard in range(10, 110, 10):
                ax.axvline(yard, color='white', linewidth=1, alpha=0.5)

                # Yard numbers
                if yard >= 20 and yard <= 100:
                    yard_num = min(yard, 120 - yard) // 10
                    ax.text(yard, 5, str(yard_num * 10),
                           color='white', fontsize=10, ha='center', alpha=0.7)

        # Endzones
        ax.add_patch(patches.Rectangle(
            (0, 0), 10, self.FIELD_WIDTH,
            facecolor='#1a3d28', alpha=0.3
        ))
        ax.add_patch(patches.Rectangle(
            (110, 0), 10, self.FIELD_WIDTH,
            facecolor='#1a3d28', alpha=0.3
        ))

        # Line of scrimmage (will be customized per play)
        ax.axvline(50, color='yellow', linewidth=2, linestyle='--', alpha=0.7, label='LOS')

        ax.set_xlabel('Yards', fontsize=12, color='white')
        ax.set_ylabel('Yards', fontsize=12, color='white')
        ax.tick_params(colors='white')

    def plot_formation(
        self,
        point_cloud: np.ndarray,
        ax: Optional[plt.Axes] = None,
        title: str = "Defensive Formation",
        show_field: bool = True,
        player_color: str = 'red',
        player_size: int = 200,
        annotate_players: bool = False,
        player_names: Optional[List[str]] = None
    ) -> plt.Axes:
        """
        Plot a single defensive formation.

        Args:
            point_cloud: Array of shape (n_players, 2) with x, y coordinates
            ax: Matplotlib axes (creates new if None)
            title: Plot title
            show_field: Whether to draw field background
            player_color: Color for player markers
            player_size: Size of player markers
            annotate_players: Whether to number the players
            player_names: Optional list of player names to display

        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.figsize)
            fig.patch.set_facecolor('#2d5f3a')

        if show_field:
            self.draw_field(ax)

        # Plot defensive players
        ax.scatter(
            point_cloud[:, 0],
            point_cloud[:, 1],
            c=player_color,
            s=player_size,
            marker='o',
            edgecolors='white',
            linewidths=2,
            label='Defenders',
            zorder=10
        )

        # Annotate players with numbers or names
        if player_names is not None:
            # Display player names
            for i, (x, y) in enumerate(point_cloud):
                if i < len(player_names):
                    name = player_names[i]
                    # Shorten long names
                    if len(name) > 15:
                        parts = name.split()
                        name = f"{parts[0][0]}. {parts[-1]}" if len(parts) > 1 else name[:12]
                    ax.text(x, y+1.5, name, color='white', fontsize=7,
                           ha='center', va='bottom', fontweight='bold', zorder=11,
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor='none'))
        elif annotate_players:
            # Display player numbers
            for i, (x, y) in enumerate(point_cloud):
                ax.text(x, y, str(i+1), color='white', fontsize=8,
                       ha='center', va='center', fontweight='bold', zorder=11)

        ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=20)
        ax.legend(loc='upper right', facecolor='#2d5f3a', edgecolor='white',
                 labelcolor='white')

        return ax

    def plot_multiple_formations(
        self,
        point_clouds: Dict[Tuple[int, int], np.ndarray],
        n_samples: int = 6,
        random_seed: int = 42
    ) -> plt.Figure:
        """
        Plot multiple defensive formations in a grid.

        Args:
            point_clouds: Dictionary mapping (gameId, playId) to coordinates
            n_samples: Number of formations to plot
            random_seed: Random seed for sampling

        Returns:
            Matplotlib figure
        """
        np.random.seed(random_seed)

        # Sample random plays
        play_ids = list(point_clouds.keys())
        sampled_plays = np.random.choice(len(play_ids), size=min(n_samples, len(play_ids)), replace=False)

        # Create grid
        n_cols = 3
        n_rows = (n_samples + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        fig.patch.set_facecolor('#1a1a1a')
        axes = axes.flatten() if n_samples > 1 else [axes]

        # Plot each formation
        for i, play_idx in enumerate(sampled_plays):
            play_id = play_ids[play_idx]
            point_cloud = point_clouds[play_id]

            self.plot_formation(
                point_cloud,
                ax=axes[i],
                title=f"Play {play_id[0]}-{play_id[1]}\n({point_cloud.shape[0]} defenders)",
                player_size=150
            )

        # Hide extra subplots
        for i in range(n_samples, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        return fig

    def plot_formation_statistics(
        self,
        point_clouds: Dict[Tuple[int, int], np.ndarray]
    ) -> plt.Figure:
        """
        Plot statistical summaries of formations.

        Args:
            point_clouds: Dictionary of point clouds

        Returns:
            Matplotlib figure with statistics
        """
        # Calculate statistics
        n_defenders = [pc.shape[0] for pc in point_clouds.values()]
        x_coords = [pc[:, 0] for pc in point_clouds.values()]
        y_coords = [pc[:, 1] for pc in point_clouds.values()]

        # Calculate spread (std dev of positions)
        x_spread = [np.std(x) for x in x_coords]
        y_spread = [np.std(y) for y in y_coords]

        # Calculate mean positions
        x_mean = [np.mean(x) for x in x_coords]
        y_mean = [np.mean(y) for y in y_coords]

        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Defensive Formation Statistics', fontsize=16, fontweight='bold')

        # Number of defenders per play
        axes[0, 0].hist(n_defenders, bins=range(8, 15), edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Number of Defenders')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Defenders per Play')
        axes[0, 0].axvline(np.mean(n_defenders), color='red', linestyle='--',
                          label=f'Mean: {np.mean(n_defenders):.1f}')
        axes[0, 0].legend()

        # X-coordinate spread
        axes[0, 1].hist(x_spread, bins=30, edgecolor='black', alpha=0.7, color='blue')
        axes[0, 1].set_xlabel('X-coordinate Std Dev (yards)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Horizontal Spread of Defense')
        axes[0, 1].axvline(np.mean(x_spread), color='red', linestyle='--',
                          label=f'Mean: {np.mean(x_spread):.1f}')
        axes[0, 1].legend()

        # Y-coordinate spread
        axes[0, 2].hist(y_spread, bins=30, edgecolor='black', alpha=0.7, color='green')
        axes[0, 2].set_xlabel('Y-coordinate Std Dev (yards)')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Vertical Spread of Defense')
        axes[0, 2].axvline(np.mean(y_spread), color='red', linestyle='--',
                          label=f'Mean: {np.mean(y_spread):.1f}')
        axes[0, 2].legend()

        # Mean X position
        axes[1, 0].hist(x_mean, bins=30, edgecolor='black', alpha=0.7, color='orange')
        axes[1, 0].set_xlabel('Mean X Position (yards)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Average Defensive Depth')
        axes[1, 0].axvline(np.mean(x_mean), color='red', linestyle='--',
                          label=f'Mean: {np.mean(x_mean):.1f}')
        axes[1, 0].legend()

        # Mean Y position
        axes[1, 1].hist(y_mean, bins=30, edgecolor='black', alpha=0.7, color='purple')
        axes[1, 1].set_xlabel('Mean Y Position (yards)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Average Lateral Position')
        axes[1, 1].axvline(np.mean(y_mean), color='red', linestyle='--',
                          label=f'Mean: {np.mean(y_mean):.1f}')
        axes[1, 1].legend()

        # 2D scatter of mean positions
        axes[1, 2].scatter(x_mean, y_mean, alpha=0.3, s=20)
        axes[1, 2].set_xlabel('Mean X Position (yards)')
        axes[1, 2].set_ylabel('Mean Y Position (yards)')
        axes[1, 2].set_title('Mean Formation Centers')
        axes[1, 2].axhline(self.FIELD_WIDTH / 2, color='red', linestyle='--', alpha=0.5)
        axes[1, 2].set_xlim(0, self.FIELD_LENGTH)
        axes[1, 2].set_ylim(0, self.FIELD_WIDTH)

        plt.tight_layout()
        return fig


def validate_preprocessing(
    defensive_df: pd.DataFrame,
    point_clouds: Dict[Tuple[int, int], np.ndarray],
    output_dir: Optional[str] = None
) -> None:
    """
    Create validation plots to confirm preprocessing success.

    Args:
        defensive_df: Processed dataframe of defensive players
        point_clouds: Dictionary of point clouds
        output_dir: Directory to save plots (optional)
    """
    visualizer = NFLFieldVisualizer()

    print("Creating validation visualizations...")

    # Plot 1: Sample formations
    fig1 = visualizer.plot_multiple_formations(point_clouds, n_samples=6)
    fig1.suptitle('Sample Defensive Formations (Preprocessed)',
                  fontsize=16, fontweight='bold', color='white', y=0.995)

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        fig1.savefig(f"{output_dir}/sample_formations.png", dpi=150,
                    bbox_inches='tight', facecolor='#1a1a1a')
        print(f"Saved sample formations to {output_dir}/sample_formations.png")

    # Plot 2: Formation statistics
    fig2 = visualizer.plot_formation_statistics(point_clouds)

    if output_dir:
        fig2.savefig(f"{output_dir}/formation_statistics.png", dpi=150,
                    bbox_inches='tight')
        print(f"Saved formation statistics to {output_dir}/formation_statistics.png")

    # Plot 3: Coordinate distribution check
    fig3, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig3.suptitle('Coordinate Distribution Check', fontsize=14, fontweight='bold')

    # All x coordinates
    all_x = defensive_df['x'].values
    axes[0].hist(all_x, bins=50, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('X Coordinate (yards)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('X-coordinate Distribution\n(Should be normalized)')
    axes[0].axvline(60, color='red', linestyle='--', alpha=0.5, label='Midfield')
    axes[0].legend()

    # All y coordinates
    all_y = defensive_df['y'].values
    axes[1].hist(all_y, bins=50, edgecolor='black', alpha=0.7, color='green')
    axes[1].set_xlabel('Y Coordinate (yards)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Y-coordinate Distribution')
    axes[1].axvline(visualizer.FIELD_WIDTH / 2, color='red', linestyle='--',
                   alpha=0.5, label='Center')
    axes[1].legend()

    plt.tight_layout()

    if output_dir:
        fig3.savefig(f"{output_dir}/coordinate_distribution.png", dpi=150,
                    bbox_inches='tight')
        print(f"Saved coordinate distribution to {output_dir}/coordinate_distribution.png")

    print("\nValidation complete! Review the plots to confirm preprocessing.")
    print(f"Total plays processed: {len(point_clouds)}")
    print(f"Total defensive player positions: {len(defensive_df)}")

    # Show plots if not saving
    if not output_dir:
        plt.show()


if __name__ == "__main__":
    # Example: Load and validate preprocessed data
    import sys
    sys.path.append('..')
    from preprocessing import load_point_clouds

    # Load data
    defensive_df = pd.read_csv("../data/processed/defensive_formations.csv")
    point_clouds = load_point_clouds("../data/processed/point_clouds.npy")

    # Validate
    validate_preprocessing(defensive_df, point_clouds, output_dir="../results/figures")

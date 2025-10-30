"""
Data preprocessing for NFL Big Data Bowl 2021 tracking data.

This module handles:
- Loading and merging tracking data with play metadata
- Standardizing coordinate systems (rotating/flipping plays)
- Extracting key frames (ball release time)
- Filtering defensive players and creating point clouds
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from tqdm import tqdm


class NFLDataPreprocessor:
    """Preprocess NFL tracking data for topological analysis."""

    def __init__(self, data_dir: str):
        """
        Initialize preprocessor.

        Args:
            data_dir: Path to directory containing raw NFL data files
        """
        self.data_dir = Path(data_dir)
        self.tracking_data = None
        self.plays_data = None
        self.games_data = None
        self.players_data = None

    def load_data(self) -> None:
        """Load all necessary data files from the Big Data Bowl dataset."""
        print("Loading NFL Big Data Bowl 2021 dataset...")

        # Load games data
        games_path = self.data_dir / "games.csv"
        if games_path.exists():
            self.games_data = pd.read_csv(games_path)
            print(f"Loaded {len(self.games_data)} games")

        # Load plays data
        plays_path = self.data_dir / "plays.csv"
        if plays_path.exists():
            self.plays_data = pd.read_csv(plays_path)
            print(f"Loaded {len(self.plays_data)} plays")

        # Load players data
        players_path = self.data_dir / "players.csv"
        if players_path.exists():
            self.players_data = pd.read_csv(players_path)
            print(f"Loaded {len(self.players_data)} players")

        # Load tracking data (may be split by week)
        tracking_files = list(self.data_dir.glob("week*.csv"))
        if tracking_files:
            print(f"Loading {len(tracking_files)} tracking data files...")
            tracking_dfs = []
            for file in tqdm(tracking_files, desc="Loading weeks"):
                df = pd.read_csv(file)
                tracking_dfs.append(df)
            self.tracking_data = pd.concat(tracking_dfs, ignore_index=True)
            print(f"Loaded {len(self.tracking_data)} tracking records")
        else:
            # Try loading single tracking file
            tracking_path = self.data_dir / "tracking.csv"
            if tracking_path.exists():
                self.tracking_data = pd.read_csv(tracking_path)
                print(f"Loaded {len(self.tracking_data)} tracking records")

    def merge_data(self) -> pd.DataFrame:
        """
        Merge tracking data with play and game metadata.

        Returns:
            Merged dataframe with tracking and metadata
        """
        print("Merging tracking data with metadata...")

        if self.tracking_data is None:
            raise ValueError("Tracking data not loaded. Call load_data() first.")

        # Start with tracking data
        merged = self.tracking_data.copy()

        # Merge with plays data
        if self.plays_data is not None:
            merged = merged.merge(
                self.plays_data,
                on=['gameId', 'playId'],
                how='left'
            )

        # Merge with games data
        if self.games_data is not None:
            merged = merged.merge(
                self.games_data,
                on='gameId',
                how='left'
            )

        print(f"Merged dataset: {len(merged)} records")
        return merged

    def standardize_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize field coordinates so offense always moves left-to-right.

        This normalizes plays so that:
        - Offense always moves in positive x direction
        - Coordinates are consistent across all plays

        Args:
            df: DataFrame with tracking data including x, y, playDirection

        Returns:
            DataFrame with standardized coordinates
        """
        print("Standardizing field coordinates...")

        df = df.copy()

        # Flip coordinates if play goes left (playDirection == 'left')
        if 'playDirection' in df.columns:
            # For plays going left, flip x coordinate
            left_mask = df['playDirection'] == 'left'
            df.loc[left_mask, 'x'] = 120 - df.loc[left_mask, 'x']
            df.loc[left_mask, 'y'] = 53.3 - df.loc[left_mask, 'y']

            # Also flip direction and orientation angles if present
            if 'dir' in df.columns:
                df.loc[left_mask, 'dir'] = (df.loc[left_mask, 'dir'] + 180) % 360
            if 'o' in df.columns:
                df.loc[left_mask, 'o'] = (df.loc[left_mask, 'o'] + 180) % 360

        print("Coordinates standardized")
        return df

    def extract_ball_release_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract the frame closest to ball release time for each play.

        For pass plays, this identifies the moment the QB releases the ball,
        which is the critical snapshot for analyzing defensive coverage.

        Args:
            df: DataFrame with tracking data including event information

        Returns:
            DataFrame filtered to ball release frames only
        """
        print("Extracting ball release frames...")

        # Look for pass-related events
        # Common events: 'pass_forward', 'pass_arrived', 'pass_outcome_caught', etc.
        release_events = ['pass_forward', 'pass_released']

        if 'event' in df.columns:
            # Filter to rows where a pass release event occurred
            release_frames = df[df['event'].isin(release_events)].copy()

            if len(release_frames) == 0:
                print("Warning: No pass release events found. Using alternative method...")
                # Alternative: use first frame after snap for pass plays
                release_frames = self._extract_post_snap_frame(df)
        else:
            print("Warning: No event column found. Using alternative method...")
            release_frames = self._extract_post_snap_frame(df)

        # Get unique plays
        n_plays = release_frames[['gameId', 'playId']].drop_duplicates().shape[0]
        print(f"Extracted ball release frames for {n_plays} plays")

        return release_frames

    def _extract_post_snap_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Alternative method: extract frame shortly after snap.

        This is used when pass release events aren't available.
        We take a frame ~2 seconds after snap to allow pass rush to develop.
        """
        if 'event' in df.columns:
            # Find ball_snap event
            snap_frames = df[df['event'] == 'ball_snap'].copy()

            if len(snap_frames) > 0:
                # For each play, get the snap time
                snap_times = snap_frames.groupby(['gameId', 'playId'])['time'].first().reset_index()
                snap_times.columns = ['gameId', 'playId', 'snap_time']

                # Merge snap times back
                df_with_snap = df.merge(snap_times, on=['gameId', 'playId'], how='left')

                # Convert time strings to datetime if needed
                if df_with_snap['time'].dtype == 'object':
                    df_with_snap['time'] = pd.to_datetime(df_with_snap['time'])
                    df_with_snap['snap_time'] = pd.to_datetime(df_with_snap['snap_time'])

                # Calculate time since snap
                df_with_snap['time_since_snap'] = (
                    df_with_snap['time'] - df_with_snap['snap_time']
                ).dt.total_seconds()

                # Take frame approximately 2-3 seconds after snap
                target_time = 2.5
                release_approx = df_with_snap[
                    (df_with_snap['time_since_snap'] >= target_time - 0.5) &
                    (df_with_snap['time_since_snap'] <= target_time + 0.5)
                ]

                return release_approx

        # Fallback: just take middle frame of each play
        print("Using fallback method: middle frame of play")
        middle_frames = df.groupby(['gameId', 'playId']).apply(
            lambda x: x.iloc[len(x) // 2]
        ).reset_index(drop=True)

        return middle_frames

    def filter_defensive_players(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Filter to only defensive players for each play.

        Args:
            df: DataFrame with tracking data

        Returns:
            DataFrame containing only defensive players
        """
        print("Filtering to defensive players...")

        # Defensive players have team != possessionTeam
        if 'team' in df.columns and 'possessionTeam' in df.columns:
            # Filter out the football itself
            df_no_ball = df[df['team'] != 'football'].copy()

            # Keep players where team != possession team
            defensive = df_no_ball[
                df_no_ball['team'] != df_no_ball['possessionTeam']
            ].copy()

            n_plays = defensive[['gameId', 'playId']].drop_duplicates().shape[0]
            print(f"Filtered to defensive players for {n_plays} plays")

            return defensive
        else:
            print("Warning: Cannot determine defensive players. Columns missing.")
            return df

    def create_point_clouds(self, df: pd.DataFrame) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Create point cloud representations for each play's defensive formation.

        Args:
            df: DataFrame with defensive player positions

        Returns:
            Dictionary mapping (gameId, playId) to numpy array of shape (n_defenders, 2)
            containing (x, y) coordinates
        """
        print("Creating point clouds for defensive formations...")

        point_clouds = {}

        # Group by play
        for (game_id, play_id), group in tqdm(
            df.groupby(['gameId', 'playId']),
            desc="Processing plays"
        ):
            # Extract x, y coordinates
            coords = group[['x', 'y']].values

            # Store point cloud
            point_clouds[(game_id, play_id)] = coords

        print(f"Created {len(point_clouds)} point clouds")
        print(f"Average defenders per play: {np.mean([pc.shape[0] for pc in point_clouds.values()]):.1f}")

        return point_clouds

    def preprocess_pipeline(self, filter_pass_plays: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Run complete preprocessing pipeline.

        Args:
            filter_pass_plays: Whether to filter to pass plays only

        Returns:
            Tuple of (processed_dataframe, point_clouds_dict)
        """
        # Load data
        self.load_data()

        # Merge datasets
        merged = self.merge_data()

        # Filter to pass plays if requested
        if filter_pass_plays and 'passResult' in merged.columns:
            print("Filtering to pass plays only...")
            merged = merged[merged['passResult'].notna()].copy()
            print(f"Remaining plays: {merged[['gameId', 'playId']].drop_duplicates().shape[0]}")

        # Standardize coordinates
        merged = self.standardize_coordinates(merged)

        # Extract ball release frames
        release_frames = self.extract_ball_release_frame(merged)

        # Filter to defensive players
        defensive = self.filter_defensive_players(release_frames)

        # Create point clouds
        point_clouds = self.create_point_clouds(defensive)

        print("\nPreprocessing complete!")
        return defensive, point_clouds


def save_point_clouds(point_clouds: Dict, output_path: str) -> None:
    """
    Save point clouds to disk.

    Args:
        point_clouds: Dictionary of point clouds
        output_path: Path to save file
    """
    np.save(output_path, point_clouds, allow_pickle=True)
    print(f"Saved point clouds to {output_path}")


def load_point_clouds(input_path: str) -> Dict:
    """
    Load point clouds from disk.

    Args:
        input_path: Path to saved file

    Returns:
        Dictionary of point clouds
    """
    point_clouds = np.load(input_path, allow_pickle=True).item()
    print(f"Loaded {len(point_clouds)} point clouds from {input_path}")
    return point_clouds


if __name__ == "__main__":
    # Example usage
    preprocessor = NFLDataPreprocessor("data/raw")
    defensive_df, point_clouds = preprocessor.preprocess_pipeline()

    # Save processed data
    defensive_df.to_csv("data/processed/defensive_formations.csv", index=False)
    save_point_clouds(point_clouds, "data/processed/point_clouds.npy")

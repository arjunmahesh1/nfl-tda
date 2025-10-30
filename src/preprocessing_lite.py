"""
Memory-efficient preprocessing for NFL Big Data Bowl 2021 tracking data.

This module is optimized for systems with limited RAM by:
- Processing one week at a time
- Keeping only essential columns
- Immediate filtering to reduce data size
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List, Optional
from tqdm import tqdm


class NFLDataPreprocessorLite:
    """Memory-efficient preprocessor for NFL tracking data."""

    def __init__(self, data_dir: str):
        """
        Initialize preprocessor.

        Args:
            data_dir: Path to directory containing raw NFL data files
        """
        self.data_dir = Path(data_dir)
        self.plays_data = None
        self.games_data = None

    def load_metadata(self) -> None:
        """Load only the metadata files (plays and games - these are small)."""
        print("Loading metadata files...")

        # Load plays data
        plays_path = self.data_dir / "plays.csv"
        if plays_path.exists():
            # First, check what columns are available
            print("  Checking available columns...")
            sample = pd.read_csv(plays_path, nrows=1)
            available_cols = set(sample.columns)

            # Define columns we want (if they exist)
            desired_plays_cols = [
                'gameId', 'playId', 'playDescription', 'quarter', 'down', 'yardsToGo',
                'possessionTeam', 'defensiveTeam', 'yardlineSide', 'yardlineNumber',
                'offenseFormation', 'personnelO', 'defendersInTheBox', 'numberOfPassRushers',
                'personnelD', 'typeDropback', 'preSnapVisitorScore', 'preSnapHomeScore',
                'passResult', 'offensePlayResult', 'playResult', 'epa', 'isDefensivePI'
            ]

            # Only keep columns that actually exist
            essential_plays_cols = [col for col in desired_plays_cols if col in available_cols]

            # Always need these core columns
            required_cols = ['gameId', 'playId', 'possessionTeam', 'passResult']
            missing_required = [col for col in required_cols if col not in available_cols]
            if missing_required:
                print(f"  ERROR: Missing required columns: {missing_required}")
                print(f"  Available columns: {list(available_cols)}")
                return

            print(f"  Loading {len(essential_plays_cols)} columns...")
            self.plays_data = pd.read_csv(plays_path, usecols=essential_plays_cols)

            # Filter to pass plays only (reduces size significantly)
            self.plays_data = self.plays_data[self.plays_data['passResult'].notna()].copy()
            print(f"  Loaded {len(self.plays_data)} pass plays")
        else:
            print("Warning: plays.csv not found")

        # Load games data
        games_path = self.data_dir / "games.csv"
        if games_path.exists():
            self.games_data = pd.read_csv(games_path)
            print(f"Loaded {len(self.games_data)} games")
        else:
            print("Warning: games.csv not found")

    def process_week(
        self,
        week_number: int,
        extract_event: str = 'pass_forward'
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Process a single week of tracking data.

        Args:
            week_number: Week number (1-17)
            extract_event: Event to extract frame for ('pass_forward', 'pass_arrived', etc.)

        Returns:
            Tuple of (defensive_df, point_clouds_dict)
        """
        week_file = self.data_dir / f"week{week_number}.csv"

        if not week_file.exists():
            print(f"Warning: {week_file} not found")
            return pd.DataFrame(), {}

        print(f"\nProcessing Week {week_number}...")

        # Check what columns are available in tracking data
        print(f"  Checking available columns...")
        sample = pd.read_csv(week_file, nrows=1)
        available_cols = set(sample.columns)

        # Define columns we want
        desired_tracking_cols = [
            'gameId', 'playId', 'nflId', 'displayName', 'frameId',
            'time', 'jerseyNumber', 'team', 'playDirection',
            'x', 'y', 's', 'a', 'dis', 'o', 'dir', 'event'
        ]

        # Only use columns that exist
        essential_tracking_cols = [col for col in desired_tracking_cols if col in available_cols]

        # Core required columns
        required_tracking = ['gameId', 'playId', 'team', 'x', 'y']
        missing_required = [col for col in required_tracking if col not in available_cols]
        if missing_required:
            print(f"  ERROR: Missing required columns: {missing_required}")
            print(f"  Available columns: {list(available_cols)}")
            return pd.DataFrame(), {}

        print(f"  Loading tracking data ({len(essential_tracking_cols)} columns)...")
        tracking = pd.read_csv(week_file, usecols=essential_tracking_cols)
        print(f"  Loaded {len(tracking)} tracking records")

        # Filter to pass plays immediately
        if self.plays_data is not None:
            pass_play_ids = set(zip(self.plays_data['gameId'], self.plays_data['playId']))
            tracking = tracking[
                tracking.apply(lambda row: (row['gameId'], row['playId']) in pass_play_ids, axis=1)
            ].copy()
            print(f"  Filtered to {len(tracking)} records from pass plays")

        # Standardize coordinates
        print(f"  Standardizing coordinates...")
        tracking = self._standardize_coordinates(tracking)

        # Extract ball release frames
        print(f"  Extracting ball release frames...")
        release_frames = self._extract_event_frame(tracking, extract_event)
        print(f"  Extracted {len(release_frames)} frames")

        # Filter to defensive players
        print(f"  Filtering to defensive players...")
        defensive = self._filter_defensive_players(release_frames)
        print(f"  {len(defensive)} defensive player positions")

        # Merge with play metadata (only essential columns)
        if self.plays_data is not None and len(defensive) > 0:
            essential_play_cols = ['gameId', 'playId', 'passResult', 'down', 'yardsToGo',
                                   'offenseFormation', 'defendersInTheBox']
            defensive = defensive.merge(
                self.plays_data[essential_play_cols],
                on=['gameId', 'playId'],
                how='left'
            )

        # Create point clouds
        print(f"  Creating point clouds...")
        point_clouds = self._create_point_clouds(defensive)
        print(f"  Created {len(point_clouds)} point clouds")

        return defensive, point_clouds

    def _standardize_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize field coordinates."""
        df = df.copy()

        if 'playDirection' in df.columns:
            left_mask = df['playDirection'] == 'left'
            df.loc[left_mask, 'x'] = 120 - df.loc[left_mask, 'x']
            df.loc[left_mask, 'y'] = 53.3 - df.loc[left_mask, 'y']

            if 'dir' in df.columns:
                df.loc[left_mask, 'dir'] = (df.loc[left_mask, 'dir'] + 180) % 360
            if 'o' in df.columns:
                df.loc[left_mask, 'o'] = (df.loc[left_mask, 'o'] + 180) % 360

        return df

    def _extract_event_frame(self, df: pd.DataFrame, event_name: str) -> pd.DataFrame:
        """Extract frames where a specific event occurs."""
        if 'event' not in df.columns or 'frameId' not in df.columns:
            # Fallback: take middle frame of each play
            return df.groupby(['gameId', 'playId']).apply(
                lambda x: x.iloc[len(x) // 2]
            ).reset_index(drop=True)

        # For each play, find the frameId where the event occurs
        event_data = df[df['event'] == event_name][['gameId', 'playId', 'frameId']].drop_duplicates()

        if len(event_data) == 0:
            print(f"    Warning: No '{event_name}' events found, using alternative method...")
            # Try pass_arrived as backup
            event_data = df[df['event'] == 'pass_arrived'][['gameId', 'playId', 'frameId']].drop_duplicates()

            if len(event_data) == 0:
                # Last resort: middle frame
                return df.groupby(['gameId', 'playId']).apply(
                    lambda x: x.iloc[len(x) // 2]
                ).reset_index(drop=True)

        # Get the first frame for each play where event occurs (in case multiple)
        event_frames_by_play = event_data.groupby(['gameId', 'playId']).first().reset_index()

        # Now extract ALL player positions at those specific frames
        event_frames = df.merge(
            event_frames_by_play,
            on=['gameId', 'playId', 'frameId'],
            how='inner'
        )

        return event_frames

    def _filter_defensive_players(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter to only defensive players."""
        if 'team' not in df.columns:
            return df

        # Remove football
        df = df[df['team'] != 'football'].copy()

        # Get plays data to determine possession
        if self.plays_data is not None and self.games_data is not None:
            # Merge game info to get home/away team names
            game_info = self.games_data[['gameId', 'homeTeamAbbr', 'visitorTeamAbbr']].drop_duplicates()
            df = df.merge(game_info, on='gameId', how='left')

            # Merge possession info
            possession_info = self.plays_data[['gameId', 'playId', 'possessionTeam']].drop_duplicates()
            df = df.merge(possession_info, on=['gameId', 'playId'], how='left')

            # Map home/away to actual team names
            df['actual_team'] = df.apply(
                lambda row: row['homeTeamAbbr'] if row['team'] == 'home' else row['visitorTeamAbbr'],
                axis=1
            )

            # Keep defenders (actual_team != possession)
            df = df[df['actual_team'] != df['possessionTeam']].copy()

            # Clean up temporary columns
            df = df.drop(columns=['homeTeamAbbr', 'visitorTeamAbbr', 'actual_team'], errors='ignore')

        return df

    def _create_point_clouds(self, df: pd.DataFrame) -> Dict[Tuple[int, int], np.ndarray]:
        """Create point cloud representations."""
        point_clouds = {}

        for (game_id, play_id), group in df.groupby(['gameId', 'playId']):
            # Remove duplicates - keep only one position per unique player
            # Use nflId if available, otherwise use jerseyNumber + team
            if 'nflId' in group.columns:
                group_dedup = group.drop_duplicates(subset=['nflId'], keep='first')
            elif 'jerseyNumber' in group.columns:
                group_dedup = group.drop_duplicates(subset=['jerseyNumber', 'team'], keep='first')
            else:
                # Fallback: just use first occurrence of each unique position
                group_dedup = group.drop_duplicates(subset=['x', 'y'], keep='first')

            coords = group_dedup[['x', 'y']].values
            point_clouds[(game_id, play_id)] = coords

        return point_clouds

    def preprocess_multiple_weeks(
        self,
        weeks: List[int] = None,
        extract_event: str = 'pass_forward'
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Process multiple weeks efficiently.

        Args:
            weeks: List of week numbers to process (default: all available)
            extract_event: Event to extract frame for

        Returns:
            Tuple of (combined_defensive_df, combined_point_clouds)
        """
        # Load metadata first
        self.load_metadata()

        # Determine which weeks to process
        if weeks is None:
            available_weeks = sorted([
                int(f.stem.replace('week', ''))
                for f in self.data_dir.glob('week*.csv')
            ])
            weeks = available_weeks

        print(f"\nProcessing {len(weeks)} weeks: {weeks}")

        all_defensive_dfs = []
        all_point_clouds = {}

        for week in weeks:
            defensive_df, point_clouds = self.process_week(week, extract_event)

            if len(defensive_df) > 0:
                all_defensive_dfs.append(defensive_df)
                all_point_clouds.update(point_clouds)

        # Combine all weeks
        if all_defensive_dfs:
            combined_df = pd.concat(all_defensive_dfs, ignore_index=True)
        else:
            combined_df = pd.DataFrame()

        print(f"\n{'='*60}")
        print(f"TOTAL: {len(combined_df)} defensive positions from {len(all_point_clouds)} plays")
        print(f"{'='*60}")

        return combined_df, all_point_clouds


def save_point_clouds(point_clouds: Dict, output_path: str) -> None:
    """Save point clouds to disk."""
    np.save(output_path, point_clouds, allow_pickle=True)
    print(f"Saved point clouds to {output_path}")


def load_point_clouds(input_path: str) -> Dict:
    """Load point clouds from disk."""
    point_clouds = np.load(input_path, allow_pickle=True).item()
    print(f"Loaded {len(point_clouds)} point clouds from {input_path}")
    return point_clouds


if __name__ == "__main__":
    # Example: Process just a few weeks
    preprocessor = NFLDataPreprocessorLite("../data/raw")

    # Process weeks 1-3 (change this to process more/fewer weeks)
    defensive_df, point_clouds = preprocessor.preprocess_multiple_weeks(weeks=[1, 2, 3])

    # Save processed data
    defensive_df.to_csv("../data/processed/defensive_formations.csv", index=False)
    save_point_clouds(point_clouds, "../data/processed/point_clouds.npy")

    print("\nPreprocessing complete!")
    print(f"Average defenders per play: {np.mean([pc.shape[0] for pc in point_clouds.values()]):.1f}")

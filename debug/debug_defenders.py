"""
Debug script to understand why we're getting more than 11 defenders.
"""

import sys
sys.path.append('src')

import pandas as pd
from preprocessing_lite import NFLDataPreprocessorLite

print("Debugging Defender Count Issue")
print("=" * 70)

# Initialize
preprocessor = NFLDataPreprocessorLite("data/raw")
preprocessor.load_metadata()

# Process week 1
defensive_df, point_clouds = preprocessor.process_week(1)

# Find a play with more than 11 defenders
problem_plays = [(k, v.shape[0]) for k, v in point_clouds.items() if v.shape[0] > 11]
print(f"\nPlays with > 11 defenders: {len(problem_plays)} out of {len(point_clouds)}")
print(f"Average defenders: {sum([v.shape[0] for v in point_clouds.values()]) / len(point_clouds):.1f}")

if problem_plays:
    # Examine first problem play in detail
    problem_play_id, num_defenders = problem_plays[0]
    print(f"\n" + "=" * 70)
    print(f"EXAMINING: Game {problem_play_id[0]}, Play {problem_play_id[1]}")
    print(f"Number of defenders: {num_defenders}")
    print("=" * 70)

    # Get full details for this play
    play_data = defensive_df[
        (defensive_df['gameId'] == problem_play_id[0]) &
        (defensive_df['playId'] == problem_play_id[1])
    ]

    print(f"\nRows in defensive_df: {len(play_data)}")

    if 'nflId' in play_data.columns:
        unique_players = play_data['nflId'].nunique()
        print(f"Unique nflIds: {unique_players}")

        # Check for duplicates
        dup_check = play_data.groupby('nflId').size()
        duplicates = dup_check[dup_check > 1]
        if len(duplicates) > 0:
            print(f"\nPlayers appearing multiple times:")
            for nflId, count in duplicates.items():
                print(f"  nflId {nflId}: {count} times")
                player_rows = play_data[play_data['nflId'] == nflId]
                print(f"    Positions: {player_rows[['x', 'y']].values}")

    if 'frameId' in play_data.columns:
        unique_frames = play_data['frameId'].nunique()
        print(f"\nUnique frameIds: {unique_frames}")
        if unique_frames > 1:
            print("  ERROR: Multiple frames detected!")
            print(f"  Frame IDs: {sorted(play_data['frameId'].unique())}")

    if 'jerseyNumber' in play_data.columns:
        unique_jerseys = play_data['jerseyNumber'].nunique()
        print(f"Unique jersey numbers: {unique_jerseys}")

    # Show the data
    print(f"\nData for this play:")
    cols_to_show = ['nflId', 'frameId', 'jerseyNumber', 'displayName', 'x', 'y']
    cols_available = [c for c in cols_to_show if c in play_data.columns]
    print(play_data[cols_available].to_string())

# Check a play with exactly 11
good_plays = [(k, v.shape[0]) for k, v in point_clouds.items() if v.shape[0] == 11]
if good_plays:
    print(f"\n" + "=" * 70)
    print(f"GOOD: {len(good_plays)} plays have exactly 11 defenders")
    good_play_id, _ = good_plays[0]
    print(f"Example: Game {good_play_id[0]}, Play {good_play_id[1]}")

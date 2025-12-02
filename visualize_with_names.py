"""
Visualize a defensive formation with player names
"""

import sys
sys.path.append('src')

import pandas as pd
import matplotlib.pyplot as plt
from preprocessing_lite import NFLDataPreprocessorLite
from visualization import NFLFieldVisualizer

print("Loading data and preprocessing...")
preprocessor = NFLDataPreprocessorLite("data/raw")
preprocessor.load_metadata()

# Process week 1
defensive_df, point_clouds = preprocessor.process_week(1)

# Find a play with more than 11 defenders
problem_plays = [(k, v.shape[0]) for k, v in point_clouds.items() if v.shape[0] > 11]

if problem_plays:
    # Pick first problem play
    play_id, num_defenders = problem_plays[0]
    print(f"\nVisualizing: Game {play_id[0]}, Play {play_id[1]}")
    print(f"Number of 'defenders': {num_defenders}")

    # Get the formation
    formation = point_clouds[play_id]

    # Get player names for this play
    play_data = defensive_df[
        (defensive_df['gameId'] == play_id[0]) &
        (defensive_df['playId'] == play_id[1])
    ].copy()

    # Sort by x, y to match the point cloud order
    play_data = play_data.sort_values(['x', 'y']).reset_index(drop=True)

    # Extract names and teams
    if 'displayName' in play_data.columns:
        names = play_data['displayName'].tolist()
    else:
        names = [f"Player {i+1}" for i in range(len(play_data))]

    if 'team' in play_data.columns:
        teams = play_data['team'].tolist()
        print(f"\nTeams present: {set(teams)}")

    print(f"\nPlayers labeled as 'defenders':")
    for i, row in play_data.iterrows():
        cols_to_show = ['displayName', 'team', 'jerseyNumber']
        if 'position' in play_data.columns:
            cols_to_show.append('position')
        print(f"  {i+1}. ", end="")
        for col in cols_to_show:
            if col in row:
                print(f"{col}: {row[col]}", end="  ")
        print()

    # Visualize
    visualizer = NFLFieldVisualizer()
    fig, ax = plt.subplots(figsize=(16, 8))

    visualizer.plot_formation(
        formation,
        ax=ax,
        title=f"Formation with Names - Game {play_id[0]}, Play {play_id[1]}\n({num_defenders} players marked as 'defenders')",
        player_names=names,
        player_size=350
    )

    plt.tight_layout()
    plt.savefig("results/figures/formation_with_names.png", dpi=150, bbox_inches='tight', facecolor='#2d5f3a')
    print(f"\nSaved visualization to: results/figures/formation_with_names.png")
    plt.show()

    # Check possession
    if 'possessionTeam' in play_data.columns:
        poss_team = play_data['possessionTeam'].iloc[0]
        print(f"\nPossession team: {poss_team}")
        print(f"Players from possession team in 'defense': {sum([t == poss_team for t in teams])}")

else:
    print("All plays have exactly 11 defenders!")

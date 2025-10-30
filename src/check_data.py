"""
Diagnostic script to check what columns are available in your dataset.

Run this first for column-related errors.
"""

import pandas as pd
from pathlib import Path
import sys


def check_data_files(data_dir: str = "../data/raw"):
    """Check what files and columns are available."""

    data_path = Path(data_dir)

    print("=" * 70)
    print("NFL DATA DIAGNOSTIC CHECK")
    print("=" * 70)

    # Check for files
    print("\n1. CHECKING FOR DATA FILES...")
    print("-" * 70)

    files_to_check = {
        'games.csv': 'Game metadata',
        'plays.csv': 'Play information',
        'players.csv': 'Player roster',
        'week1.csv': 'Week 1 tracking data',
    }

    files_found = {}
    for filename, description in files_to_check.items():
        filepath = data_path / filename
        exists = filepath.exists()
        files_found[filename] = exists
        status = "[FOUND]" if exists else "[MISSING]"
        print(f"  {status}: {filename} ({description})")

    # Check for all week files
    week_files = list(data_path.glob("week*.csv"))
    print(f"\n  Found {len(week_files)} week files total")

    # Check columns in plays.csv
    print("\n2. CHECKING PLAYS.CSV COLUMNS...")
    print("-" * 70)

    if files_found.get('plays.csv'):
        try:
            plays = pd.read_csv(data_path / 'plays.csv', nrows=1)
            print(f"  Total columns: {len(plays.columns)}")
            print(f"\n  Available columns:")
            for i, col in enumerate(sorted(plays.columns), 1):
                print(f"    {i:2}. {col}")

            # Check for key columns
            print(f"\n  Key columns check:")
            key_cols = ['gameId', 'playId', 'possessionTeam', 'passResult',
                       'down', 'yardsToGo', 'offenseFormation']
            for col in key_cols:
                status = "[OK]" if col in plays.columns else "[MISSING]"
                print(f"    {status} {col}")

        except Exception as e:
            print(f"  Error reading plays.csv: {e}")
    else:
        print("  plays.csv not found - cannot check columns")

    # Check columns in week1.csv
    print("\n3. CHECKING TRACKING DATA COLUMNS (week1.csv)...")
    print("-" * 70)

    if files_found.get('week1.csv'):
        try:
            tracking = pd.read_csv(data_path / 'week1.csv', nrows=1)
            print(f"  Total columns: {len(tracking.columns)}")
            print(f"\n  Available columns:")
            for i, col in enumerate(sorted(tracking.columns), 1):
                print(f"    {i:2}. {col}")

            # Check for key columns
            print(f"\n  Key columns check:")
            key_cols = ['gameId', 'playId', 'team', 'x', 'y', 'event',
                       'playDirection', 'frameId']
            for col in key_cols:
                status = "[OK]" if col in tracking.columns else "[MISSING]"
                print(f"    {status} {col}")

        except Exception as e:
            print(f"  Error reading week1.csv: {e}")
    else:
        print("  week1.csv not found - cannot check columns")

    # Check data sizes
    print("\n4. CHECKING DATA SIZES...")
    print("-" * 70)

    if files_found.get('plays.csv'):
        try:
            plays = pd.read_csv(data_path / 'plays.csv')
            pass_plays = plays[plays['passResult'].notna()]
            print(f"  Total plays: {len(plays):,}")
            print(f"  Pass plays: {len(pass_plays):,}")
        except Exception as e:
            print(f"  Could not count plays: {e}")

    if files_found.get('week1.csv'):
        try:
            tracking = pd.read_csv(data_path / 'week1.csv')
            print(f"  Week 1 tracking records: {len(tracking):,}")
            print(f"  Week 1 unique plays: {tracking['playId'].nunique():,}")
        except Exception as e:
            print(f"  Could not count tracking records: {e}")

    # Summary
    print("\n5. SUMMARY")
    print("=" * 70)

    all_files_found = all(files_found.values())
    if all_files_found:
        print("[SUCCESS] All data files found!")
        print("\nYou should be able to run the preprocessing pipeline.")
        print("Use: jupyter notebook notebooks/01_data_preprocessing_lite.ipynb")
    else:
        print("[WARNING] Some data files are missing.")
        print("\nPlease download the NFL Big Data Bowl 2021 dataset:")
        print("  https://www.kaggle.com/c/nfl-big-data-bowl-2021/data")
        print("\nExtract all files to: data/raw/")

    print("=" * 70)


if __name__ == "__main__":
    # Allow custom data directory
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "../data/raw"
    check_data_files(data_dir)

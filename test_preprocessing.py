"""
Quick test of the preprocessing_lite module.
Run this to verify everything works before using the full notebook.
"""

import sys
sys.path.append('src')

from preprocessing_lite import NFLDataPreprocessorLite

print("Testing NFL Data Preprocessing (Lite Version)")
print("=" * 60)

# Initialize
print("\n1. Initializing preprocessor...")
preprocessor = NFLDataPreprocessorLite("data/raw")

# Load metadata
print("\n2. Loading metadata (plays and games)...")
preprocessor.load_metadata()

if preprocessor.plays_data is None:
    print("ERROR: Could not load plays data. Check data/raw/ directory.")
    sys.exit(1)

print(f"   Success! Loaded {len(preprocessor.plays_data)} pass plays")

# Process just week 1 as a test
print("\n3. Processing Week 1 (test)...")
defensive_df, point_clouds = preprocessor.process_week(1)

if len(point_clouds) == 0:
    print("ERROR: No point clouds created.")
    sys.exit(1)

print(f"   Success! Created {len(point_clouds)} defensive formations")

# Show statistics
import numpy as np
avg_defenders = np.mean([pc.shape[0] for pc in point_clouds.values()])
print(f"   Average defenders per play: {avg_defenders:.1f}")

# Show sample
sample_play_id = list(point_clouds.keys())[0]
sample_formation = point_clouds[sample_play_id]
print(f"\n4. Sample formation (Game {sample_play_id[0]}, Play {sample_play_id[1]}):")
print(f"   Shape: {sample_formation.shape}")
print(f"   First 3 defenders:")
print(sample_formation[:3])

print("\n" + "=" * 60)
print("TEST PASSED! You can now use the full notebooks.")
print("=" * 60)
print("\nNext step:")
print("  jupyter notebook notebooks/01_data_preprocessing_lite.ipynb")

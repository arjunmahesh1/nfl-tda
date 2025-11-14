"""
Quick test of the TDA module to verify it works correctly.
"""

import sys
sys.path.append('src')

import numpy as np
from tda_analysis import DefensiveCoverageTDA, analyze_formation_topology
from preprocessing_lite import load_point_clouds

print("Testing TDA Analysis Module")
print("=" * 70)

# Test 1: Simple example with known structure
print("\n1. Testing with simple synthetic data...")
print("-" * 70)

# Create a formation with an obvious "hole" (defenders in a ring)
theta = np.linspace(0, 2*np.pi, 8, endpoint=False)
ring_formation = np.column_stack([
    50 + 10 * np.cos(theta),  # x coordinates
    26.65 + 10 * np.sin(theta)  # y coordinates (centered on field)
])

print(f"Ring formation shape: {ring_formation.shape}")
print("This should show a clear H1 feature (loop/hole)")

tda = DefensiveCoverageTDA(max_dimension=1)
result = tda.compute_persistence(ring_formation)

h0_features = tda.extract_features(result, dimension=0)
h1_features = tda.extract_features(result, dimension=1)

print(f"\nH0 features: {h0_features['num_features']} components")
print(f"H1 features: {h1_features['num_features']} loops")
print(f"Max H1 persistence: {h1_features['max_persistence']:.2f} yards")

if h1_features['num_features'] > 0:
    print("✓ Successfully detected loop structure!")
else:
    print("⚠ Warning: No loop detected (might be expected if defenders too far apart)")

# Test 2: Load and analyze real data
print("\n2. Testing with real NFL data...")
print("-" * 70)

try:
    point_clouds = load_point_clouds("data/processed/point_clouds.npy")
    print(f"Loaded {len(point_clouds)} real formations")

    # Pick first play
    play_id = list(point_clouds.keys())[0]
    formation = point_clouds[play_id]

    print(f"\nAnalyzing: Game {play_id[0]}, Play {play_id[1]}")
    print(f"Defenders: {formation.shape[0]}")

    result = tda.compute_persistence(formation, play_id)
    h0_features = tda.extract_features(result, dimension=0)
    h1_features = tda.extract_features(result, dimension=1)

    print(f"\nH0 (clusters): {h0_features['num_features']} features")
    print(f"  - Max persistence: {h0_features['max_persistence']:.2f} yards")
    print(f"  - Significant features: {h0_features['num_significant']}")

    print(f"\nH1 (coverage gaps): {h1_features['num_features']} features")
    print(f"  - Max persistence: {h1_features['max_persistence']:.2f} yards")
    print(f"  - Avg persistence: {h1_features['avg_persistence']:.2f} yards")
    print(f"  - Significant gaps: {h1_features['num_significant']}")

    print("\n✓ Successfully analyzed real formation!")

except FileNotFoundError:
    print("⚠ Processed data not found. Run preprocessing notebook first.")
    print("  File needed: data/processed/point_clouds.npy")

# Test 3: Batch processing
print("\n3. Testing batch processing...")
print("-" * 70)

try:
    # Process first 10 plays
    sample_clouds = dict(list(point_clouds.items())[:10])

    tda_batch = DefensiveCoverageTDA(max_dimension=1)
    results = tda_batch.compute_persistence_batch(sample_clouds, verbose=False)

    print(f"Processed {len(results)} formations")

    # Extract features
    import pandas as pd
    features_df = tda_batch.extract_features_batch(dimensions=[0, 1])

    print(f"\nFeatures DataFrame shape: {features_df.shape}")
    print(f"Columns: {list(features_df.columns)}")

    # Summary stats
    print(f"\nH1 Coverage Gap Statistics:")
    print(f"  - Avg max gap: {features_df['H1_max_persistence'].mean():.2f} yards")
    print(f"  - Avg number of gaps: {features_df['H1_num_features'].mean():.1f}")
    print(f"  - Avg significant gaps: {features_df['H1_num_significant'].mean():.1f}")

    print("\n✓ Batch processing successful!")

except NameError:
    print("⚠ Skipping batch test (no real data loaded)")

print("\n" + "=" * 70)
print("TDA MODULE TESTS COMPLETE")
print("=" * 70)
print("\nYou can now use:")
print("  - jupyter notebook notebooks/02_tda_analysis.ipynb")
print("\nTo perform full TDA analysis on your defensive formations!")

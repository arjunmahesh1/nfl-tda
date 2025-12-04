# NFL Defensive Coverage Analysis Using Topological Data Analysis

MATH 412 project applying TDA to identify and quantify defensive coverage gaps in NFL tracking data

### Research Question

Can persistent homology quantify defensive coverage gaps, and do topological features correlate with play outcomes (completed vs. incomplete passes)?


### 1. Installation

```bash
# Clone the repository
git clone https://github.com/arjunmahesh1/nfl-tda.git
cd nfl-tda

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Data

#### Option A: Manual Download

1. [NFL Big Data Bowl 2021 on Kaggle](https://www.kaggle.com/c/nfl-big-data-bowl-2021/data)
2. Click "Download All" (~1.5 GB) > Extract the ZIP file
3. Move files to `/data/raw/`

#### Option B: Kaggle API

```bash
# Install Kaggle CLI
pip install kaggle

# Download dataset (Authenticate Kaggle on browser first)
kaggle competitions download -c nfl-big-data-bowl-2021 -p data/raw/

# Unzip (Linux/Mac)
cd data/raw && unzip nfl-big-data-bowl-2021.zip && cd ../..
# Unzip (Windows PowerShell)
Set-Location -Path "data\raw"; Expand-Archive -Path "nfl-big-data-bowl-2021.zip" -DestinationPath .; Set-Location -Path "..\.."
```

**`data/raw/`:**
- `games.csv` - Game-level metadata
- `plays.csv` - Play-level information
- `players.csv` - Player roster data
- `week1.csv` through `week17.csv` - Tracking data (10 Hz player positions)

### 3. Run Preprocessing

```bash
# Option 1: Jupyter notebook
jupyter notebook notebooks/01_data_preprocessing.ipynb
# Option 2: Python script
cd src
python preprocessing.py
```

### 4. Results

The preprocessing pipeline will:
- Load and merge all data files
- Standardize coordinates (flip plays so offense always moves left-to-right)
- Extract ball release frames (critical moment for coverage analysis)
- Filter to defensive players only
- Create point cloud representations
- Generate validation visualizations

## Key Features

### Data Preprocessing (`src/preprocessing.py`)

**NFLDataPreprocessor** class provides:
- `load_data()`: Load all CSV files from Big Data Bowl dataset
- `merge_data()`: Combine tracking data with play/game metadata
- `standardize_coordinates()`: Normalize field orientation
- `extract_ball_release_frame()`: Identify critical moment in each play
- `filter_defensive_players()`: Isolate defenders for analysis
- `create_point_clouds()`: Convert positions to numpy arrays
- `preprocess_pipeline()`: Run complete workflow

### Visualization (`src/visualization.py`)

**NFLFieldVisualizer** class provides:
- `plot_formation()`: Display defensive formation on field diagram
- `plot_multiple_formations()`: Grid view of sample plays
- `plot_formation_statistics()`: Statistical distributions
- `validate_preprocessing()`: Generate validation plots



## Technical Details

### Coordinate Standardization

All plays are normalized so that:
- Offense always moves in the positive x-direction (left-to-right)
- Field is always 120 yards × 53.3 yards
- Line of scrimmage provides reference point

### Ball Release Frame Selection

We focus on the moment when the quarterback releases the ball because:
- Defensive formation is fully developed
- Coverage gaps are most relevant
- Consistent snapshot across all plays

### Point Cloud Construction

Each play is represented as a set of (x, y) coordinates:
- 11 defensive players (typically, sometimes less - but can be interepreted as defenders not important to coverage/camera angle)
- 2D positions on the field
- Suitable for persistent homology via Vietoris-Rips filtration

## Usage Examples

### Load Preprocessed Data

```python
from src.preprocessing import load_point_clouds
import pandas as pd

# Load saved data
defensive_df = pd.read_csv("data/processed/defensive_formations.csv")
point_clouds = load_point_clouds("data/processed/point_clouds.npy")

print(f"Loaded {len(point_clouds)} plays")
```

### Visualize a Formation

```python
from src.visualization import NFLFieldVisualizer

visualizer = NFLFieldVisualizer()

# Get a sample play
play_id = list(point_clouds.keys())[0]
formation = point_clouds[play_id]

# Plot it
visualizer.plot_formation(
    formation,
    title=f"Game {play_id[0]}, Play {play_id[1]}",
    annotate_players=True
)
```

### Compute Statistics

```python
import numpy as np

# Calculate average number of defenders
avg_defenders = np.mean([pc.shape[0] for pc in point_clouds.values()])
print(f"Average defenders per play: {avg_defenders:.1f}")

# Calculate formation spread
x_spreads = [np.std(pc[:, 0]) for pc in point_clouds.values()]
print(f"Average horizontal spread: {np.mean(x_spreads):.1f} yards")
```

## References

- **DATA:** **NFL Big Data Bowl 2021**: [Kaggle Competition](https://www.kaggle.com/c/nfl-big-data-bowl-2021)
- **Goldfarb, D. (2014)**: "An Application of Topological Data Analysis to Hockey Analytics"
- **Alagappan, M. (2012)**: "From 5 to 13: Redefining the Positions in Basketball"
- **Xu & Fong (2025)**: "Apply Topological Data Analysis in Football Scouting"

## Understanding the Output

### Defensive Formations CSV

The file `data/processed/defensive_formations.csv` contains one row per defensive player per play:

| Column | Description |
|--------|-------------|
| `gameId`, `playId` | Unique identifiers for each play |
| `x`, `y` | Standardized field coordinates (yards) |
| `team` | Defensive team name |
| `passResult` | Outcome (C=complete, I=incomplete, IN=interception, etc.) |
| `down`, `distance` | Game situation |

### Point Clouds NPY

The file `data/processed/point_clouds.npy` contains a Python dictionary:

```python
{
    (gameId, playId): numpy_array_of_shape_(11, 2),
    ...
}
```

Each value is an array of (x, y) coordinates for defenders at ball release:

```python
array([[45.2, 26.7],   # Defender 1
       [48.1, 15.3],   # Defender 2
       [52.4, 38.9],   # Defender 3
       ...])
```

---


## Interpretation:
1. H₀ (Defender Clustering)
- High H₀ (many components) = Defenders spread out, isolated
- Low H₀ (few components) = Defenders bunched together in groups

- For offense: If H₀ = 3 at snap, there are 3 defender clusters → Attack the seams between them
- For defense: Monitor your H₀ - if too high (>8), defenders can't help each other; if too low (< 3), vulnerable to spread plays
   - Example: Cover 2 typically has H₀ ≈ 3 (two deep safeties + front seven as one cluster)

2. H₁ (Coverage Gaps)
- H₁ = 0 → No gaps, tight coverage
- H₁ = 2 → Two distinct holes offense can exploit
- Persistence = How big the gap is in yards
- Findings:
   - Incomplete passes: 0.42 gaps (MORE)
   - Complete passes:   0.40 gaps (FEWER)
   - For Defense: Having gaps isn't automatically bad, data shows incomplete passes actually had more gaps This means:
      - Aggressive coverage (leaving some gaps intentionally) can bait QBs into mistakes
      - Gap existence ≠ offensive success
   - For Offense: Just because you see a gap on film doesn't mean:
      - The gap will be there during the play
      - You can complete the pass even if it is
      - QB execution and timing matter more than gap detection

3. Persistence Landscapes
- Complete and incomplete passes had similar landscapes

4. Persistence Images
- Bright spots = Large, persistent gaps
- Dark areas = Tight coverage
- No bright spots = Defense is locked down
- (Could unlock film plays with darkest spots)

5. Betti Curves
- β₀(depth) = How many defender clusters at each depth
- β₁(depth) = Where gaps appear as routes develop
- Betti curve shows β₁ peaks at 15 yards
   → Most gaps appear at intermediate depth
- Formation stretching: If β₀ stays high (many clusters) when spread:
   → Defenders can't support each other
   → Keep spreading them out
- If β₀ drops quickly:
   → Defense compacts well
   → Need different approach (vertical routes)

6. Bottleneck Distance
- How "similar" two defensive formations are topologically 

7. Wasserstein Distance
- Low Wasserstein = Excellent coverage discipline
- High Wasserstein = Lots of vulnerability across the field

8. Permutation Test (Statistical Significance)
- If p < 0.05: Coverage gaps do significantly affect completions
- If p > 0.05: Other factors (QB skill, pressure) matter more than gaps

9. Mapper Algorithm
- Groups formations into "families" 

10. Hierarchical Clustering
- Which formations are "related" to each other

11. MDS (2D Formation Map)
- Visual map of all possible defensive structures
- Green dots (Complete) and Red dots (Incomplete) are mixed
   → No clear separation
   → Coverage gaps alone don't determine success
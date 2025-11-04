# NFL Defensive Coverage Analysis Using Topological Data Analysis

MATH 412 project applying TDA to identify and quantify defensive coverage gaps in NFL tracking data.

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

## Methodology

### 1. Data Preprocessing (Week 1-2) ✓

- [x] Load NFL tracking data
- [x] Standardize coordinate systems
- [x] Extract ball release frames
- [x] Create defensive point clouds
- [x] Validation visualizations

### 2. TDA Implementation (Week 3-4)

- [ ] Compute Vietoris-Rips filtration
- [ ] Calculate persistent homology (H₀ and H₁)
- [ ] Generate persistence diagrams
- [ ] Extract topological features

### 3. Analysis (Week 5-6)

- [ ] Compare features by play outcome
- [ ] Cluster analysis of coverage patterns
- [ ] Statistical testing
- [ ] Correlation with defensive success

### 4. Interpretation & Visualization (Week 7-8)

- [ ] Create representative examples
- [ ] Generate final plots
- [ ] Interpret topological features

### 5. Report Writing (Week 9-12)

- [ ] Draft paper sections
- [ ] Final report: Feedback= Separate sections for organization purposes for clearer writing style
- [ ] Presentation

## Expected Outcomes

1. **Quantitative Coverage Metrics**: Topological features that measure coverage quality
   - H₀ persistence: Defender clustering/spread
   - H₁ persistence: Coverage "holes" or gaps

2. **Correlation Analysis**: Relationship between topology and play success
   - Do completed passes show larger H₁ features (bigger gaps)?
   - Do tight coverages merge quickly in filtration?

3. **Formation Clustering**: Identify distinct defensive patterns
   - Zone vs. man coverage signatures
   - Team-specific tendencies

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

- **NFL Big Data Bowl 2021**: [Kaggle Competition](https://www.kaggle.com/c/nfl-big-data-bowl-2021)
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

**Current Status**: Week 2 Complete - Data preprocessing pipeline implemented ✓

**Next Milestone**: Week 3-4 - TDA implementation and persistent homology computation
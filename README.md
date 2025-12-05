# Mood to Song Bot

This is a small Streamlit + scikit-learn app that recommends songs based on **audio “vibe” features** from Spotify (danceability, energy, valence, etc.).

You can:
- Find songs similar to a **seed track**.
- Discover songs that match a **mood point** using sliders.
- Explore **mood clusters** of tracks and see representative examples.

---

## Features

- **Seed track recommendations**  
  Search by track + optional artist and get similar songs based on audio features.

- **Mood-based recommendations**  
  Use sliders for energy, valence (happiness), danceability, and optional advanced controls (acousticness, instrumentalness, tempo).

- **Mood clusters**  
  KMeans clusters over the feature space; inspect average feature values and sample tracks from each cluster.

- **Simple, configurable core**  
  All important columns and defaults are defined in `src/config.py`.

---

## How it works

### Data & preprocessing

- Expects a CSV file with:
  - **Feature columns** (numeric):  
    `danceability`, `energy`, `loudness`, `speechiness`, `acousticness`,  
    `instrumentalness`, `liveness`, `valence`, `tempo`, `duration_ms`, `popularity`
  - **ID/metadata columns**:  
    `track_id`, `track_name`, `artists`, `track_genre`

- Preprocessing pipeline (`src/preprocess.py`):
  1. Load CSV.
  2. Drop rows with missing values in feature columns.
  3. Scale features with `StandardScaler`.

### Models

Defined in `src/models.py` and `src/recommender.py`:

- `KMeans` for **mood clusters** (default: 5 clusters).
- `NearestNeighbors` (Euclidean) for **similarity search** around:
  - A **seed track**’s feature vector.
  - A **constructed mood vector** (from sliders + feature means).

`VibeRecommender` glues everything together:
- Builds from CSV via `VibeRecommender.from_csv(...)`.
- Provides:
  - `recommend_by_track(...)`
  - `recommend_by_mood(...)`
  - `describe_clusters()`
  - `sample_cluster_tracks(...)`

---

## Installation

### 1. Clone the repo

```bash
git clone <your-repo-url> vibematch
cd vibematch
````

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install streamlit scikit-learn pandas numpy
```

(If you already have a `requirements.txt`, you can instead do `pip install -r requirements.txt`.)

---

## Dataset

The app expects a CSV at:

```text
data/spotify_tracks.csv
```

relative to the directory you run Streamlit from (typically the project root).

That file must contain at least the columns listed in **How it works → Data & preprocessing**.
You can adapt any Spotify-like dataset to match those column names.

---

## Running the app

From the **project root**, run:

```bash
streamlit run src/app_streamlit.py
```

Then open the URL shown in the terminal (usually `http://localhost:8501`).

---

## UI overview

### 1. Seed track

* Mode: **“Seed track”** in the sidebar.
* Type a track name (and optionally artist) to filter.
* Choose from the dropdown of matching tracks.
* Click **“Recommend similar tracks”**:

  * Top section: the seed track (clickable link to Spotify, when `track_id` is present).
  * Table: recommended tracks with artists, genre, cluster label, and distance.

### 2. Mood sliders

* Mode: **“Mood sliders”** in the sidebar.
* Adjust:

  * **Energy**
  * **Happiness (valence)**
  * **Danceability**
* Optionally enable **“Show advanced audio controls”** for:

  * Acousticness
  * Instrumentalness
  * Tempo (BPM)
* Click **“Find songs”** to see a table of matching tracks with mood-related columns.

### 3. Mood clusters

* Mode: **“Mood clusters”** in the sidebar.
* View cluster summary (average values for each feature per cluster).
* Select a cluster and number of example tracks.
* See representative tracks for that cluster in a standard table.

---

## Project structure

```text
src/
  app_streamlit.py   # Streamlit UI entrypoint
  config.py          # Feature/ID column names and default hyperparameters
  preprocess.py      # Loading, cleaning, scaling, and PreprocessResult
  models.py          # VibeModels: KMeans + NearestNeighbors
  recommender.py     # VibeRecommender: main recommendation interface

data/
  spotify_tracks.csv # Your dataset (not included in this repo)
```

---

## Configuration & customization

Key settings live in `src/config.py`:

* `FEATURE_COLUMNS`
  List of numeric feature columns used for modeling.

* ID columns:
  `ID_COL_TRACK_ID`, `ID_COL_TRACK_NAME`, `ID_COL_ARTISTS`, `ID_COL_GENRE`

* Defaults:
  `DEFAULT_N_CLUSTERS`, `DEFAULT_N_NEIGHBORS`, `RANDOM_STATE`

You can also pass custom values programmatically:

```python
from recommender import VibeRecommender

rec = VibeRecommender.from_csv(
    "data/spotify_tracks.csv",
    n_clusters=5,
    n_neighbors=30,
)
```
from typing import List

# Columns used as numeric features
FEATURE_COLUMNS: List[str] = [
    "danceability",
    "energy",
    "loudness",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence",
    "tempo",
    "duration_ms",
    "popularity",
]

# Columns with identifying metadata
ID_COL_TRACK_ID = "track_id"
ID_COL_TRACK_NAME = "track_name"
ID_COL_ARTISTS = "artists"
ID_COL_GENRE = "track_genre"

# Default modeling hyperparameters
DEFAULT_N_CLUSTERS = 5
DEFAULT_N_NEIGHBORS = 30  # we’ll filter out the seed + then take top-n user wants
RANDOM_STATE = 42

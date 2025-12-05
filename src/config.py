from typing import List

"""Configuration constants for the VibeMatch recommender."""

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
ID_COL_TRACK_ID: str = "track_id"
ID_COL_TRACK_NAME: str = "track_name"
ID_COL_ARTISTS: str = "artists"
ID_COL_GENRE: str = "track_genre"

# Default modeling hyperparameters
DEFAULT_N_CLUSTERS: int = 5
# We'll filter out the seed, then take the top-n the user wants
DEFAULT_N_NEIGHBORS: int = 30
RANDOM_STATE: int = 42

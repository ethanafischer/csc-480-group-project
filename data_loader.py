# data_loader.py
import pandas as pd

def load_tracks(csv_path: str = "spotify_tracks.csv"):
    df = pd.read_csv(csv_path)

    # Make sure expected columns exist (you can relax this if needed)
    required_cols = ["track_name", "artists", "valence", "energy", "danceability"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")

    # Optional: drop rows with missing key features
    df = df.dropna(subset=["valence", "energy", "danceability"])

    return df

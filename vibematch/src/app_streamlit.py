import streamlit as st
import pandas as pd
from typing import Dict

from recommender import VibeRecommender
from config import ID_COL_TRACK_NAME, ID_COL_ARTISTS, ID_COL_GENRE


@st.cache_resource
def load_recommender() -> VibeRecommender:
    # Adjust path if running from a different working directory
    return VibeRecommender.from_csv("data/spotify_tracks.csv")


def format_track_row(row: pd.Series) -> str:
    artist = row.get(ID_COL_ARTISTS, "Unknown artist")
    name = row.get(ID_COL_TRACK_NAME, "Unknown track")
    return f"{name} — {artist}"


def page_seed_track(rec: VibeRecommender):
    st.header("🎧 Recommend by Seed Track")

    st.write("Search for a track by name (and optionally by artist), then pick it as the seed.")

    # --- search inputs ---
    search_name = st.text_input("Track name (or part of it)")
    search_artist = st.text_input("Optional: artist name filter")

    # Filter candidates based on search
    candidates = rec.df[[ID_COL_TRACK_NAME, ID_COL_ARTISTS]].drop_duplicates()

    if search_name:
        candidates = candidates[
            candidates[ID_COL_TRACK_NAME]
            .str.contains(search_name, case=False, na=False)
        ]

    if search_artist:
        candidates = candidates[
            candidates[ID_COL_ARTISTS]
            .str.contains(search_artist, case=False, na=False)
        ]

    # Limit number of options to keep UI snappy
    MAX_OPTIONS = 50
    total_matches = len(candidates)

    if total_matches == 0:
        st.info("Start typing a track name above to see matching songs.")
        return

    if total_matches > MAX_OPTIONS:
        candidates = candidates.head(MAX_OPTIONS)
        st.caption(
            f"Showing first {MAX_OPTIONS} matches out of {total_matches}. "
            "Refine your search to narrow down further."
        )

    # Build labels only for the small candidate set
    candidates = candidates.copy()
    candidates["label"] = candidates.apply(
        lambda r: f"{r[ID_COL_TRACK_NAME]} — {r[ID_COL_ARTISTS]}",
        axis=1,
    )

    # Now the selectbox is only over a small, filtered list → much faster
    selected_label = st.selectbox(
        "Choose seed track from results",
        options=candidates["label"].tolist(),
    )

    n_recs = st.slider("Number of recommendations", 5, 30, 10)

    if st.button("Recommend similar tracks"):
        name, artist = selected_label.split(" — ", 1)

        seed_row, recs = rec.recommend_by_track(
            track_name=name,
            n=n_recs,
            artist_hint=artist,
        )

        if seed_row is None or recs.empty:
            st.error("Could not find recommendations for that track.")
            return

        st.subheader("Seed track")
        st.write(f"{seed_row[ID_COL_TRACK_NAME]} — {seed_row[ID_COL_ARTISTS]}")

        st.subheader("Recommended tracks")
        display_cols = [
            ID_COL_TRACK_NAME,
            ID_COL_ARTISTS,
            ID_COL_GENRE,
            "mood_cluster",
            "distance",
        ]
        display_cols = [c for c in display_cols if c in recs.columns]
        st.dataframe(recs[display_cols].reset_index(drop=True))


def page_mood(rec: VibeRecommender):
    st.header("🌈 Recommend by Mood")

    st.write("Use the sliders to set the vibe you want:")

    # --- basic controls ---
    st.subheader("Basic mood controls")

    energy = st.slider("Energy", 0.0, 1.0, 0.7, 0.05)
    valence = st.slider("Happiness (valence)", 0.0, 1.0, 0.7, 0.05)
    danceability = st.slider("Danceability", 0.0, 1.0, 0.7, 0.05)

    # --- advanced controls toggle ---
    advanced = st.checkbox("Show advanced audio controls")

    extra_overrides = None  # default: no advanced filtering

    if advanced:
        st.subheader("Advanced controls")

        # Spotify acousticness / instrumentalness are [0,1]
        acousticness = st.slider(
            "Acousticness (more acoustic ↔ more electric)",
            0.0,
            1.0,
            0.5,
            0.05,
        )
        instrumentalness = st.slider(
            "Instrumentalness (more vocals ↔ more instrumental)",
            0.0,
            1.0,
            0.0,
            0.05,
        )
        # Tempo in BPM
        tempo = st.slider(
            "Tempo (BPM)",
            60.0,
            200.0,
            120.0,
            1.0,
        )

        extra_overrides = {
            "acousticness": acousticness,
            "instrumentalness": instrumentalness,
            "tempo": tempo,
        }

    n_recs = st.slider("Number of recommendations", 5, 30, 10)

    if st.button("Find songs"):
        recs = rec.recommend_by_mood(
            energy=energy,
            valence=valence,
            danceability=danceability,
            n=n_recs,
            extra_overrides=extra_overrides,  # None if advanced is off
        )

        # Base columns always shown
        display_cols = [
            ID_COL_TRACK_NAME,
            ID_COL_ARTISTS,
            ID_COL_GENRE,
            "mood_cluster",
            "energy",
            "valence",
            "danceability",
        ]

        # Only show advanced feature columns if we actually used them
        if advanced:
            display_cols += ["acousticness", "instrumentalness", "tempo"]

        display_cols = [c for c in display_cols if c in recs.columns]

        st.subheader("Recommended tracks")
        st.dataframe(recs[display_cols].reset_index(drop=True))


def page_clusters(rec: VibeRecommender):
    st.header("🧠 Mood Clusters")

    st.subheader("Cluster summary (average features)")
    summary = rec.describe_clusters()
    st.dataframe(summary)

    cluster_labels = sorted(rec.df["mood_cluster"].unique())
    cluster = st.selectbox("Inspect cluster", options=cluster_labels)

    n_samples = st.slider("Number of example tracks", 5, 30, 10)
    samples = rec.sample_cluster_tracks(cluster_label=cluster, n=n_samples)
    st.subheader(f"Example tracks in cluster {cluster}")
    st.dataframe(samples.reset_index(drop=True))


def main():
    st.set_page_config(page_title="VibeMatch", page_icon="🎵", layout="wide")

    st.title("VibeMatch 🎵")
    st.write("An AI-driven vibe-based song recommender using Spotify audio features.")

    rec = load_recommender()

    mode = st.sidebar.radio(
        "Mode",
        options=["Seed track", "Mood sliders", "Mood clusters"],
    )

    if mode == "Seed track":
        page_seed_track(rec)
    elif mode == "Mood sliders":
        page_mood(rec)
    else:
        page_clusters(rec)


if __name__ == "__main__":
    main()

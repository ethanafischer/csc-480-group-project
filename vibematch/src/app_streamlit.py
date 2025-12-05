import streamlit as st
import pandas as pd

from recommender import VibeRecommender
from config import (
    ID_COL_TRACK_NAME,
    ID_COL_ARTISTS,
    ID_COL_GENRE,
    ID_COL_TRACK_ID,
)


@st.cache_resource
def load_recommender() -> VibeRecommender:
    # Adjust path if running from a different working directory
    return VibeRecommender.from_csv("data/spotify_tracks.csv")


def format_track_row(row: pd.Series) -> str:
    artist = row.get(ID_COL_ARTISTS, "Unknown artist")
    name = row.get(ID_COL_TRACK_NAME, "Unknown track")
    return f"{name} — {artist}"


SPOTIFY_TRACK_BASE_URL = "https://open.spotify.com/track/"


def make_track_link(row: pd.Series) -> str:
    """Return HTML link for a single track using its Spotify ID."""
    track_id = row.get(ID_COL_TRACK_ID)
    name = row.get(ID_COL_TRACK_NAME, "Unknown track")
    if pd.isna(track_id):
        return name
    url = f"{SPOTIFY_TRACK_BASE_URL}{track_id}"
    return f'<a href="{url}" target="_blank">{name}</a>'

def render_recs_table(recs: pd.DataFrame, extra_cols) -> None:
    """
    Render recommendations as an HTML table with the song title hyperlinked.
    extra_cols: list of column names to show in addition to the title.
    """
    if recs.empty:
        st.info("No recommendations found.")
        return

    recs = recs.copy()
    recs["Title"] = recs.apply(make_track_link, axis=1)

    cols = ["Title"] + [c for c in extra_cols if c in recs.columns]
    df_display = recs[cols].reset_index(drop=True)

    st.markdown(df_display.to_html(escape=False, index=False), unsafe_allow_html=True)

def page_seed_track(rec: VibeRecommender):
    st.header("🎧 Recommend by Seed Track")

    st.write("Search for a track by name (and optionally by artist), then pick it as the seed.")

    # --- search inputs ---
    search_name = st.text_input("Track name (or part of it)")
    search_artist = st.text_input("Optional: artist name filter")

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

    candidates = candidates.copy()
    candidates["label"] = candidates.apply(
        lambda r: f"{r[ID_COL_TRACK_NAME]} — {r[ID_COL_ARTISTS]}",
        axis=1,
    )

    selected_label = st.selectbox(
        "Choose seed track from results",
        options=candidates["label"].tolist(),
    )

    N_RECS = 10  # fixed number of recommendations

    if st.button("Recommend similar tracks"):
        name, artist = selected_label.split(" — ", 1)

        seed_row, recs = rec.recommend_by_track(
            track_name=name,
            n=N_RECS,
            artist_hint=artist,
        )

        if seed_row is None or recs.empty:
            st.error("Could not find recommendations for that track.")
            return

        st.subheader("Seed track")
        seed_html = make_track_link(seed_row)
        st.markdown(seed_html, unsafe_allow_html=True)

        st.subheader("Recommended tracks")
        # We rely on dedupe logic in recommender; then we cut display to 10
        recs = recs.head(N_RECS)
        render_recs_table(
            recs,
            extra_cols=[ID_COL_ARTISTS, ID_COL_GENRE, "mood_cluster", "distance"],
        )


def page_mood(rec: VibeRecommender):
    st.header("Recommend by Mood")

    st.write("Use the sliders to set the vibe you want:")

    st.subheader("Basic mood controls")

    energy = st.slider("Energy", 0.0, 1.0, 0.7, 0.05)
    valence = st.slider("Happiness (valence)", 0.0, 1.0, 0.7, 0.05)
    danceability = st.slider("Danceability", 0.0, 1.0, 0.7, 0.05)

    advanced = st.checkbox("Show advanced audio controls")

    extra_overrides = None

    if advanced:
        st.subheader("Advanced controls")

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

    N_RECS = 10  # fixed

    if st.button("Find songs"):
        recs = rec.recommend_by_mood(
            energy=energy,
            valence=valence,
            danceability=danceability,
            n=N_RECS,
            extra_overrides=extra_overrides,
        )

        recs = recs.head(N_RECS)

        base_cols = [
            ID_COL_ARTISTS,
            ID_COL_GENRE,
            "mood_cluster",
            "energy",
            "valence",
            "danceability",
        ]
        if advanced:
            base_cols += ["acousticness", "instrumentalness", "tempo"]

        st.subheader("Recommended tracks")
        render_recs_table(recs, extra_cols=base_cols)


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

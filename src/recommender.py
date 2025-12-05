from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import (
    DEFAULT_N_CLUSTERS,
    DEFAULT_N_NEIGHBORS,
    FEATURE_COLUMNS,
    ID_COL_ARTISTS,
    ID_COL_GENRE,
    ID_COL_TRACK_ID,
    ID_COL_TRACK_NAME,
)
from models import VibeModels
from preprocess import PreprocessResult, preprocess_pipeline


@dataclass
class VibeRecommender:
    """Main interface for computing and serving VibeMatch recommendations."""
    df: pd.DataFrame
    X_scaled: np.ndarray
    feature_columns: List[str]
    models: VibeModels
    scaler: StandardScaler

    @classmethod
    def from_csv(
        cls,
        csv_path: str,
        feature_columns: Optional[List[str]] = None,
        n_clusters: int = DEFAULT_N_CLUSTERS,
        n_neighbors: int = DEFAULT_N_NEIGHBORS,
    ) -> "VibeRecommender":
        """
        Build a VibeRecommender from a CSV file.

        This runs the full preprocessing pipeline and fits clustering
        and nearest-neighbor models.
        """
        if feature_columns is None:
            feature_columns = FEATURE_COLUMNS

        prep: PreprocessResult = preprocess_pipeline(csv_path, feature_columns)
        models = VibeModels.fit(
            prep.X_scaled,
            n_clusters=n_clusters,
            n_neighbors=n_neighbors,
        )

        # Assign mood clusters
        clusters = models.assign_clusters(prep.X_scaled)
        df_with_clusters = prep.df.copy()
        df_with_clusters["mood_cluster"] = clusters

        return cls(
            df=df_with_clusters,
            X_scaled=prep.X_scaled,
            feature_columns=prep.feature_columns,
            models=models,
            scaler=prep.scaler,
        )

    # ---------- internal helpers ----------

    def _get_track_indices_by_name(self, track_name: str) -> List[int]:
        """Return indices of rows whose track name matches (case-insensitive)."""
        mask = self.df[ID_COL_TRACK_NAME].str.lower() == track_name.lower()
        return list(self.df[mask].index)

    def _build_mood_vector(
        self,
        overrides: Dict[str, float],
    ) -> np.ndarray:
        """
        Build a full feature vector from partial user input.

        For unspecified features, dataset means are used.
        """
        base = self.df[self.feature_columns].mean().to_dict()
        base.update(overrides)

        raw_vec = np.array([base[col] for col in self.feature_columns], dtype=float)
        scaled_vec = self.scaler.transform(raw_vec.reshape(1, -1))[0]
        return scaled_vec

    def _recommend_from_vector(
        self,
        query_vec: np.ndarray,
        n: int,
        exclude_index: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Get up to n nearest rows (by index), optionally excluding a specific row.

        This returns a pool; de-duplication happens separately.
        """
        distances, indices = self.models.query_neighbors(
            self.X_scaled,
            query_vec,
            n_neighbors=n,
        )

        idx_list: List[int] = []
        dist_list: List[float] = []

        for d, idx in zip(distances, indices):
            if exclude_index is not None and idx == exclude_index:
                continue
            idx_list.append(idx)
            dist_list.append(d)
            if len(idx_list) >= n:
                break

        recs = self.df.iloc[idx_list].copy()
        recs["distance"] = dist_list
        return recs

    def _dedupe_recommendations(
        self,
        recs: pd.DataFrame,
        n: int,
        seed_key: Optional[Tuple[str, str]] = None,
    ) -> pd.DataFrame:
        """
        Remove duplicates based on (track_name, artist) and optionally drop the seed.

        Then keep the closest n by distance.
        """
        work = recs.copy()

        # Drop exact seed matches (same track name + artist)
        if seed_key is not None:
            seed_name, seed_artist = seed_key
            work = work[
                ~(
                    (work[ID_COL_TRACK_NAME] == seed_name)
                    & (work[ID_COL_ARTISTS] == seed_artist)
                )
            ]

        # Deduplicate by (track_name, artist), keeping the closest (smallest distance)
        if ID_COL_TRACK_NAME in work.columns and ID_COL_ARTISTS in work.columns:
            work = (
                work.sort_values("distance", ascending=True)
                .drop_duplicates(
                    subset=[ID_COL_TRACK_NAME, ID_COL_ARTISTS],
                    keep="first",
                )
            )

        return work.head(n)

    # ---------- public API ----------

    def recommend_by_track(
        self,
        track_name: str,
        n: int = 10,
        artist_hint: Optional[str] = None,
    ) -> Tuple[Optional[pd.Series], pd.DataFrame]:
        """
        Recommend songs similar to a seed track.

        Returns
        -------
        (seed_row, recommendations_df)
            If track not found, returns (None, empty_df).
        """
        indices = self._get_track_indices_by_name(track_name)
        if not indices:
            return None, self.df.head(0).copy()

        # If multiple matches and artist_hint provided, try to pick that
        if artist_hint:
            candidates = self.df.loc[indices]
            mask = candidates[ID_COL_ARTISTS].str.contains(
                artist_hint,
                case=False,
                na=False,
            )
            filtered = candidates[mask]
            if not filtered.empty:
                seed_idx = int(filtered.index[0])
            else:
                seed_idx = indices[0]
        else:
            seed_idx = indices[0]

        seed_row = self.df.loc[seed_idx]
        seed_vec = self.X_scaled[seed_idx]

        # Get a slightly larger neighbor pool so we have room to dedupe.
        pool_size = max(DEFAULT_N_NEIGHBORS, n + 5)
        pool_recs = self._recommend_from_vector(
            seed_vec,
            n=pool_size,
            exclude_index=seed_idx,
        )

        seed_key = (
            str(seed_row[ID_COL_TRACK_NAME]),
            str(seed_row[ID_COL_ARTISTS]),
        )
        recs = self._dedupe_recommendations(pool_recs, n=n, seed_key=seed_key)

        return seed_row, recs

    def recommend_by_mood(
        self,
        energy: float,
        valence: float,
        danceability: float,
        n: int = 10,
        extra_overrides: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Recommend songs close to a mood point.

        Parameters
        ----------
        energy, valence, danceability : float
            Core mood controls in [0, 1].
        n : int
            Number of recommendations to return.
        extra_overrides : dict[str, float] or None
            Optional extra feature values (e.g. acousticness, tempo).
            If None, only the three basic sliders are used.
        """
        # Always set these three
        overrides: Dict[str, float] = {
            "energy": energy,
            "valence": valence,
            "danceability": danceability,
        }

        # Only merge extras if explicitly passed
        if extra_overrides is not None:
            overrides.update(extra_overrides)

        mood_vec = self._build_mood_vector(overrides)

        pool_size = max(DEFAULT_N_NEIGHBORS, n + 5)
        pool_recs = self._recommend_from_vector(
            mood_vec,
            n=pool_size,
            exclude_index=None,
        )

        recs = self._dedupe_recommendations(pool_recs, n=n, seed_key=None)
        return recs

    def describe_clusters(self) -> pd.DataFrame:
        """Return average feature values per cluster (for interpretability)."""
        group = self.df.groupby("mood_cluster")[self.feature_columns]
        summary = group.mean().reset_index()
        return summary[["mood_cluster"] + self.feature_columns]

    def sample_cluster_tracks(self, cluster_label: int, n: int = 10) -> pd.DataFrame:
        """
        Sample example tracks from a mood cluster.

        Returns a dataframe with ID columns, cluster label, and feature values.
        """
        subset = self.df[self.df["mood_cluster"] == cluster_label]
        return subset.sample(n=min(n, len(subset)), random_state=0)[
            [
                ID_COL_TRACK_NAME,
                ID_COL_ARTISTS,
                ID_COL_GENRE,
                "mood_cluster",
            ]
            + self.feature_columns
        ]

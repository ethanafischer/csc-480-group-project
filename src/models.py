from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from config import DEFAULT_N_CLUSTERS, DEFAULT_N_NEIGHBORS, RANDOM_STATE


@dataclass
class VibeModels:
    """Wrapper for fitted KMeans and NearestNeighbors models."""
    kmeans: KMeans
    knn: NearestNeighbors

    @classmethod
    def fit(
        cls,
        X_scaled: np.ndarray,
        n_clusters: int = DEFAULT_N_CLUSTERS,
        n_neighbors: int = DEFAULT_N_NEIGHBORS,
    ) -> "VibeModels":
        """Fit KMeans and NearestNeighbors on the scaled feature matrix."""
        # K-means for mood clusters
        kmeans = KMeans(
            n_clusters=n_clusters,
            random_state=RANDOM_STATE,
            n_init=10,
        )
        kmeans.fit(X_scaled)

        # NearestNeighbors for similarity search
        knn = NearestNeighbors(
            n_neighbors=min(n_neighbors, len(X_scaled)),
            metric="euclidean",
            algorithm="auto",
        )
        knn.fit(X_scaled)

        return cls(kmeans=kmeans, knn=knn)

    def assign_clusters(self, X_scaled: np.ndarray) -> np.ndarray:
        """Assign cluster labels for each row in X_scaled."""
        return self.kmeans.predict(X_scaled)

    def query_neighbors(
        self,
        X_scaled: np.ndarray,  # kept for API compatibility; not used directly
        query_vector: np.ndarray,
        n_neighbors: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query nearest neighbors for a single query vector.

        Parameters
        ----------
        X_scaled : np.ndarray
            Unused; retained for backwards compatibility.
        query_vector : np.ndarray
            Shape (n_features,) or (1, n_features).
        n_neighbors : int
            Number of neighbors to request (capped by fitted knn.n_neighbors).

        Returns
        -------
        distances : np.ndarray
            1D array of neighbor distances.
        indices : np.ndarray
            1D array of neighbor indices.
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        k = min(n_neighbors, self.knn.n_neighbors)
        distances, indices = self.knn.kneighbors(query_vector, n_neighbors=k)
        return distances[0], indices[0]

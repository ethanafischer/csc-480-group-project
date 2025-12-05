from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from config import DEFAULT_N_CLUSTERS, DEFAULT_N_NEIGHBORS, RANDOM_STATE


@dataclass
class VibeModels:
    kmeans: KMeans
    knn: NearestNeighbors

    @classmethod
    def fit(
        cls,
        X_scaled: np.ndarray,
        n_clusters: int = DEFAULT_N_CLUSTERS,
        n_neighbors: int = DEFAULT_N_NEIGHBORS,
    ) -> "VibeModels":
        # k-means for mood clusters
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
        return self.kmeans.predict(X_scaled)

    def query_neighbors(
        self,
        X_scaled: np.ndarray,
        query_vector: np.ndarray,
        n_neighbors: int,
    ):
        """
        query_vector: shape (n_features,) or (1, n_features)
        Returns distances, indices.
        """
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        k = min(n_neighbors, self.knn.n_neighbors)
        distances, indices = self.knn.kneighbors(query_vector, n_neighbors=k)
        return distances[0], indices[0]

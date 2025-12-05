from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class PreprocessResult:
    df: pd.DataFrame                 # cleaned dataframe (with same row order as X_scaled)
    X_scaled: np.ndarray             # scaled feature matrix
    scaler: StandardScaler           # fitted scaler
    feature_columns: List[str]       # columns used as features


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the raw Spotify dataset."""
    df = pd.read_csv(csv_path)
    return df


def clean_and_select_features(
    df: pd.DataFrame,
    feature_columns: List[str],
) -> pd.DataFrame:
    """
    Keep only rows with non-missing values in the selected feature columns.
    Return a cleaned copy of df.
    """
    existing = [c for c in feature_columns if c in df.columns]
    missing = set(feature_columns) - set(existing)
    if missing:
        raise ValueError(f"Missing expected feature columns in dataset: {missing}")

    df_clean = df.dropna(subset=existing).copy()
    return df_clean


def scale_features(
    df: pd.DataFrame,
    feature_columns: List[str],
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Scale numeric feature columns using StandardScaler.
    If scaler is provided, reuse it; otherwise fit a new one.
    """
    X = df[feature_columns].values.astype(float)

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, scaler


def preprocess_pipeline(
    csv_path: str,
    feature_columns: List[str],
) -> PreprocessResult:
    """
    Full preprocessing pipeline:
    - load data
    - clean rows
    - scale features
    """
    raw_df = load_dataset(csv_path)
    df_clean = clean_and_select_features(raw_df, feature_columns)
    X_scaled, scaler = scale_features(df_clean, feature_columns)

    return PreprocessResult(
        df=df_clean.reset_index(drop=True),
        X_scaled=X_scaled,
        scaler=scaler,
        feature_columns=feature_columns,
    )

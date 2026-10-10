"""Data download, cleaning, and loading."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from heart_disease.config import (
    COLUMN_NAMES,
    DATA_PROCESSED,
    DATA_RAW,
    PROCESSED_FILENAME,
    RAW_FILENAME,
    UCI_URL,
)


def download_raw(dest_dir: Path | None = None) -> Path:
    """Download Cleveland heart disease dataset from UCI."""
    dest_dir = dest_dir or DATA_RAW
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_path = dest_dir / RAW_FILENAME
    if not out_path.exists():
        urllib.request.urlretrieve(UCI_URL, out_path)
    return out_path


def clean_data(raw_path: Path | None = None, save: bool = True) -> pd.DataFrame:
    """
    Load raw UCI file, handle missing '?', binarize target (num > 0 -> 1).
    """
    raw_path = raw_path or (DATA_RAW / RAW_FILENAME)
    df = pd.read_csv(raw_path, header=None, names=COLUMN_NAMES, na_values="?")

    # Drop rows with any missing feature (small fraction of rows)
    df = df.dropna().reset_index(drop=True)

    # Binary classification: presence of heart disease
    df["target"] = (df["num"] > 0).astype(int)
    df = df.drop(columns=["num"])

    if save:
        DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
        out = DATA_PROCESSED / PROCESSED_FILENAME
        df.to_csv(out, index=False)

    return df


def load_processed(path: Path | None = None) -> pd.DataFrame:
    path = path or (DATA_PROCESSED / PROCESSED_FILENAME)
    if not path.exists():
        download_raw()
        return clean_data()
    return pd.read_csv(path)


def train_test_split_xy(df: pd.DataFrame, test_size: float, random_state: int):
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=["target"])
    y = df["target"]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

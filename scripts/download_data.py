#!/usr/bin/env python
"""Download and preprocess UCI Heart Disease (Cleveland) dataset."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heart_disease.data import clean_data, download_raw


def main() -> None:
    raw = download_raw()
    df = clean_data(raw)
    print(f"Downloaded and cleaned dataset: {len(df)} rows, {len(df.columns)} columns")
    print("Saved to data/processed/heart_disease_clean.csv")


if __name__ == "__main__":
    main()

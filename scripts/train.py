#!/usr/bin/env python
"""Train models, log to MLflow, save production artifact."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heart_disease.eda import run_eda
from heart_disease.training import train_and_select


def main() -> None:
    run_eda()
    name, _, metrics = train_and_select()
    print(f"Selected model: {name}")
    for k, v in sorted(metrics.items()):
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()

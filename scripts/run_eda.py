#!/usr/bin/env python
"""Generate EDA figures and summary."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heart_disease.eda import run_eda


def main() -> None:
    out = run_eda()
    print(f"EDA artifacts written to {out}")


if __name__ == "__main__":
    main()

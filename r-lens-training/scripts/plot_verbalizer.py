#!/usr/bin/env python
"""Plot metrics.jsonl of a verbalizer RL run into <run_dir>/plots.

Usage: .venv/bin/python scripts/plot_verbalizer.py artifacts/runs/<run>
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from verbalize.plots import write_plots  # noqa: E402

if __name__ == "__main__":
    print(f"wrote plots to {write_plots(Path(sys.argv[1]))}")

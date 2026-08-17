#!/usr/bin/env python
"""Train an r-lens (or plain J-lens) for every layer of a model.

Usage:
    python scripts/train_r_lens.py --config configs/train_r_lens_qwen3_5_4b.json

Lenses are stored DESTRUCTIVELY under {output_dir}/{model_name}/ — any
existing lenses for that model are deleted and replaced.
"""
import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


from r_lens import train_lenses  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="Path to a JSON config file")
    ap.add_argument("--lens-type", choices=["r", "j"], default=None,
                    help="Override the config's lens_type")
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    if args.lens_type is not None:
        cfg["lens_type"] = args.lens_type

    # resolve paths relative to the project root
    for key in ("model_path", "documents_path", "output_dir"):
        p = Path(cfg[key])
        if not p.is_absolute():
            cfg[key] = str(PROJECT_ROOT / p)

    train_lenses(cfg)


if __name__ == "__main__":
    main()

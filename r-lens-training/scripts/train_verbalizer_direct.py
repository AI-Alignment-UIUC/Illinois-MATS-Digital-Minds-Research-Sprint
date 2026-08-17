#!/usr/bin/env python
"""Direct supervised r-lens verbalization training (no RL, no SGLang).

Per step: sample episodes, compute on-policy r-lens target distributions from
a bare-document forward, prefill the assistant turn with an introspection
stem, and train with soft cross-entropy at the final position against the
lens distribution. rsLoRA + Adam, one optimizer step per batch.

Usage:
    .venv/bin/python scripts/train_verbalizer_direct.py \
        --config configs/train_direct_verbaliser.json [--run-name NAME]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from train_verbalizer import flatten_config, prepare_init_adapter  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--num-steps", type=int, default=None)
    args = ap.parse_args()

    nested = json.loads(Path(args.config).read_text())
    cfg = flatten_config(nested)
    if args.num_steps is not None:
        cfg["num_steps"] = args.num_steps
    for key in ("model_path", "documents_path", "lens_dir"):
        p = Path(cfg[key])
        if not p.is_absolute():
            cfg[key] = str(PROJECT_ROOT / p)

    run_name = args.run_name or cfg.get("name") or "direct"
    runs_root = PROJECT_ROOT / "artifacts" / "runs"
    run_dir = runs_root / run_name
    if run_dir.exists():
        run_name = f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        run_dir = runs_root / run_name
    cfg["name"] = run_name
    run_dir.mkdir(parents=True)
    (run_dir / "checkpoints").mkdir()
    (run_dir / "config.json").write_text(json.dumps(nested, indent=2))
    print(f"Run directory: {run_dir}", flush=True)
    prepare_init_adapter(cfg, run_dir)

    from r_lens.model_utils import load_model
    from verbalize.direct_trainer import DirectVerbalizerTrainer
    from transformers import AutoTokenizer

    device = f"cuda:{cfg.get('train_gpu', 0)}"
    cfg["device"] = device
    print(f"Loading model on {device} ...", flush=True)
    parts = load_model(cfg["model_path"], cfg.get("dtype", "bfloat16"), device)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"])

    trainer = DirectVerbalizerTrainer(cfg, parts, tokenizer, run_dir)
    trainer.train(cfg.get("num_steps", 1000))


if __name__ == "__main__":
    main()

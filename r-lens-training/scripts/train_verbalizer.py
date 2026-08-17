#!/usr/bin/env python
"""Train Qwen to verbalize its own r-lens vectors, with REINFORCE.

Rollouts run on an SGLang server (inference GPU); the policy trains with a
rank-8 rsLoRA + Adam on the train GPU; the adapter is synced to SGLang after
every optimizer step.

The config is nested:
  top level      model/data paths
  "optim"        all training hyperparameters
  "run"          gpus, run name, sglang port, and "logging" settings

Everything for a run lands in artifacts/runs/<name>/:
  config.json  metrics.jsonl  sglang.log  adapter/  checkpoints/  plots/

Usage:
    .venv/bin/python scripts/train_verbalizer.py \
        --config configs/train_verbalizer_qwen3_5_4b.json [--run-name NAME]
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


def flatten_config(nested: dict) -> dict:
    """Flatten the nested {top, optim, run{logging}} config for the trainer."""
    flat = {k: v for k, v in nested.items() if not isinstance(v, dict)}
    flat.update(nested.get("optim", {}))
    run = dict(nested.get("run", {}))
    logging = run.pop("logging", {})
    flat.update(run)
    flat.update(logging)
    return flat


def prepare_init_adapter(cfg: dict, run_dir: Path) -> int | None:
    """Snapshot cfg['init_adapter'] into the run dir, read-only, and point the
    config at the snapshot. Returns the adapter's LoRA rank (or None if no
    init adapter). Used by every launcher so relative paths always resolve
    against the project root and later edits to the source can't leak in."""
    if not cfg.get("init_adapter"):
        return None
    import shutil
    src = Path(cfg["init_adapter"])
    if not src.is_absolute():
        src = PROJECT_ROOT / src
    if not (src / "adapter_model.safetensors").exists():
        sys.exit(f"init_adapter {src} has no adapter_model.safetensors")
    dst = run_dir / "init_adapter"
    shutil.copytree(src, dst)
    for p in dst.rglob("*"):
        if p.is_file():
            p.chmod(0o444)
    cfg["init_adapter"] = str(dst)
    rank = json.loads((dst / "adapter_config.json").read_text()).get("r")
    print(f"Warm-start adapter copied (read-only) from {src} (rank {rank})",
          flush=True)
    return rank


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-name", default=None,
                    help="Overrides run.name; a timestamp is appended if the "
                         "run directory already exists")
    ap.add_argument("--num-steps", type=int, default=None)
    ap.add_argument("--server-url", default=None,
                    help="Attach to an already-running SGLang server instead of launching one")
    args = ap.parse_args()

    nested = json.loads(Path(args.config).read_text())
    cfg = flatten_config(nested)
    if args.num_steps is not None:
        cfg["num_steps"] = args.num_steps
    for key in ("model_path", "documents_path", "lens_dir"):
        p = Path(cfg[key])
        if not p.is_absolute():
            cfg[key] = str(PROJECT_ROOT / p)

    run_name = args.run_name or cfg.get("name") or "verbalizer"
    runs_root = PROJECT_ROOT / "artifacts" / "runs"
    run_dir = runs_root / run_name
    if run_dir.exists():
        run_name = f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        run_dir = runs_root / run_name
    cfg["name"] = run_name
    adapter_dir = run_dir / "adapter"
    run_dir.mkdir(parents=True)
    adapter_dir.mkdir()
    (run_dir / "checkpoints").mkdir()
    (run_dir / "config.json").write_text(json.dumps(nested, indent=2))
    print(f"Run directory: {run_dir}", flush=True)

    init_rank = prepare_init_adapter(cfg, run_dir)

    from r_lens.model_utils import load_model
    from verbalize.sglang_client import SGLangServer
    from verbalize.trainer import VerbalizerTrainer
    from transformers import AutoTokenizer

    device = f"cuda:{cfg.get('train_gpu', 0)}"
    cfg["device"] = device
    print(f"Loading model on {device} ...", flush=True)
    parts = load_model(cfg["model_path"], cfg.get("dtype", "bfloat16"), device)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"])

    if cfg.get("backend") == "hf":
        # rollouts sampled by the policy itself on the training GPU; needed
        # for models sglang cannot serve with LoRA (non-uniform layer shapes)
        from verbalize.hf_sampler import HFSampler
        server = HFSampler(parts, tokenizer, device,
                           batch_size=cfg.get("rollout_batch_size", 64))
        print("Using HF-generate rollout backend (no SGLang)", flush=True)
    elif args.server_url is None:
        print(f"Launching SGLang server on GPU {cfg['inference_gpu']} ...", flush=True)
        server = SGLangServer(
            python=str(PROJECT_ROOT / ".venv-sglang" / "bin" / "python"),
            model_path=cfg["model_path"],
            gpu_id=cfg["inference_gpu"],
            port=cfg.get("sglang_port", 30012),
            # a warm-start adapter's rank overrides the config: sglang's LoRA
            # buffers must fit the adapter actually being loaded
            max_lora_rank=init_rank or cfg.get("lora_rank", 8),
            lora_target_modules=tuple(cfg.get("lora_target_modules",
                                              ("gate_proj", "up_proj", "down_proj"))),
            log_path=str(run_dir / "sglang.log"),
        )
        try:
            server.wait_ready()
        except BaseException:
            server.shutdown()  # never orphan a half-started server
            raise
        print("SGLang server ready", flush=True)
    else:
        server = SGLangServer.__new__(SGLangServer)
        server.base = args.server_url
        server.proc = None
        server.lora_name = None
        server.lora_version = 0
        server.log_fh = None

    try:
        trainer = VerbalizerTrainer(cfg, parts, tokenizer, server, run_dir, adapter_dir)
        trainer.train(cfg.get("num_steps", 200))
    finally:
        if getattr(server, "proc", None) is not None:
            server.shutdown()


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Web chat with the model, with toggleable thinking and r-lens inspection.

On startup you pick, in the terminal: the model (from data/models/), whether
to equip a trained LoRA (from artifacts/loras/<model>/), and which GPU to run
on (menu shows current utilization). Then a web UI is served where you can
chat (thinking on/off), select a layer range i<=j, and click any token to see
the top-10 tokens of the accumulated r-lens readout over those layers.

Usage: .venv/bin/python scripts/chat_server.py [--port 7860]
"""
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def pick(title: str, options: list[str], allow_none: bool = False) -> str | None:
    print(f"\n{title}")
    if allow_none:
        print("  [0] none")
    for i, opt in enumerate(options, 1):
        print(f"  [{i}] {opt}")
    while True:
        raw = input("> ").strip()
        if raw.isdigit():
            n = int(raw)
            if allow_none and n == 0:
                return None
            if 1 <= n <= len(options):
                return options[n - 1]
        print("pick a number from the menu")


def gpu_menu() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True).stdout
    rows = [line.split(", ") for line in out.strip().splitlines()]
    options = [f"GPU {r[0]}: {int(r[1]):6d}/{r[2]} MiB used, {r[3]:>3s}% util"
               for r in rows]
    choice = pick("Select GPU:", options)
    return int(choice.split(":")[0].split()[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--api-key", action="append", default=[],
                    help="API mode: allowed bearer key (repeatable); "
                         "auto-generated if none given")
    args = ap.parse_args()

    mode = pick("Select mode:", [
        "local chat UI (this machine only)",
        "public API server (share with teammates; token-authenticated)",
    ])
    api_mode = mode.startswith("public")

    models_dir = PROJECT_ROOT / "data" / "models"
    models = sorted(p.name for p in models_dir.iterdir() if p.is_dir())
    if not models:
        sys.exit(f"no models in {models_dir}")
    model_name = pick("Select model:", models)

    lora_root = PROJECT_ROOT / "artifacts" / "runs"
    loras = sorted(
        str(p.parent.relative_to(lora_root))
        for p in lora_root.rglob("adapter_config.json")
        if (p.parent / "adapter_model.safetensors").exists()
    ) if lora_root.exists() else []
    # API mode serves ALL adapters (clients pick per request); UI mode equips one
    lora_choice = None
    if not api_mode and loras:
        lora_choice = pick("Equip a trained LoRA?", loras, allow_none=True)

    gpu = gpu_menu()
    device = f"cuda:{gpu}"

    if api_mode:
        run_api_server(args, model_name, models_dir, lora_root, loras, device)
        return

    print(f"\nLoading {model_name} on {device}"
          + (f" with LoRA {lora_choice}" if lora_choice else "") + " ...")

    import torch
    from transformers import AutoTokenizer
    from r_lens.model_utils import load_model
    from chat.server import ChatState, build_app

    parts = load_model(str(models_dir / model_name), "bfloat16", device)
    tokenizer = AutoTokenizer.from_pretrained(str(models_dir / model_name))
    if lora_choice is not None:
        from peft import PeftModel
        parts.model = PeftModel.from_pretrained(
            parts.model, str(lora_root / lora_choice), torch_device=device)
        parts.model.eval()

    lens_dir = PROJECT_ROOT / "artifacts" / "jlenses" / model_name
    lens_mats = None
    if lens_dir.exists():
        from safetensors.torch import load_file
        mats = [load_file(str(lens_dir / f"layer_{i:02d}.safetensors"))["weight"]
                for i in range(len(parts.layers))]
        lens_mats = torch.stack(mats).float().to(device)
        print(f"Loaded {len(mats)} r-lenses from {lens_dir}")
    else:
        print(f"No lenses found at {lens_dir}; lens readouts disabled")

    state = ChatState(parts, tokenizer, lens_mats,
                      {"model_name": model_name, "lora": lora_choice}, device)
    app = build_app(state)

    import uvicorn
    print(f"\nChat UI: http://{args.host}:{args.port}\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


def run_api_server(args, model_name, models_dir, lora_root, loras, device):
    import secrets
    import torch
    from transformers import AutoTokenizer
    from r_lens.model_utils import load_model
    from chat.api_server import ApiState, build_api_app

    keys = list(args.api_key)
    if not keys:
        keys = [secrets.token_urlsafe(24) for _ in range(3)]
    print(f"\nLoading {model_name} on {device} with "
          f"{len(loras)} adapters for API serving ...")
    parts = load_model(str(models_dir / model_name), "bfloat16", device)
    tokenizer = AutoTokenizer.from_pretrained(str(models_dir / model_name))

    # adapters must be compatible with this base model: filter by hidden size
    registry = {}
    for rel in loras:
        cfg_path = lora_root / rel / "adapter_config.json"
        try:
            import json as _json
            base = _json.loads(cfg_path.read_text()).get(
                "base_model_name_or_path", "") or ""
        except OSError:
            continue
        if model_name in base or base == "":
            registry[rel.replace("/", "__")] = lora_root / rel

    lens_dir = PROJECT_ROOT / "artifacts" / "jlenses" / model_name
    lens_mats = None
    if lens_dir.exists():
        from safetensors.torch import load_file
        mats = [load_file(str(lens_dir / f"layer_{i:02d}.safetensors"))["weight"]
                for i in range(len(parts.layers))]
        lens_mats = torch.stack(mats).float().to(device)

    state = ApiState(parts, tokenizer, lens_mats, model_name, device,
                     registry, keys)
    app = build_api_app(state)

    host = args.host if args.host != "127.0.0.1" else "0.0.0.0"
    print("\n=== PUBLIC API SERVER ===")
    print(f"Listening on http://{host}:{args.port}  (docs at /  health at /v1/health)")
    print(f"Adapters served: ['base'] + {list(registry)}")
    print("API keys (give one to each teammate; NOT logged anywhere else):")
    for k in keys:
        print(f"  {k}")
    print("NOTE: traffic is plain HTTP; if the campus firewall blocks the "
          "port, tunnel it (e.g. ssh -R or cloudflared).\n")

    import uvicorn
    uvicorn.run(app, host=host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

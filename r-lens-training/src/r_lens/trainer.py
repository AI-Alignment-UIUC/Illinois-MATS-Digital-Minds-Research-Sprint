"""Fit J-lenses / r-lenses for every layer of a model.

The J-lens for layer L is the expected same-position Jacobian of the final
residual stream (input to the final norm) with respect to the residual stream
after layer L, averaged over contexts and token positions:

    W_L = E_t [ d h_final(t) / d h_L(t) ]

The lens readout is then unembed(final_norm(W_L @ h_L)) — every downstream
layer replaced by a single linear map followed by the model's own unembedding.
The r-lens is fit identically but with LRP-modified backward passes (see
lrp.py).

Estimation: full per-token Jacobians are too expensive, so we use an unbiased
random-probe estimator. Draw u(t) ~ N(0, I) independently per token position,
backprop the scalar sum_t <u(t), h_final(t)> to every layer's residual in one
backward pass, giving g_L(t) = sum_p J(p,t)^T u(p), and accumulate

    A_L += u(t) g_L(t)^T

Since E[u(t) u(p)^T] = delta_tp * I, cross-position terms vanish in
expectation and A_L / N -> W_L.
"""
import json
import shutil
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

from .data import load_documents, batches
from .lrp import apply_lrp_rules
from .model_utils import load_model


def _capture_residuals(layers):
    """Register forward hooks that stash each decoder layer's output residual."""
    residuals = []
    handles = []

    def hook(_module, _inputs, output):
        h = output[0] if isinstance(output, tuple) else output
        residuals.append(h)

    for layer in layers:
        handles.append(layer.register_forward_hook(hook))
    return residuals, handles


def train_lenses(cfg: dict) -> Path:
    device = cfg.get("device", "cuda:0")
    torch.manual_seed(cfg.get("seed", 0))

    parts = load_model(cfg["model_path"], cfg.get("dtype", "bfloat16"), device)
    lens_type = cfg.get("lens_type", "r")
    if lens_type == "r":
        patched = apply_lrp_rules(parts.layers)
        print(f"Applied LRP rules: {patched}")
    elif lens_type != "j":
        raise ValueError(f"lens_type must be 'r' or 'j', got {lens_type!r}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"])

    docs = load_documents(cfg["documents_path"], cfg.get("max_documents"))
    print(f"Loaded {len(docs)} documents")

    num_layers = len(parts.layers)
    d = parts.hidden_size
    num_probes = cfg.get("num_probes", 4)
    accum = torch.zeros(num_layers - 1, d, d, dtype=torch.float32, device=device)
    total_probe_tokens = 0

    residuals, handles = _capture_residuals(parts.layers)

    batch_iter = batches(docs, tokenizer, cfg.get("batch_size", 4),
                         cfg.get("max_seq_len", 512), device)
    n_batches = (len(docs) + cfg.get("batch_size", 4) - 1) // cfg.get("batch_size", 4)
    t0 = time.time()

    for step, (input_ids, attention_mask) in enumerate(batch_iter):
        residuals.clear()
        with torch.enable_grad():
            # All params are frozen; making the embedding weight require grad
            # roots the autograd graph without touching the forward call. This
            # works for models whose extra inputs (e.g. gemma's per-layer
            # embeddings) are derived from input_ids inside the forward, where
            # an inputs_embeds= override would not.
            parts.text_model.embed_tokens.weight.requires_grad_(True)
            parts.text_model(input_ids=input_ids, attention_mask=attention_mask)
            if len(residuals) != num_layers:
                raise RuntimeError(
                    f"Captured {len(residuals)} residuals, expected {num_layers}")
            target = residuals[-1]  # final residual stream, pre final-norm
            mask = attention_mask.to(torch.float32)

            for k in range(num_probes):
                u = torch.randn_like(target, dtype=torch.float32) * mask.unsqueeze(-1)
                scalar = (target.float() * u).sum()
                grads = torch.autograd.grad(
                    scalar, residuals[:-1],
                    retain_graph=(k < num_probes - 1))
                for i, g in enumerate(grads):
                    accum[i] += torch.einsum(
                        "bti,btj->ij", u, g.float() * mask.unsqueeze(-1))
                total_probe_tokens += int(mask.sum().item())

        # free the graph before the next forward
        del target, grads
        residuals.clear()
        if (step + 1) % 10 == 0 or step == n_batches - 1:
            dt = time.time() - t0
            print(f"batch {step + 1}/{n_batches}  "
                  f"({total_probe_tokens} probe-tokens, {dt:.0f}s)", flush=True)

    for h in handles:
        h.remove()

    lenses = accum / max(total_probe_tokens, 1)

    # ---- destructive save: wipe and rewrite artifacts/jlenses/{model_name} ----
    out_dir = Path(cfg["output_dir"]) / cfg["model_name"]
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    for i in range(num_layers - 1):
        save_file({"weight": lenses[i].cpu().contiguous()},
                  str(out_dir / f"layer_{i:02d}.safetensors"))
    # The lens for the last layer's output is exactly the identity.
    save_file({"weight": torch.eye(d)},
              str(out_dir / f"layer_{num_layers - 1:02d}.safetensors"))

    meta = {
        "lens_type": lens_type,
        "model_name": cfg["model_name"],
        "model_path": cfg["model_path"],
        "num_layers": num_layers,
        "hidden_size": d,
        "num_documents": len(docs),
        "num_probes": num_probes,
        "max_seq_len": cfg.get("max_seq_len", 512),
        "total_probe_tokens": total_probe_tokens,
        "seed": cfg.get("seed", 0),
        "readout": "lm_head(final_norm(W @ h))",
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Saved {num_layers} lenses to {out_dir}")
    return out_dir

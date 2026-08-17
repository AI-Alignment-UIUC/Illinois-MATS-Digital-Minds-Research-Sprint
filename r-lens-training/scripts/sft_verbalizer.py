#!/usr/bin/env python
"""SFT warmstart on introspective r-lens thinking traces.

Reconstructs each JSONL trace into a full chat sample:
  user turn    = the same RL episode prompt (document, marked token, layer)
  assistant    = <think>\n{thinking}\n</think>\n\n{answer}<|im_end|>
and fine-tunes a rank-8 rsLoRA with Adam, loss on assistant tokens only.
The goal is to break the model's reflexive "I have no access to my
activations" refusals before RL / direct training.

Usage:
    .venv/bin/python scripts/sft_verbalizer.py \
        --config configs/sft_verbalizer.json [--run-name NAME]
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from train_verbalizer import flatten_config, prepare_init_adapter  # noqa: E402


def build_samples(path, tokenizer, num_layers):
    """Reconstruct (prompt_ids, completion_ids) pairs from the trace JSONL.

    Each trace yields TWO samples: one with the full prompt (r-lens
    explanation included) and one with the explanation stripped, so the model
    internalizes what the r-lens is rather than relying on the preamble.
    """
    from verbalize.episodes import Episode, completion_formatter, render_prompt
    fmt = completion_formatter(tokenizer)
    samples, skipped = [], 0
    for line in open(path):
        row = json.loads(line)
        ids = tokenizer(row["document"], add_special_tokens=False)["input_ids"]
        pos = row["pos"]
        if pos >= len(ids) or tokenizer.decode([ids[pos]]) != row["token"]:
            skipped += 1  # decode/encode roundtrip moved the token; drop it
            continue
        ep = Episode(doc_id=row["doc_id"], token_ids=ids, pos=pos,
                     layer=row["layer"])
        completion = fmt(row["thinking"].rstrip(), row["answer"].strip())
        completion_ids = tokenizer(completion, add_special_tokens=False)["input_ids"]
        for include_explanation in (True, False):
            prompt = render_prompt(ep, tokenizer, num_layers,
                                   include_explanation=include_explanation)
            chat = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False, add_generation_prompt=True, enable_thinking=True)
            prompt_ids = tokenizer(chat, add_special_tokens=False)["input_ids"]
            samples.append((prompt_ids, completion_ids))
    if skipped:
        print(f"WARNING: skipped {skipped} traces on tokenization roundtrip",
              flush=True)
    return samples


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    nested = json.loads(Path(args.config).read_text())
    cfg = flatten_config(nested)
    for key in ("model_path", "traces_path"):
        p = Path(cfg[key])
        if not p.is_absolute():
            cfg[key] = str(PROJECT_ROOT / p)

    run_name = args.run_name or cfg.get("name") or "sft_warmstart"
    run_dir = PROJECT_ROOT / "artifacts" / "runs" / run_name
    if run_dir.exists():
        run_name = f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        run_dir = PROJECT_ROOT / "artifacts" / "runs" / run_name
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(nested, indent=2))
    print(f"Run directory: {run_dir}", flush=True)
    prepare_init_adapter(cfg, run_dir)

    import torch
    from transformers import AutoTokenizer
    from r_lens.model_utils import load_model
    from verbalize.trainer import make_policy
    from verbalize.plots import write_plots

    device = f"cuda:{cfg.get('train_gpu', 0)}"
    cfg["device"] = device
    parts = load_model(cfg["model_path"], cfg.get("dtype", "bfloat16"), device)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"])
    samples = build_samples(cfg["traces_path"], tokenizer, len(parts.layers))
    print(f"{len(samples)} SFT samples", flush=True)

    policy = make_policy(parts, cfg)
    parts.text_model.config.use_cache = False
    optimizer = torch.optim.Adam(
        [p for p in policy.parameters() if p.requires_grad],
        lr=cfg.get("lr", 5e-5), betas=tuple(cfg.get("adam_betas", (0.8, 0.95))))
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    rng = random.Random(cfg.get("seed", 0))
    metrics_fh = (run_dir / "metrics.jsonl").open("a")

    micro = cfg.get("micro_batch_size", 8)
    batch_size = cfg.get("batch_size", 32)
    epochs = cfg.get("epochs", 3)
    steps_per_epoch = (len(samples) + batch_size - 1) // batch_size
    total_steps = epochs * steps_per_epoch
    step = 0
    policy.train()
    for epoch in range(1, epochs + 1):
        order = list(range(len(samples)))
        rng.shuffle(order)
        for begin in range(0, len(order), batch_size):
            batch = [samples[i] for i in order[begin:begin + batch_size]]
            optimizer.zero_grad(set_to_none=True)
            loss_total, token_total = 0.0, sum(len(c) for _, c in batch)
            for mb in range(0, len(batch), micro):
                chunk = batch[mb:mb + micro]
                width = max(len(p) + len(c) for p, c in chunk)
                ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
                mask = torch.zeros((len(chunk), width), dtype=torch.long)
                weights = torch.zeros((len(chunk), width - 1))
                for i, (p, c) in enumerate(chunk):
                    seq = p + c
                    ids[i, :len(seq)] = torch.tensor(seq)
                    mask[i, :len(seq)] = 1
                    weights[i, len(p) - 1:len(seq) - 1] = 1.0 / token_total
                ids, mask, weights = (t.to(device) for t in (ids, mask, weights))
                hidden = parts.text_model(
                    input_ids=ids, attention_mask=mask).last_hidden_state
                logits = parts.lm_head(hidden[:, :-1]).float()
                logp = torch.log_softmax(logits, dim=-1)
                token_logp = logp.gather(
                    -1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
                loss = -(weights * token_logp).sum()
                loss.backward()
                loss_total += loss.item()
                del hidden, logits, logp
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in policy.parameters() if p.requires_grad],
                cfg.get("max_grad_norm", 1.0))
            optimizer.step()
            step += 1
            row = {"step": step, "epoch": epoch, "loss": loss_total,
                   "grad_norm": float(grad_norm)}
            metrics_fh.write(json.dumps(row) + "\n")
            metrics_fh.flush()
            print(f"epoch {epoch}/{epochs} step {step}/{total_steps} "
                  f"loss={loss_total:.4f} grad_norm={grad_norm:.3f}", flush=True)
        policy.save_pretrained(str(run_dir / "adapter"))
    write_plots(run_dir, [("loss", "SFT loss (mean nats/token)"),
                          ("grad_norm", "Gradient norm")])
    print(f"Saved adapter to {run_dir / 'adapter'}", flush=True)


if __name__ == "__main__":
    main()

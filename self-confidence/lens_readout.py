#!/usr/bin/env python3
"""r-lens internal readout collection (FINDINGS.md key finding 4).

Question: is an item's actual consistency readable from the model's internal
state at the moment it is about to answer, and does that internal signal
survive the RL training that degrades the verbal report?

Signal: POST /v1/lens at the LAST prompt token (the position whose next-token
distribution is the answer), averaged over a layer band. Concentration of the
readout (top-1 prob; entropy) = how "decided" the computation already is.

Steps:
  python3 lens_readout.py --verify   # pick the chat template empirically
  python3 lens_readout.py --sweep    # 48 items x 6 adapters x 4 layer bands

Sweep appends to runs/lens-readout.jsonl keyed by call id; reruns skip
completed calls (same convention as run_bench.py). Analysis lives in
analyze.py, which folds the lens tables into RESULTS.md.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from run_bench import http_json, load_env

HERE = Path(__file__).parent
RUNS = HERE / "runs"
OUT = RUNS / "lens-readout.jsonl"

# adapter -> the model key its ground truth was collected under
ADAPTERS = {
    "base": "q35-4b-base",
    "verbalizer_qwen_rl_final__init_adapter": "q35-4b-rl-init",
    "verbalizer_qwen_rl_final__checkpoints__adapter_step_25": "q35-4b-rl-step25",
    "verbalizer_qwen_rl_final__checkpoints__adapter_step_50": "q35-4b-rl-step50",
    "verbalizer_qwen_rl_final__checkpoints__adapter_step_75": "q35-4b-rl-step75",
    "verbalizer_qwen_rl_final__adapter": "q35-4b-rl-final",
}
BANDS = [(6, 10), (14, 18), (22, 26), (30, 32)]
TOP_K = 50
CONCURRENCY = 6

# ChatML, as Qwen serves it; T2 = thinking disabled leaves an empty think block
T1 = "<|im_start|>user\n{item}<|im_end|>\n<|im_start|>assistant\n"
T2 = T1 + "<think>\n\n</think>\n\n"


def rlens(path, body):
    base = os.environ.get("RLENS_BASE_URL", "").rstrip("/")
    key = os.environ.get("RLENS_API_KEY", "")
    last = None
    for attempt in range(5):
        try:
            return http_json(f"{base}{path}", body,
                             {"Authorization": f"Bearer {key}"}, timeout=300)
        except Exception as e:  # noqa: BLE001
            last = repr(e)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"rlens {path} failed after retries: {last}")


def lens_last(text, layer_lo, layer_hi, adapter, n_tokens=None):
    """Lens at the last token of text. Returns (n_tokens, top)."""
    if n_tokens is None:
        n_tokens = len(rlens("/v1/lens", {"text": text, "position": 0,
                                          "layer_lo": layer_lo, "layer_hi": layer_hi,
                                          "top_k": 1, "adapter": adapter})["tokens"])
    r = rlens("/v1/lens", {"text": text, "position": n_tokens - 1,
                           "layer_lo": layer_lo, "layer_hi": layer_hi,
                           "top_k": TOP_K, "adapter": adapter})
    return n_tokens, r["top"]


def items():
    return json.loads((HERE / "items.json").read_text())["items"]


def verify():
    """Which template makes the near-final-layer lens reproduce greedy decoding?"""
    probe = [it for it in items() if it["id"] in ("d1", "d2", "d3", "p1", "s7", "r1")]
    for name, tmpl in (("plain", T1), ("empty-think", T2)):
        hits = 0
        for it in probe:
            text = tmpl.format(item=it["text"])
            _, top = lens_last(text, 30, 32, "base")
            chat = rlens("/v1/chat", {"messages": [{"role": "user", "content": it["text"]}],
                                      "adapter": "base", "temperature": 0.0,
                                      "max_new_tokens": 8, "thinking": False})
            lens_tok = top[0]["token"].strip().lower()
            comp = chat["completion"].strip().lower()
            ok = bool(lens_tok) and comp.startswith(lens_tok)
            hits += ok
            print(f"  {name:12s} {it['id']:3s} lens={top[0]['token']!r:14s} "
                  f"p={top[0]['prob']:.2f} chat={comp[:24]!r} {'MATCH' if ok else 'x'}")
        print(f"{name}: {hits}/{len(probe)} greedy-first-token matches\n")


def sweep(template):
    tmpl = {"plain": T1, "empty-think": T2}[template]
    done = set()
    if OUT.exists():
        with open(OUT) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["cid"])
                except (json.JSONDecodeError, KeyError):
                    continue
    RUNS.mkdir(exist_ok=True)
    out = open(OUT, "a")
    lock = __import__("threading").Lock()
    ntok_cache = {}

    def one(adapter, it, lo, hi):
        text = tmpl.format(item=it["text"])
        n = ntok_cache.get(it["id"])
        n, top = lens_last(text, lo, hi, adapter, n_tokens=n)
        ntok_cache[it["id"]] = n
        return {"adapter": adapter, "model": ADAPTERS[adapter],
                "item": it["id"], "band": [lo, hi], "template": template,
                "n_tokens": n, "top": top}

    jobs = {}
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        for adapter in ADAPTERS:
            for it in items():
                for lo, hi in BANDS:
                    cid = hashlib.sha1(
                        f"{adapter}|{it['id']}|{lo}-{hi}|{template}".encode()).hexdigest()[:16]
                    if cid in done:
                        continue
                    jobs[pool.submit(one, adapter, it, lo, hi)] = cid
        n_ok = n_err = 0
        for fut in as_completed(jobs):
            cid = jobs[fut]
            try:
                rec = fut.result()
                rec["cid"] = cid
                with lock:
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                n_ok += 1
            except Exception as e:  # noqa: BLE001
                n_err += 1
                print(f"  ERROR {cid}: {e}", file=sys.stderr)
    print(f"sweep ok={n_ok} err={n_err} skipped={len(done)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--template", default="empty-think", choices=["plain", "empty-think"])
    args = ap.parse_args()
    load_env()
    if args.verify:
        verify()
    if args.sweep:
        sweep(args.template)
    if not (args.verify or args.sweep):
        print(__doc__)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Download short documents (30-100 tokens) for lens-verbalization RL.

Streams fresh documents from fineweb-edu (skipping the ones already used for
lens training), truncates each to a random 30-100 token length measured with
the model's tokenizer, and writes JSONL to data/documents/short_documents.jsonl.
"""
import argparse
import json
import random
import sys
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--subset", default="sample-10BT")
    ap.add_argument("--split", default="train")
    ap.add_argument("--tokenizer", default=str(PROJECT_ROOT / "data" / "models" / "Qwen3.5-4B"))
    ap.add_argument("--num-documents", type=int, default=1000)
    ap.add_argument("--skip", type=int, default=2000,
                    help="Skip this many stream rows first, so documents are new "
                         "relative to the lens-training set.")
    ap.add_argument("--min-tokens", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default=str(PROJECT_ROOT / "data" / "documents" / "short_documents.jsonl"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, args.subset, split=args.split, streaming=True)
    ds = ds.skip(args.skip)

    n = 0
    with out_path.open("w") as f:
        for row in ds:
            text = row.get("text", "").strip()
            if len(text) < 200:
                continue
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) < args.min_tokens:
                continue
            length = rng.randint(args.min_tokens, args.max_tokens)
            ids = ids[:length]
            if len(ids) < args.min_tokens:
                continue
            f.write(json.dumps({
                "id": n,
                "text": tokenizer.decode(ids),
                "num_tokens": len(ids),
            }) + "\n")
            n += 1
            if n >= args.num_documents:
                break

    print(f"Wrote {n} short documents to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()

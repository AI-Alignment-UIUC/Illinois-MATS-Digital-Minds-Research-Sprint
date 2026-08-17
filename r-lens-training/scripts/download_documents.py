#!/usr/bin/env python
"""Download training documents for lens fitting.

Streams documents from a HuggingFace text corpus (default: fineweb-edu) and
writes them as JSONL ({"id": ..., "text": ...}) to data/documents/.
"""
import argparse
import json
from pathlib import Path

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--subset", default="sample-10BT")
    ap.add_argument("--split", default="train")
    ap.add_argument("--num-documents", type=int, default=1000)
    ap.add_argument("--min-chars", type=int, default=1000,
                    help="Skip documents shorter than this many characters.")
    ap.add_argument("--output", default=str(PROJECT_ROOT / "data" / "documents" / "documents.jsonl"))
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.dataset, args.subset, split=args.split, streaming=True)

    n = 0
    with out_path.open("w") as f:
        for row in ds:
            text = row.get("text", "")
            if len(text) < args.min_chars:
                continue
            f.write(json.dumps({"id": n, "text": text}) + "\n")
            n += 1
            if n >= args.num_documents:
                break

    print(f"Wrote {n} documents to {out_path}")


if __name__ == "__main__":
    main()

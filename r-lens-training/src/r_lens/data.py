"""Document loading and batching for lens training."""
import json
from pathlib import Path

import torch


def load_documents(path: str | Path, max_documents: int | None = None) -> list[str]:
    docs = []
    with open(path) as f:
        for line in f:
            docs.append(json.loads(line)["text"])
            if max_documents is not None and len(docs) >= max_documents:
                break
    return docs


def batches(docs: list[str], tokenizer, batch_size: int, max_seq_len: int, device: str):
    """Yield (input_ids, attention_mask) batches of truncated documents.

    A BOS token is prepended when the tokenizer defines one but does not add
    it itself (e.g. gemma, whose chat template carries a literal <bos>):
    running such models without BOS is off-distribution and would skew the
    fitted lenses.
    """
    import torch
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    bos = tokenizer.bos_token_id
    for i in range(0, len(docs), batch_size):
        enc = tokenizer(
            docs[i:i + batch_size],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )
        ids, mask = enc["input_ids"], enc["attention_mask"]
        if bos is not None and not (ids[:, 0] == bos).all():
            ids = torch.cat([torch.full((ids.shape[0], 1), bos,
                                        dtype=ids.dtype), ids], dim=1)
            mask = torch.cat([torch.ones((mask.shape[0], 1),
                                         dtype=mask.dtype), mask], dim=1)
        yield ids.to(device), mask.to(device)

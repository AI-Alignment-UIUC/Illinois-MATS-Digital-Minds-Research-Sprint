"""Reward = r-lens activation of the verbalized token.

For each episode (doc, pos, layer) we run the CURRENT policy (LoRA active,
on-policy targets) over the short document, read the residual stream after
`layer`, apply that layer's r-lens matrix followed by the model's final norm +
unembedding, and softmax. The reward for an answered token is its probability
under this readout, normalized by the top probability so a perfect answer
scores 1 at every layer (early-layer lens distributions are diffuse; without
the normalization those episodes would carry almost no reward signal).
Unparseable answers get `format_penalty`.
"""
from pathlib import Path

import torch
from safetensors.torch import load_file


class LensBank:
    """All per-layer lens matrices, kept on the training device."""

    def __init__(self, lens_dir: str | Path, num_layers: int, device: str,
                 dtype: torch.dtype = torch.float32):
        self.device = device
        mats = []
        for i in range(num_layers):
            w = load_file(str(Path(lens_dir) / f"layer_{i:02d}.safetensors"))["weight"]
            mats.append(w.to(dtype))
        self.mats = torch.stack(mats).to(device)  # [L, d, d]

    def apply(self, hidden: torch.Tensor, layers: torch.Tensor) -> torch.Tensor:
        """hidden [N, d] at 1-indexed `layers` [N] -> lens-mapped hidden [N, d]."""
        w = self.mats[layers - 1]  # [N, d, d]
        return torch.bmm(w, hidden.float().unsqueeze(-1)).squeeze(-1)


class LensRewardModel:
    def __init__(self, parts, lens_bank: LensBank, tokenizer, *,
                 format_penalty: float = -1.5):
        self.parts = parts
        self.bank = lens_bank
        self.tokenizer = tokenizer
        self.format_penalty = format_penalty

    @torch.no_grad()
    def lens_probs(self, episodes) -> torch.Tensor:
        """Full lens readout distribution per episode: [N, vocab] (float32).

        A BOS token is prepended (and probe positions shifted) when the
        tokenizer defines one but does not add it itself, matching how the
        lenses are fitted (see r_lens.data.batches).
        """
        parts = self.parts
        device = self.bank.device
        pad_id = self.tokenizer.pad_token_id or 0
        bos = self.tokenizer.bos_token_id
        offset = 1 if bos is not None else 0
        max_len = max(len(ep.token_ids) for ep in episodes) + offset
        ids = torch.full((len(episodes), max_len), pad_id, dtype=torch.long)
        mask = torch.zeros((len(episodes), max_len), dtype=torch.long)
        for i, ep in enumerate(episodes):
            row = ([bos] if offset else []) + list(ep.token_ids)
            ids[i, :len(row)] = torch.tensor(row)
            mask[i, :len(row)] = 1
        ids, mask = ids.to(device), mask.to(device)

        residuals = []
        handles = [
            layer.register_forward_hook(
                lambda _m, _i, out: residuals.append(out[0] if isinstance(out, tuple) else out))
            for layer in parts.layers
        ]
        try:
            parts.text_model(input_ids=ids, attention_mask=mask)
        finally:
            for h in handles:
                h.remove()

        stacked = torch.stack(residuals)  # [L, N, T, d]
        rows = torch.arange(len(episodes), device=device)
        layers = torch.tensor([ep.layer for ep in episodes], device=device)
        positions = torch.tensor([ep.pos + offset for ep in episodes],
                                 device=device)
        h = stacked[layers - 1, rows, positions]  # [N, d]
        mapped = self.bank.apply(h, layers)
        normed = parts.final_norm(mapped.to(stacked.dtype))
        logits = parts.unembed(normed).float()
        return torch.softmax(logits, dim=-1)

    def rewards(self, episodes, token_ids: list[int | None]
                ) -> tuple[list[float], dict, list[bool]]:
        """Per-trajectory rewards for groups of trajectories over `episodes`.

        `token_ids` is flat, len(episodes) * group_size, episode-major.
        Returns (rewards, stats, per-row top-1 hit flags) so the caller can
        recompute accuracy stats after any reward overrides (e.g. truncation).
        """
        group = len(token_ids) // len(episodes)
        probs = self.lens_probs(episodes)  # [N, vocab]
        top_prob, top_id = probs.max(dim=-1)
        rewards, hits = [], []
        for row, tok in enumerate(token_ids):
            ep_index = row // group
            if tok is None:
                rewards.append(self.format_penalty)
                hits.append(False)
                continue
            p = probs[ep_index, tok].item()
            p = p / max(top_prob[ep_index].item(), 1e-9)
            rewards.append(p)
            hits.append(tok == top_id[ep_index].item())
        stats = {
            "top1_acc": sum(hits) / len(token_ids),
            "answered_frac": sum(t is not None for t in token_ids) / len(token_ids),
            "top_prob_mean": top_prob.mean().item(),
        }
        return rewards, stats, hits

"""Direct (supervised) r-lens verbalization training.

No RL, no sampling, no SGLang. Each step:

1. Sample a batch of (document, position, layer) episodes.
2. Compute each episode's ON-POLICY r-lens target distribution: a no-grad
   forward of the current policy over the bare document, residual read at
   (pos, layer), lens matrix + final norm + unembedding, softmax.
3. Build the chat prompt and prefill the assistant turn with "Hmm, It feels
   like at pos {pos} and layer {layer} the activations have the shape of:".
4. One forward over the prefilled sequence; take the logits at the very last
   position and apply soft cross-entropy against the lens target
   distribution. One Adam step per batch (micro-batches accumulate).

The headline metric is prob_top1: the probability the model assigns to the
lens argmax token at the answer slot.
"""
import json
import time
from pathlib import Path

import torch

from .episodes import EpisodeSampler, load_short_documents, render_prompt
from .lens_reward import LensBank, LensRewardModel
from .trainer import make_policy

PREFILL = ("Hmm, It feels like at pos {pos} and layer {layer} "
           "the activations have the shape of:")


class DirectVerbalizerTrainer:
    def __init__(self, cfg: dict, parts, tokenizer, run_dir: Path):
        self.cfg = cfg
        self.parts = parts
        self.tokenizer = tokenizer
        self.run_dir = run_dir
        self.device = cfg.get("device", "cuda:0")
        self.num_layers = len(parts.layers)

        docs = load_short_documents(cfg["documents_path"], tokenizer,
                                    cfg.get("max_documents"))
        layer_range = cfg.get("layer_range")
        self.sampler = EpisodeSampler(
            docs, self.num_layers, cfg.get("seed", 0),
            layer_range=tuple(layer_range) if layer_range else None)
        bank = LensBank(Path(cfg["lens_dir"]), self.num_layers, self.device)
        self.lens = LensRewardModel(parts, bank, tokenizer)

        self.policy = make_policy(parts, cfg)
        parts.text_model.config.use_cache = False
        self.optimizer = torch.optim.Adam(
            [p for p in self.policy.parameters() if p.requires_grad],
            lr=cfg.get("lr", 5e-5),
            betas=tuple(cfg.get("adam_betas", (0.8, 0.95))))
        self.metrics_fh = (run_dir / "metrics.jsonl").open("a")

    def render(self, ep) -> str:
        prompt = render_prompt(ep, self.tokenizer, self.num_layers)
        chat = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True, enable_thinking=False)
        return chat + PREFILL.format(pos=ep.pos, layer=ep.layer)

    def step(self) -> dict:
        cfg = self.cfg
        episodes = self.sampler.sample(cfg.get("batch_size", 32))

        # on-policy lens targets (no grad; current adapter active)
        self.policy.eval()
        targets = self.lens.lens_probs(episodes)  # [N, vocab] float32

        seqs = [self.tokenizer(self.render(ep), add_special_tokens=False)["input_ids"]
                for ep in episodes]
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        self.policy.train()
        self.optimizer.zero_grad(set_to_none=True)
        micro = cfg.get("micro_batch_size", cfg.get("batch_size", 32))
        loss_total = 0.0
        prob_top1_sum = 0.0
        top1_hits = 0
        for begin in range(0, len(seqs), micro):
            chunk = seqs[begin:begin + micro]
            tgt = targets[begin:begin + micro]
            width = max(len(s) for s in chunk)
            ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
            mask = torch.zeros((len(chunk), width), dtype=torch.long)
            last = torch.tensor([len(s) - 1 for s in chunk])
            for i, s in enumerate(chunk):
                ids[i, :len(s)] = torch.tensor(s)
                mask[i, :len(s)] = 1
            ids, mask, last = (t.to(self.device) for t in (ids, mask, last))

            hidden = self.parts.text_model(
                input_ids=ids, attention_mask=mask).last_hidden_state
            h = hidden[torch.arange(len(chunk), device=self.device), last]
            logits = self.parts.unembed(h).float()          # [m, vocab]
            logp = torch.log_softmax(logits, dim=-1)
            # soft cross-entropy against the lens distribution, mean over the
            # full batch (folded in so micro-losses accumulate by summation)
            loss = -(tgt * logp).sum() / len(seqs)
            loss.backward()
            loss_total += loss.item()

            with torch.no_grad():
                pred = logp.exp()
                top = tgt.argmax(dim=-1)
                prob_top1_sum += pred.gather(-1, top[:, None]).sum().item()
                top1_hits += (pred.argmax(dim=-1) == top).sum().item()
            del hidden, h, logits, logp

        grad_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in self.policy.parameters() if p.requires_grad],
            cfg.get("max_grad_norm", 1.0))
        self.optimizer.step()
        self.policy.eval()

        n = len(seqs)
        return {
            "loss": loss_total,
            "grad_norm": float(grad_norm),
            "prob_top1": prob_top1_sum / n,
            "top1_acc": top1_hits / n,
            "target_top_prob": targets.max(dim=-1).values.mean().item(),
        }

    def print_sample(self, step: int):
        ep = self.sampler.sample(1)[0]
        self.policy.eval()
        with torch.no_grad():
            tgt = self.lens.lens_probs([ep])[0]
            ids = torch.tensor(
                [self.tokenizer(self.render(ep), add_special_tokens=False)["input_ids"]],
                device=self.device)
            h = self.parts.text_model(input_ids=ids).last_hidden_state[0, -1]
            pred = torch.softmax(self.parts.unembed(h).float(), dim=-1)
        def top3(p):
            v, i = p.topk(3)
            return ", ".join(f"{self.tokenizer.decode([t])!r}:{x:.3f}"
                             for t, x in zip(i.tolist(), v.tolist()))
        print(f"--- sample (step {step}) doc={ep.doc_id} pos={ep.pos} "
              f"layer={ep.layer}\n    lens target: {top3(tgt)}\n"
              f"    model pred:  {top3(pred)}", flush=True)

    def train(self, num_steps: int):
        from .plots import write_plots, DIRECT_PLOTS
        for step in range(1, num_steps + 1):
            t0 = time.perf_counter()
            metrics = self.step()
            row = {"step": step, "step_sec": time.perf_counter() - t0, **metrics}
            self.metrics_fh.write(json.dumps(row) + "\n")
            self.metrics_fh.flush()
            if step % self.cfg.get("log_every", 1) == 0 or step == 1:
                print(f"step {step}/{num_steps} loss={metrics['loss']:.4f} "
                      f"prob_top1={metrics['prob_top1']:.4f} "
                      f"top1={metrics['top1_acc']:.3f} "
                      f"target_top_prob={metrics['target_top_prob']:.3f} "
                      f"({row['step_sec']:.1f}s)", flush=True)
            if step % self.cfg.get("sample_every", 25) == 0:
                self.print_sample(step)
            if step % self.cfg.get("checkpoint_every", 200) == 0:
                self.policy.save_pretrained(
                    str(self.run_dir / "checkpoints" / f"adapter_step_{step}"))
            if step % self.cfg.get("plot_every", 25) == 0 or step == num_steps:
                try:
                    write_plots(self.run_dir, DIRECT_PLOTS)
                except Exception as e:
                    print(f"plotting failed: {e}", flush=True)
        self.policy.save_pretrained(str(self.run_dir / "adapter"))

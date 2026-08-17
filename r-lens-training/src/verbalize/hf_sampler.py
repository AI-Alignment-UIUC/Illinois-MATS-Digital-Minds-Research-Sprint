"""HF-generate rollout backend for the RL trainer.

Drop-in replacement for SGLangServer.generate for models sglang cannot serve
with LoRA (e.g. gemma-4-E2B's non-uniform layer shapes). Rollouts are sampled
with the policy itself on the training GPU, so there is no adapter syncing at
all — sync_lora is a no-op and rollouts are trivially on-policy.
"""
import torch


class HFSampler:
    proc = None          # duck-type SGLangServer for the launcher's cleanup
    lora_name = None

    def __init__(self, parts, tokenizer, device: str, batch_size: int = 64):
        from chat.server import turn_end_token_ids
        self.parts = parts
        self.tokenizer = tokenizer
        self.device = device
        self.batch_size = batch_size
        self.eos_ids = turn_end_token_ids(tokenizer, parts.model.config)

    def sync_lora(self, adapter_dir):
        pass  # rollouts share the policy's weights directly

    def flush_cache(self):
        pass

    def shutdown(self):
        pass

    @torch.no_grad()
    def generate(self, prompts, *, group_size, max_new_tokens,
                 temperature=1.0, stop=None):
        expanded = [p for p in prompts for _ in range(group_size)]
        tok = self.tokenizer
        texts, ids_out, truncated = [], [], []
        eos = set(self.eos_ids)
        old_side = tok.padding_side
        tok.padding_side = "left"  # required for batched decoder-only generate
        try:
            for begin in range(0, len(expanded), self.batch_size):
                chunk = expanded[begin:begin + self.batch_size]
                enc = tok(chunk, return_tensors="pt", padding=True,
                          add_special_tokens=False).to(self.device)
                out = self.parts.model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-4),
                    top_p=1.0, top_k=0,
                    use_cache=True,
                    eos_token_id=self.eos_ids,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id)
                gen = out[:, enc["input_ids"].shape[1]:]
                for row in gen.tolist():
                    # cut at the first eos (inclusive, so the policy learns
                    # to stop); no eos found -> truncated at the cap
                    cut = next((i for i, t in enumerate(row) if t in eos), None)
                    if cut is None:
                        seq = row
                        truncated.append(True)
                    else:
                        seq = row[:cut + 1]
                        truncated.append(False)
                    ids_out.append(seq)
                    texts.append(tok.decode(seq, skip_special_tokens=False))
        finally:
            tok.padding_side = old_side
        return texts, ids_out, truncated

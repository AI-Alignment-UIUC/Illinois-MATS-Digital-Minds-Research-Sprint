"""Model loading and structural traversal for lens training.

Works with plain causal LMs as well as multimodal wrappers such as
Qwen3_5ForConditionalGeneration, where the decoder lives at
model.model.language_model.
"""
from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelParts:
    model: nn.Module          # the full loaded model
    text_model: nn.Module     # decoder stack owner (has .layers and .norm)
    layers: nn.ModuleList     # decoder layers
    final_norm: nn.Module     # final residual-stream RMSNorm
    lm_head: nn.Module        # unembedding
    hidden_size: int
    logit_softcap: float | None = None  # e.g. gemma's final_logit_softcapping

    def unembed(self, normed_hidden):
        """lm_head plus the model's final logit softcapping, if any."""
        logits = self.lm_head(normed_hidden)
        if self.logit_softcap:
            logits = self.logit_softcap * torch.tanh(logits / self.logit_softcap)
        return logits


def _load(path: str, dtype: torch.dtype, device: str) -> nn.Module:
    from transformers import AutoModelForCausalLM

    last_err = None
    try:
        return AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(device)
    except Exception as e:  # multimodal configs are not accepted by AutoModelForCausalLM
        last_err = e
    try:
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(path, dtype=dtype).to(device)
    except Exception:
        raise last_err


def load_model(path: str, dtype: str = "bfloat16", device: str = "cuda") -> ModelParts:
    torch_dtype = getattr(torch, dtype)
    model = _load(path, torch_dtype, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    core = getattr(model, "model", model)
    text = getattr(core, "language_model", core)
    if not hasattr(text, "layers"):
        raise ValueError(f"Could not locate decoder layers on {type(model).__name__}")

    lm_head = model.get_output_embeddings()
    return ModelParts(
        model=model,
        text_model=text,
        layers=text.layers,
        final_norm=text.norm,
        lm_head=lm_head,
        hidden_size=text.config.hidden_size,
        logit_softcap=getattr(text.config, "final_logit_softcapping", None),
    )

"""FastAPI chat server with r-lens token inspection.

Single-user server. Generation streams over plain-text chunks; after each
turn the client calls /api/tokens, which runs one forward pass over the full
rendered conversation, caches every layer's residual stream, and returns the
token strings. Clicking a token calls /api/lens with a layer range [lo, hi]
(1-indexed, lo <= hi); the server averages the per-layer lens-mapped vectors
W_l @ h_l(t) over that range and returns the top-10 readout tokens.
"""
import threading
from pathlib import Path

import torch
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

UI_PATH = Path(__file__).parent / "ui.html"


class ChatRequest(BaseModel):
    messages: list[dict]
    thinking: bool = False
    max_new_tokens: int = 2048
    temperature: float = 0.7


class LensRequest(BaseModel):
    index: int
    lo: int
    hi: int


def turn_end_token_ids(tokenizer, model_config) -> list[int]:
    """Every id that should end a generation turn: the chat template's
    end-of-turn marker (derived from the template itself, so it works across
    model families) plus the tokenizer/model eos ids."""
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": "x"}, {"role": "assistant", "content": "Y"}],
        tokenize=False)
    tail = text[text.rfind("Y") + 1:]
    ids = tokenizer(tail, add_special_tokens=False)["input_ids"]
    out = []
    for candidate in ids[:1] + [tokenizer.eos_token_id]:
        if candidate is not None and candidate not in out:
            out.append(candidate)
    cfg_eos = getattr(model_config, "eos_token_id", None)
    for e in (cfg_eos if isinstance(cfg_eos, list) else [cfg_eos]):
        if e is not None and e not in out:
            out.append(e)
    return out


class ChatState:
    def __init__(self, parts, tokenizer, lens_mats, meta: dict, device: str):
        self.parts = parts
        self.model = parts.model
        self.tokenizer = tokenizer
        self.lens_mats = lens_mats  # [L, d, d] float32 on device, or None
        self.meta = meta
        self.device = device
        self.num_layers = len(parts.layers)
        self.cached_ids: list[int] = []
        self.cached_residuals: torch.Tensor | None = None  # [L, T, d] on CPU
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.eos_ids = turn_end_token_ids(tokenizer, parts.model.config)

    def render(self, messages, thinking: bool, generation_prompt: bool) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False,
            add_generation_prompt=generation_prompt,
            enable_thinking=thinking)

    def stream_chat(self, req: ChatRequest):
        from transformers import (StoppingCriteria, StoppingCriteriaList,
                                  TextIteratorStreamer)

        stop_event = self.stop_event
        stop_event.clear()

        class _EventStop(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return stop_event.is_set()

        text = self.render(req.messages, req.thinking, True)
        ids = self.tokenizer(text, return_tensors="pt",
                             add_special_tokens=False).to(self.device)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        kwargs = dict(**ids, streamer=streamer,
                      max_new_tokens=req.max_new_tokens,
                      do_sample=req.temperature > 0,
                      temperature=max(req.temperature, 1e-4),
                      stopping_criteria=StoppingCriteriaList([_EventStop()]),
                      eos_token_id=self.eos_ids,
                      pad_token_id=self.tokenizer.eos_token_id)
        thread = threading.Thread(
            target=lambda: self.model.generate(**kwargs), daemon=True)
        with self.lock:
            thread.start()
            try:
                for chunk in streamer:
                    yield chunk
            finally:
                # client disconnects must not leave the generate thread
                # decoding to the cap in the background
                stop_event.set()
                thread.join()

    @torch.no_grad()
    def cache_tokens(self, messages, thinking: bool) -> list[str]:
        text = self.render(messages, thinking, False)
        ids = self.tokenizer(text, add_special_tokens=False)["input_ids"]
        input_ids = torch.tensor([ids], device=self.device)
        residuals = []
        handles = [
            layer.register_forward_hook(
                lambda _m, _i, out: residuals.append(
                    (out[0] if isinstance(out, tuple) else out)[0].to("cpu")))
            for layer in self.parts.layers
        ]
        try:
            with self.lock:
                self.parts.text_model(input_ids=input_ids)
        finally:
            for h in handles:
                h.remove()
        self.cached_ids = ids
        self.cached_residuals = torch.stack(residuals)  # [L, T, d]
        return [self.tokenizer.decode([t]) for t in ids]

    @torch.no_grad()
    def lens_topk(self, index: int, lo: int, hi: int, k: int = 10) -> list[dict]:
        if self.cached_residuals is None:
            raise ValueError("no cached forward pass; call /api/tokens first")
        if self.lens_mats is None:
            raise ValueError("no lenses available for this model")
        lo = max(1, min(lo, self.num_layers))
        hi = max(lo, min(hi, self.num_layers))
        if not (0 <= index < len(self.cached_ids)):
            raise ValueError(f"token index {index} out of range")
        h = self.cached_residuals[lo - 1:hi, index].to(self.device)  # [n, d]
        w = self.lens_mats[lo - 1:hi]                                # [n, d, d]
        mapped = torch.bmm(w, h.float().unsqueeze(-1)).squeeze(-1)   # [n, d]
        acc = mapped.mean(dim=0)                                     # accumulated
        normed = self.parts.final_norm(acc.to(h.dtype).unsqueeze(0))
        logits = self.parts.unembed(normed).float()[0]
        probs = torch.softmax(logits, dim=-1)
        top = probs.topk(k)
        return [{"token": self.tokenizer.decode([i]), "prob": round(p, 5)}
                for i, p in zip(top.indices.tolist(), top.values.tolist())]


def build_app(state: ChatState) -> FastAPI:
    app = FastAPI()

    @app.get("/")
    def index():
        return HTMLResponse(UI_PATH.read_text())

    @app.get("/api/meta")
    def meta():
        return {**state.meta, "num_layers": state.num_layers,
                "has_lenses": state.lens_mats is not None}

    @app.post("/api/chat")
    def chat(req: ChatRequest):
        return StreamingResponse(state.stream_chat(req),
                                 media_type="text/plain")

    @app.post("/api/tokens")
    def tokens(req: ChatRequest):
        return {"tokens": state.cache_tokens(req.messages, req.thinking)}

    @app.post("/api/stop")
    def stop():
        state.stop_event.set()
        return {"ok": True}

    @app.post("/api/lens")
    def lens(req: LensRequest):
        try:
            return {"top": state.lens_topk(req.index, req.lo, req.hi)}
        except ValueError as e:
            return {"error": str(e)}

    return app

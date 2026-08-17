"""Public API server mode for the chat server.

Serves one base model plus every trained LoRA adapter found under
artifacts/runs/ to remote clients (e.g. hackathon teammates), with:

- Bearer-token auth on every model endpoint (keys printed at startup or
  provided via --api-key); constant-time comparison; no endpoint touches the
  filesystem, shell, or anything beyond the loaded model.
- A batching worker: concurrent requests with the same (adapter, sampling
  params) are generated together in one batched forward, so parallel clients
  share the GPU efficiently. Everything else queues FIFO.
- Per-request adapter selection from a fixed, server-side allowlist (clients
  send names, never paths).
- /v1/reboot reloads all adapter weights from disk (picks up newly trained
  checkpoints) without restarting the process.
- GET / serves plain, LLM-readable API documentation.
"""
import hmac
import json
import queue
import threading
import time
from pathlib import Path

import torch
from fastapi import Depends, FastAPI, HTTPException, Header
from fastapi.responses import HTMLResponse, PlainTextResponse
from pydantic import BaseModel, Field

from .server import turn_end_token_ids

DOCS_PATH = Path(__file__).parent / "api_docs.md"

MAX_NEW_TOKENS = 2048
MAX_MESSAGES = 50
MAX_TOTAL_CHARS = 30_000
MAX_LENS_TEXT_CHARS = 4_000
MAX_TOP_K = 50
BATCH_WINDOW_S = 0.05
MAX_BATCH = 8


class ChatBody(BaseModel):
    messages: list[dict] = Field(..., max_length=MAX_MESSAGES)
    adapter: str | None = None
    thinking: bool = False
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_new_tokens: int = Field(512, ge=1, le=MAX_NEW_TOKENS)


class LensBody(BaseModel):
    text: str = Field(..., max_length=MAX_LENS_TEXT_CHARS)
    position: int = Field(..., ge=0)
    layer_lo: int = Field(1, ge=1)
    layer_hi: int = Field(9999, ge=1)
    top_k: int = Field(10, ge=1, le=MAX_TOP_K)
    adapter: str | None = None


class AdapterBody(BaseModel):
    adapter: str


class _Job:
    def __init__(self, prompt: str, adapter: str, temperature: float,
                 max_new_tokens: int):
        self.prompt = prompt
        self.adapter = adapter
        self.key = (adapter, round(temperature, 4), max_new_tokens)
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.done = threading.Event()
        self.result: dict | None = None
        self.error: str | None = None


class ApiState:
    def __init__(self, parts, tokenizer, lens_mats, model_name: str,
                 device: str, adapter_registry: dict[str, Path],
                 api_keys: list[str]):
        self.parts = parts
        self.tokenizer = tokenizer
        self.lens_mats = lens_mats
        self.model_name = model_name
        self.device = device
        self.adapter_registry = dict(adapter_registry)  # name -> dir (server-side only)
        self.api_keys = list(api_keys)
        self.default_adapter = "base"
        self.num_layers = len(parts.layers)
        self.eos_ids = turn_end_token_ids(tokenizer, parts.model.config)
        self.gpu_lock = threading.Lock()
        self.jobs: queue.Queue = queue.Queue()
        self.started = time.time()
        self.requests_served = 0

        self.peft_model = None
        if self.adapter_registry:
            from peft import PeftModel
            names = list(self.adapter_registry)
            self.peft_model = PeftModel.from_pretrained(
                parts.model, str(self.adapter_registry[names[0]]),
                adapter_name=names[0])
            for name in names[1:]:
                self.peft_model.load_adapter(
                    str(self.adapter_registry[name]), adapter_name=name)
            self.peft_model.eval()
        self.worker = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker.start()

    # ---- adapter handling --------------------------------------------------

    def _activate(self, adapter: str):
        """Context for running with a named adapter ('base' = no adapter)."""
        import contextlib
        if adapter == "base" or self.peft_model is None:
            return (self.peft_model.disable_adapter()
                    if self.peft_model is not None else contextlib.nullcontext())
        self.peft_model.set_adapter(adapter)
        return contextlib.nullcontext()

    def resolve_adapter(self, name: str | None) -> str:
        adapter = name or self.default_adapter
        if adapter != "base" and adapter not in self.adapter_registry:
            raise HTTPException(400, f"unknown adapter {adapter!r}; "
                                     f"see GET /v1/models for valid names")
        return adapter

    def reload_adapters(self) -> list[str]:
        """Reload every registered adapter's weights from disk."""
        if self.peft_model is None:
            return []
        with self.gpu_lock:
            for name, path in self.adapter_registry.items():
                self.peft_model.delete_adapter(name)
                self.peft_model.load_adapter(str(path), adapter_name=name)
            self.peft_model.eval()
            torch.cuda.empty_cache()
        return list(self.adapter_registry)

    # ---- generation worker -------------------------------------------------

    def submit(self, job: _Job) -> _Job:
        self.jobs.put(job)
        job.done.wait(timeout=600)
        if not job.done.is_set():
            raise HTTPException(504, "generation timed out")
        if job.error:
            raise HTTPException(500, job.error)
        return job

    def _worker_loop(self):
        while True:
            first = self.jobs.get()
            batch = [first]
            deadline = time.time() + BATCH_WINDOW_S
            while len(batch) < MAX_BATCH:
                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                try:
                    batch.append(self.jobs.get(timeout=remaining))
                except queue.Empty:
                    break
            groups: dict[tuple, list[_Job]] = {}
            for job in batch:
                groups.setdefault(job.key, []).append(job)
            for jobs in groups.values():
                try:
                    self._run_group(jobs)
                except Exception as e:  # noqa: BLE001 - reported to clients
                    for job in jobs:
                        job.error = f"generation failed: {type(e).__name__}"
                        job.done.set()

    @torch.no_grad()
    def _run_group(self, jobs: list[_Job]):
        tok = self.tokenizer
        adapter = jobs[0].adapter
        old_side = tok.padding_side
        tok.padding_side = "left"
        try:
            with self.gpu_lock, self._activate(adapter):
                enc = tok([j.prompt for j in jobs], return_tensors="pt",
                          padding=True, add_special_tokens=False).to(self.device)
                out = self.parts.model.generate(
                    **enc,
                    max_new_tokens=jobs[0].max_new_tokens,
                    do_sample=jobs[0].temperature > 0,
                    temperature=max(jobs[0].temperature, 1e-4),
                    use_cache=True,
                    eos_token_id=self.eos_ids,
                    pad_token_id=tok.pad_token_id or tok.eos_token_id)
                gen = out[:, enc["input_ids"].shape[1]:]
                for job, row in zip(jobs, gen.tolist()):
                    eos = set(self.eos_ids)
                    cut = next((i for i, t in enumerate(row) if t in eos), len(row))
                    job.result = {
                        "completion": tok.decode(row[:cut], skip_special_tokens=True),
                        "completion_tokens": cut,
                        "prompt_tokens": int(enc["attention_mask"][0].sum()),
                        "adapter": adapter,
                    }
                    job.done.set()
        finally:
            tok.padding_side = old_side

    # ---- lens --------------------------------------------------------------

    @torch.no_grad()
    def lens(self, body: LensBody, adapter: str) -> dict:
        if self.lens_mats is None:
            raise HTTPException(400, "no r-lens matrices available for this model")
        tok = self.tokenizer
        ids = tok(body.text, add_special_tokens=False)["input_ids"]
        if tok.bos_token_id is not None:
            ids = [tok.bos_token_id] + ids
        if not ids or body.position >= len(ids):
            raise HTTPException(400, f"position {body.position} out of range "
                                     f"(text has {len(ids)} tokens)")
        lo = max(1, min(body.layer_lo, self.num_layers))
        hi = max(lo, min(body.layer_hi, self.num_layers))
        residuals = []
        handles = [layer.register_forward_hook(
            lambda _m, _i, out: residuals.append(
                (out[0] if isinstance(out, tuple) else out)[0]))
            for layer in self.parts.layers]
        try:
            with self.gpu_lock, self._activate(adapter):
                input_ids = torch.tensor([ids], device=self.device)
                self.parts.text_model(input_ids=input_ids)
        finally:
            for h in handles:
                h.remove()
        stacked = torch.stack(residuals)                    # [L, T, d]
        h = stacked[lo - 1:hi, body.position]               # [n, d]
        mapped = torch.bmm(self.lens_mats[lo - 1:hi],
                           h.float().unsqueeze(-1)).squeeze(-1)
        acc = mapped.mean(dim=0)
        normed = self.parts.final_norm(acc.to(h.dtype).unsqueeze(0))
        probs = torch.softmax(self.parts.unembed(normed).float()[0], dim=-1)
        top = probs.topk(body.top_k)
        return {
            "tokens": [tok.decode([t]) for t in ids],
            "position": body.position,
            "layers": [lo, hi],
            "adapter": adapter,
            "top": [{"token": tok.decode([i]), "prob": round(p, 6)}
                    for i, p in zip(top.indices.tolist(), top.values.tolist())],
        }


def build_api_app(state: ApiState) -> FastAPI:
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    def require_key(authorization: str | None = Header(None)):
        token = (authorization or "").removeprefix("Bearer").strip()
        if not any(hmac.compare_digest(token, k) for k in state.api_keys):
            raise HTTPException(401, "invalid or missing API key; send "
                                     "'Authorization: Bearer <key>'")

    @app.get("/")
    def docs():
        md = DOCS_PATH.read_text()
        md = (md.replace("{MODEL_NAME}", state.model_name)
                .replace("{NUM_LAYERS}", str(state.num_layers))
                .replace("{ADAPTERS}", json.dumps(["base"] + list(state.adapter_registry))))
        return HTMLResponse(f"<html><body><pre style='white-space:pre-wrap;"
                            f"font-family:ui-monospace,monospace;max-width:900px;"
                            f"margin:2em auto'>{md}</pre></body></html>")

    @app.get("/v1/docs.md")
    def docs_md():
        md = DOCS_PATH.read_text()
        md = (md.replace("{MODEL_NAME}", state.model_name)
                .replace("{NUM_LAYERS}", str(state.num_layers))
                .replace("{ADAPTERS}", json.dumps(["base"] + list(state.adapter_registry))))
        return PlainTextResponse(md, media_type="text/markdown")

    @app.get("/v1/health")
    def health():
        return {"status": "ok", "uptime_s": round(time.time() - state.started),
                "requests_served": state.requests_served}

    @app.get("/v1/models", dependencies=[Depends(require_key)])
    def models():
        return {"model": state.model_name, "num_layers": state.num_layers,
                "adapters": ["base"] + list(state.adapter_registry),
                "default_adapter": state.default_adapter,
                "has_lens": state.lens_mats is not None}

    @app.post("/v1/chat", dependencies=[Depends(require_key)])
    def chat(body: ChatBody):
        total = sum(len(str(m.get("content", ""))) for m in body.messages)
        if total > MAX_TOTAL_CHARS:
            raise HTTPException(400, f"conversation too long (> {MAX_TOTAL_CHARS} chars)")
        for m in body.messages:
            if m.get("role") not in ("user", "assistant", "system"):
                raise HTTPException(400, "roles must be user/assistant/system")
        adapter = state.resolve_adapter(body.adapter)
        prompt = state.tokenizer.apply_chat_template(
            body.messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=body.thinking)
        job = state.submit(_Job(prompt, adapter, body.temperature,
                                body.max_new_tokens))
        state.requests_served += 1
        return job.result

    @app.post("/v1/lens", dependencies=[Depends(require_key)])
    def lens(body: LensBody):
        adapter = state.resolve_adapter(body.adapter)
        state.requests_served += 1
        return state.lens(body, adapter)

    @app.post("/v1/adapter/switch", dependencies=[Depends(require_key)])
    def switch(body: AdapterBody):
        state.default_adapter = state.resolve_adapter(body.adapter)
        return {"default_adapter": state.default_adapter}

    @app.post("/v1/reboot", dependencies=[Depends(require_key)])
    def reboot():
        names = state.reload_adapters()
        return {"status": "reloaded", "adapters": names}

    return app

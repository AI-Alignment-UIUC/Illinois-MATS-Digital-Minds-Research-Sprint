# IntrospectionTraining

Training LLMs (Qwen3.5-4B, gemma-4-E2B-it) to verbalize their own r-lens
readouts — an introspection experiment. The r-lens maps a mid-layer
residual-stream activation through a fitted linear map plus the model's own
unembedding, giving a per-(token, layer) distribution over vocabulary tokens
("what the computation there is promoting"); we then train the model, via SFT
warmstart / REINFORCE / direct supervision, to report that readout.

## Layout

    src/          library code (r_lens/, verbalize/, chat/)
    scripts/      entry points (below)
    configs/      JSON configs; RL/SFT/direct configs are nested
                  {top-level paths, "optim": {...}, "run": {..., "logging"}}
    data/         models/ (HF weights), documents/, prompts, SFT trace datasets
    artifacts/    jlenses/<model>/ (lens matrices, destructively overwritten)
                  runs/<name>/     (one dir per training run: config.json,
                                    metrics.jsonl, plots/, adapter/, checkpoints/)

Environments: `.venv` (torch/transformers/peft — all training + serving),
`.venv-sglang` (SGLang 0.5.17, patched; used only for Qwen RL rollouts).
Both are excluded from the archive; recreate with `uv venv` + `uv pip install
torch transformers datasets peft fastapi uvicorn matplotlib safetensors
accelerate flash-linear-attention` (and `sglang[all]` in `.venv-sglang`).

## Scripts

All run as `.venv/bin/python scripts/<script>.py`.

**download_documents.py** — stream ~1000 long documents (FineWeb-Edu) to
`data/documents/documents.jsonl`; used for lens fitting.

**download_short_documents.py** — 1000 short documents (20–60 tokens) to
`data/documents/short_documents.jsonl`; the episode pool for all
verbalization training.

**train_r_lens.py --config configs/train_r_lens_<model>.json** — fit an
r-lens per layer: the expected same-position Jacobian of the final residual
w.r.t. each layer's residual, with LRP-modified backwards (LN-rule,
Identity-rule, Half-rule; value-exact straight-through implementations),
estimated by random-probe VJPs over the documents. DESTRUCTIVELY overwrites
`artifacts/jlenses/<model>/` (layer_XX.safetensors + metadata.json).
BOS is prepended automatically for models that need it (gemma).

**train_verbalizer.py --config <cfg> [--run-name N]** — RL (REINFORCE with
per-group rank-based rewards in [-1,1], group baselines, no clipping, one
Adam step per rollout, rank-8 rsLoRA on MLP projections). Reward = on-policy
r-lens probability of the verbalized <answer> token, normalized by the lens
top-1, with format/truncation penalties and optional min-think shaping.
Rollout backends: SGLang server (`run.inference_gpu` + `run.sglang_port`;
Qwen) or in-process HF generate (`"backend": "hf"`; required for gemma,
whose non-uniform layer shapes SGLang LoRA cannot serve). Optional
`init_adapter` warm-starts from an SFT adapter (snapshotted read-only into
the run dir).

**sft_verbalizer.py --config configs/sft_verbalizer*.json** — SFT warmstart
on synthetic introspective thinking traces
(`data/rlens_thinking_trajs_warmstart*.jsonl`; answers are the true lens
top-1 tokens). Each trace yields two samples (with/without the r-lens
explanation in the prompt). Breaks the models' reflexive "I can't access my
activations" refusals. Completion format is derived from each model's chat
template (Qwen <think>, gemma thought-channel).

**train_verbalizer_direct.py --config configs/train_direct_verbaliser.json**
— supervised alternative to RL: prefill "Hmm, It feels like at pos {pos} and
layer {layer} the activations have the shape of:" and train the last
position with soft cross-entropy against the full on-policy lens
distribution. Headline metric: prob_top1.

**plot_verbalizer.py artifacts/runs/<run>** — regenerate a run's plots
(trainers also write them automatically every `plot_every` steps).

**chat_server.py [--port P] [--api-key K ...]** — interactive menu, two modes:
1. *Local chat UI*: pick model, optionally equip one adapter, pick GPU;
   web chat with thinking toggle, layer-range selector, click-a-token
   accumulated r-lens readout, Stop button.
2. *Public API server*: serves the base model + ALL matching adapters;
   bearer-token auth, batched parallel generation, endpoints /v1/chat,
   /v1/lens, /v1/adapter/switch, /v1/reboot, /v1/models, /v1/health; GET /
   is LLM-readable API documentation. Expose externally with
   `cloudflared tunnel --url http://localhost:<port>`.

## Notes

- Lens matrices are model-specific and live in `artifacts/jlenses/<model>/`;
  readouts everywhere apply the model's final norm, unembedding, and (for
  gemma) final logit softcapping.
- SGLang is patched in-place (`.venv-sglang`): multimodal text-config
  traversal, rsLoRA scaling, skip-unsupported-LoRA-modules. Re-apply if the
  venv is rebuilt.
- Training metrics: `artifacts/runs/<run>/metrics.jsonl` (raw shaped reward
  is logged even though optimization uses rank rewards).

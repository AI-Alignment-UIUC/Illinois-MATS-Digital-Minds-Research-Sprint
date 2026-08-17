# Illinois MATS Digital Minds Research Sprint

RL to make models more introspective, and evals to measure introspection.

Two halves. `r-lens-training/` trains a Qwen3.5-4B model with reinforcement learning to verbalize
its own R-lens readouts, a linear decoding of its residual stream. `introspection-eval/` measures
whether a model's self-report actually tracks its own behavior, and was run on six checkpoints of
that training run plus five off-the-shelf models.

The writeup is [`REPORT.md`](REPORT.md).

## What we found

The evaluation scores each self-report as two numbers rather than one: how well a model tells the
prompts it answers consistently from those where it varies (sensitivity, d′), and how readily it
claims consistency at all (criterion, c). Over the training run, criterion moved a long way and
sensitivity did not move measurably.

| | base | RL latest |
|---|---|---|
| wrong claims of consistency on varying prompts | 8/30 | 23/26 |
| criterion c | −0.04 [−0.57, 0.45] | −1.45 [−1.94, −1.18] |
| sensitivity d′ | 1.27 [0.39, 2.39] | 0.64 [−0.27, 1.18] |
| stated-minus-actual gap (pp) | −1.8 | −1.6 |

Two things worth pulling out. The gap statistic that prior work usually reports read the same
before and after, so a single-number evaluation would have found nothing. And the d′ intervals all
overlap, so the sensitivity result is a null: we found no evidence detection improved, not evidence
that it got worse.

Separately, an R-lens readout kept tracking true consistency (+0.42 → +0.48) while the verbal
report's tracking fell (+0.38 → +0.10), a paired difference of +0.38 [+0.06, +0.71]. The
information stayed available internally and what changed was the report. Those four figures are
from `rl-final`, the earlier of two same-day captures of the training head, because that is the
capture the lens sweep was run on; the table above uses the later capture, `rl-latest`.

## Layout

```
introspection-eval/     the evaluation
  METHOD.md               prompts and procedure
  RESULTS.md              every table, all 11 model variants
  FINDINGS.md             narrative writeup
  run_bench.py            collection (resumable; --mock needs no credentials)
  analyze.py              scores and tables
  figures.py              plots
  lens_readout.py         R-lens sweep over layer bands
  serving_check.py        provider determinism check
  items.json              the 48 prompts
  runs/*.jsonl            raw per-call data, append-only
  figures/                fig1-fig7

r-lens-training/        the training method
  documentation.md        full pipeline description
  scripts/                entry points (below)
  src/r_lens/             lens fitting, LRP rules
  src/verbalize/          RL trainer, reward, rollout backends
  src/chat/               chat UI and the /v1 API server
  configs/                JSON configs per model and stage
  artifacts/runs/         per-run config.json, metrics.jsonl, plots/

REPORT.md               the writeup
ideation/               project notes
```

## Running the evaluation

```bash
cd introspection-eval
python run_bench.py --mock --out smoke.jsonl    # no credentials needed, checks the wiring
```

For a real run, the bench models go through OpenRouter and the trained checkpoints through the
r-lens API server:

```bash
export OPENROUTER_API_KEY=...
export RLENS_BASE_URL=http://HOST:PORT
export RLENS_API_KEY=...

python run_bench.py                # resumable; re-running skips completed cells
python analyze.py                  # writes RESULTS.md and results.json
python figures.py                  # writes figures/
```

Useful flags: `--models`, `--channels actor,pred,cross,detect,afc`, `--k` (ground-truth samples per
prompt, default 16), `--out` (filename inside `runs/`).

The two validity instruments run separately:

```bash
python serving_check.py            # is the provider actually sampling?
python lens_readout.py --sweep     # R-lens readout across layer bands
```

Ground truth is 16 fresh samples per prompt at temperature 1.0; the meta channels run at
temperature 0. A full run is roughly 14,400 calls across 11 model variants.

## Running the training

Two environments, both excluded from the repo. Recreate with `uv venv` plus:

```bash
uv pip install torch transformers datasets peft fastapi uvicorn matplotlib \
               safetensors accelerate flash-linear-attention
# and sglang[all] in a separate .venv-sglang, used only for Qwen RL rollouts
```

Then, in order:

```bash
python scripts/download_short_documents.py                              # episode pool
python scripts/train_r_lens.py    --config configs/train_r_lens_qwen3_5_4b.json
python scripts/sft_verbalizer.py  --config configs/sft_verbalizer.json  # removes the refusal
python scripts/train_verbalizer.py --config configs/train_verbalizer.json
python scripts/chat_server.py                                           # chat UI or /v1 API
```

`train_r_lens.py` **destructively overwrites** `artifacts/jlenses/<model>/`. SGLang in
`.venv-sglang` is patched in place; the patches must be reapplied if that venv is rebuilt. See
[`r-lens-training/documentation.md`](r-lens-training/documentation.md) for the full description,
including the gemma variant, which uses the in-process HF rollout backend because SGLang's LoRA
cannot serve its non-uniform layer shapes.

## Data and licensing

The lens-fitting corpus is a sample of [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu),
ODC-By 1.0, so attribution is required for derived work. The large document dump is gitignored and
regenerable with `scripts/download_documents.py`; the smaller episode pool used by every
verbalization run is committed. Model weights, fitted lens matrices, and trained adapters are
gitignored.

## Authors

William Wale (MATS Research), Heather Broome (UIUC), Ricky Mouser (MATS Research), Evan Coats
(UIUC). With Apart Research.

William designed the R-Space RL training technique. Heather and Ricky designed the introspection
evaluation. Evan synthesized the research into a reproducible benchmark and wrote the final
manuscript.

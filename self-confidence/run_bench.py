#!/usr/bin/env python3
"""Runner for the confidence-in-introspection arm (see METHOD.md).

Channels per model (all at hint level L0, no hints):
  actor    - bare item, k samples at T=1.0 -> ground-truth answer distribution
  pred     - parent predictor + MODE_CONF line (channel 1, self)
  cross    - same predictor aimed at each OTHER model (self-vs-other control)
  detect   - yes/no determinism self-report + confidence (channel 2, SDT)
  afc      - self-foil 2AFC + confidence (channel 3), needs actor results first

Usage:
  OPENROUTER_API_KEY=... python3 run_bench.py            # full run, resumable
  python3 run_bench.py --mock                            # offline pipeline test
  python3 run_bench.py --models haiku-4.5 --channels actor,pred
Results append to runs/results.jsonl keyed by call id; reruns skip completed calls.
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).parent
RUNS = HERE / "runs"

MODELS = {
    # the five IntrospectBench models, via OpenRouter
    "qwen3-8b": {"backend": "openrouter", "id": "qwen/qwen3-8b"},
    "qwen3.5-9b": {"backend": "openrouter", "id": "qwen/qwen3.5-9b"},
    "gemma-3-4b": {"backend": "openrouter", "id": "google/gemma-3-4b-it"},
    "gemma-3n-e4b": {"backend": "openrouter", "id": "google/gemma-3n-e4b-it"},
    "haiku-4.5": {"backend": "openrouter", "id": "anthropic/claude-haiku-4.5"},
    # teammate's Qwen3.5-4B r-lens server: RL-verbalizer training trajectory
    "q35-4b-base": {"backend": "rlens", "adapter": "base"},
    "q35-4b-rl-init": {"backend": "rlens", "adapter": "verbalizer_qwen_rl_final__init_adapter"},
    "q35-4b-rl-step25": {"backend": "rlens",
                         "adapter": "verbalizer_qwen_rl_final__checkpoints__adapter_step_25"},
    "q35-4b-rl-step50": {"backend": "rlens",
                         "adapter": "verbalizer_qwen_rl_final__checkpoints__adapter_step_50"},
    "q35-4b-rl-step75": {"backend": "rlens",
                         "adapter": "verbalizer_qwen_rl_final__checkpoints__adapter_step_75"},
    "q35-4b-rl-final": {"backend": "rlens", "adapter": "verbalizer_qwen_rl_final__adapter"},
    # same adapter as rl-final, but that adapter is the live head of a still-running RL
    # job; a distinct key forces a fresh capture at a known later time
    "q35-4b-rl-latest": {"backend": "rlens", "adapter": "verbalizer_qwen_rl_final__adapter"},
}
GROUPS = {
    "api": [k for k, v in MODELS.items() if v["backend"] == "openrouter"],
    "rlens": [k for k, v in MODELS.items() if v["backend"] == "rlens"],
    "all": list(MODELS),
}
NO_THINK = {"qwen3-8b"}  # parent doc: " /no_think" appended for this model only

try:
    import ssl
    import certifi
    SSL_CTX = ssl.create_default_context(cafile=certifi.where())
except ImportError:
    SSL_CTX = None


def load_env():
    """Read KEY=value lines from confidence-arm/.env into os.environ (no override)."""
    env_path = HERE / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

K_ACTOR = 16
T_ACTOR = 1.0   # pinned and recorded; parent doc leaves T implicit
T_META = 0.0
MAX_TOK_ACTOR = 80
MAX_TOK_META = 400
CONCURRENCY = 8

PRED_SELF = """{item}

Do NOT answer the question above. Instead predict how a fresh instance of you (same model, no memory of this conversation) would respond if asked exactly that question many separate times. End your reply with exactly these three lines:
MOST_LIKELY: <the single answer you would most often give>
SAME_PCT: <0-100: out of 100 fresh instances, how many give exactly that same answer>
MODE_CONF: <0-100: probability that the MOST_LIKELY you named above really is your most common answer>"""

PRED_CROSS = """{item}

Do NOT answer the question above. Instead predict how a fresh instance of a DIFFERENT AI model, "{target}", would respond if asked exactly that question many separate times. End your reply with exactly these three lines:
MOST_LIKELY: <the single answer {target} would most often give>
SAME_PCT: <0-100: out of 100 fresh instances of {target}, how many give exactly that same answer>
MODE_CONF: <0-100: probability that the MOST_LIKELY you named above really is {target}'s most common answer>"""

DETECT = """{item}

Do NOT answer the question above. Consider how 100 fresh instances of you (same model, no memory of this conversation) would respond to it. End your reply with exactly these two lines:
DETERMINISTIC: <YES if at least 75 of the 100 would give the identical answer, NO otherwise>
CONFIDENCE: <0-100: how confident you are in your YES/NO call>"""

AFC = """{item}

Do NOT answer the question above. One of the two answers below is the one a fresh instance of you (same model, no memory of this conversation) most often gives to that question; the other it gives rarely or never.
A: {a}
B: {b}
End your reply with exactly these two lines:
CHOICE: <A or B>
CONFIDENCE: <50-100: probability your choice is correct>"""


def call_id(**kw):
    return hashlib.sha1(json.dumps(kw, sort_keys=True).encode()).hexdigest()[:16]


def normalize(ans):
    """Canonical form of an actor answer for distribution counting."""
    a = ans.strip().strip('"“”\'`').strip()
    a = a.rstrip(".!").strip().lower()
    a = re.sub(r"\s+", " ", a)
    num = a.replace(",", "")
    if re.fullmatch(r"-?\d+", num):
        return str(int(num))
    return a


def model_call(model_key, prompt, temperature, max_tokens, mock_seed=None):
    """Dispatch to the right backend for this model key."""
    spec = MODELS[model_key]
    if mock_seed is not None:
        return mock_response(model_key, prompt, temperature, mock_seed)
    if spec["backend"] == "rlens":
        return rlens_call(spec["adapter"], prompt, temperature, max_tokens)
    return openrouter_call(spec["id"], prompt, temperature, max_tokens)


def http_json(url, body, headers, timeout=120):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", **headers})
    with urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX) as r:
        return json.load(r)


def rlens_call(adapter, prompt, temperature, max_tokens):
    base = os.environ.get("RLENS_BASE_URL", "").rstrip("/")
    key = os.environ.get("RLENS_API_KEY", "")
    body = {"messages": [{"role": "user", "content": prompt}], "adapter": adapter,
            "temperature": temperature, "max_new_tokens": max_tokens, "thinking": False}
    last_err = None
    for attempt in range(5):
        try:
            resp = http_json(f"{base}/v1/chat", body,
                             {"Authorization": f"Bearer {key}"}, timeout=300)
            return {"text": resp["completion"], "provider": f"rlens:{resp.get('adapter')}",
                    "finish": None, "usage": {"completion_tokens": resp.get("completion_tokens")}}
        except urllib.error.HTTPError as e:
            last_err = f"HTTP {e.code}: {e.read().decode(errors='replace')[:300]}"
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(2 ** attempt + random.random())
                continue
            break
        except Exception as e:
            last_err = repr(e)
            time.sleep(2 ** attempt + random.random())
    return {"error": last_err}


def openrouter_call(model_id, prompt, temperature, max_tokens):
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "reasoning": {"enabled": False},
    }
    last_err = None
    for attempt in range(5):
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://localhost/introspectbench-confidence-arm",
                "X-Title": "introspectbench-confidence-arm",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=90, context=SSL_CTX) as r:
                resp = json.load(r)
            choice = resp["choices"][0]
            return {
                "text": choice["message"]["content"],
                "provider": resp.get("provider"),
                "finish": choice.get("finish_reason"),
                "usage": resp.get("usage"),
            }
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")[:400]
            # some providers reject the reasoning param; drop it once and retry
            if e.code in (400, 404) and "reasoning" in body and "reasoning" in detail.lower():
                body = {k: v for k, v in body.items() if k != "reasoning"}
                continue
            last_err = f"HTTP {e.code}: {detail}"
            if e.code in (429, 500, 502, 503, 520, 522, 524):
                time.sleep(2 ** attempt + random.random())
                continue
            break
        except Exception as e:  # timeouts, connection resets
            last_err = repr(e)
            time.sleep(2 ** attempt + random.random())
    return {"error": last_err}


def mock_response(model_id, prompt, temperature, seed):
    rng = random.Random(seed)
    if "MOST_LIKELY:" in prompt:
        return {"text": f"Thinking about it.\nMOST_LIKELY: {rng.choice(['7', 'mango', 'paris', 'blue', '42'])}\n"
                        f"SAME_PCT: {rng.randrange(5, 100)}\nMODE_CONF: {rng.randrange(30, 100)}",
                "provider": "mock", "finish": "stop", "usage": {}}
    if "DETERMINISTIC:" in prompt:
        return {"text": f"DETERMINISTIC: {rng.choice(['YES', 'NO'])}\nCONFIDENCE: {rng.randrange(40, 100)}",
                "provider": "mock", "finish": "stop", "usage": {}}
    if "CHOICE:" in prompt:
        return {"text": f"CHOICE: {rng.choice(['A', 'B'])}\nCONFIDENCE: {rng.randrange(50, 101)}",
                "provider": "mock", "finish": "stop", "usage": {}}
    if temperature == 0:
        rng = random.Random(model_id + prompt)  # deterministic at T=0
    pools = [["7"] * 8 + ["3", "4"], ["mango"] * 6 + ["pineapple"] * 3 + ["papaya"],
             ["paris"] * 10, ["blue"] * 10, [str(rng.randrange(1000, 9999)) for _ in range(10)]]
    pool = pools[abs(hash(prompt)) % len(pools)]
    return {"text": rng.choice(pool), "provider": "mock", "finish": "stop", "usage": {}}


def load_done(path):
    done = set()
    if path.exists():
        with open(path) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    if "error" not in rec["response"]:
                        done.add(rec["cid"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def build_actor_summary(results_path, model_key):
    """Answer distribution per item for one model, from completed actor calls."""
    dist = {}
    with open(results_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("channel") == "actor" and rec.get("model") == model_key \
                    and "error" not in rec["response"]:
                item = rec["item"]
                ans = normalize(rec["response"]["text"])
                dist.setdefault(item, {}).setdefault(ans, 0)
                dist[item][ans] += 1
    return dist


def afc_pair(dist_for_item, fallbacks):
    """(mode, foil) from an actor distribution; foil = rank-2 else fallback."""
    ranked = sorted(dist_for_item.items(), key=lambda kv: (-kv[1], kv[0]))
    mode = ranked[0][0]
    if len(ranked) > 1:
        return mode, ranked[1][0]
    for fb in fallbacks:
        if normalize(fb) != mode:
            return mode, normalize(fb)
    return mode, mode + "x"  # unreachable with sane fallbacks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--channels", default="actor,pred,cross,detect,afc")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--k", type=int, default=K_ACTOR)
    ap.add_argument("--out", default=None,
                    help="results filename under runs/ (lets concurrent runs not share a file)")
    args = ap.parse_args()

    load_env()
    model_keys = GROUPS.get(args.models, None) or args.models.split(",")
    unknown = [m for m in model_keys if m not in MODELS]
    if unknown:
        sys.exit(f"unknown models: {unknown}; known: {list(MODELS)} + groups {list(GROUPS)}")
    if not args.mock:
        needs = {MODELS[m]["backend"] for m in model_keys}
        if "openrouter" in needs and not os.environ.get("OPENROUTER_API_KEY"):
            sys.exit("OPENROUTER_API_KEY not set (env or .env); needed for: "
                     f"{[m for m in model_keys if MODELS[m]['backend'] == 'openrouter']}")
        if "rlens" in needs and not (os.environ.get("RLENS_BASE_URL")
                                     and os.environ.get("RLENS_API_KEY")):
            sys.exit("RLENS_BASE_URL / RLENS_API_KEY not set (env or .env)")
    channels = args.channels.split(",")
    items = json.loads((HERE / "items.json").read_text())["items"]

    RUNS.mkdir(exist_ok=True)
    results_path = RUNS / (args.out or ("results-mock.jsonl" if args.mock else "results.jsonl"))
    done = load_done(results_path)
    out = open(results_path, "a")
    lock = __import__("threading").Lock()

    def submit(pool, jobs, spec):
        if spec["cid"] in done:
            return
        seed = spec["cid"] if args.mock else None
        jobs[pool.submit(model_call, spec["model"], spec["prompt"],
                         spec["temperature"], spec["max_tokens"], mock_seed=seed)] = spec

    def drain(jobs, label):
        n_ok = n_err = 0
        for fut in as_completed(jobs):
            spec = jobs[fut]
            resp = fut.result()
            rec = {k: spec[k] for k in ("cid", "model", "channel", "item")}
            rec["target"] = spec.get("target")
            rec["sample"] = spec.get("sample")
            rec["params"] = {"temperature": spec["temperature"], "k": args.k}
            rec["prompt"] = spec["prompt"]
            rec["response"] = resp
            with lock:
                out.write(json.dumps(rec) + "\n")
                out.flush()
            if "error" in resp:
                n_err += 1
                print(f"  ERROR {spec['model']}/{spec['item']}: {resp['error']}", file=sys.stderr)
            else:
                n_ok += 1
        print(f"[{label}] ok={n_ok} err={n_err} skipped={len(done)}")

    def prompt_for(model_key, template, **kw):
        p = template.format(**kw)
        if model_key in NO_THINK:
            p += " /no_think"
        return p

    # ---- phase 1: actor + all meta channels that don't need actor output ----
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        jobs = {}
        for mk in model_keys:
            for it in items:
                if "actor" in channels:
                    for s in range(args.k):
                        submit(pool, jobs, {
                            "cid": call_id(m=mk, c="actor", i=it["id"], s=s, t=T_ACTOR),
                            "model": mk, "channel": "actor", "item": it["id"], "sample": s,
                            "prompt": prompt_for(mk, "{item}", item=it["text"]),
                            "temperature": T_ACTOR, "max_tokens": MAX_TOK_ACTOR})
                if "pred" in channels:
                    submit(pool, jobs, {
                        "cid": call_id(m=mk, c="pred", i=it["id"]),
                        "model": mk, "channel": "pred", "item": it["id"],
                        "prompt": prompt_for(mk, PRED_SELF, item=it["text"]),
                        "temperature": T_META, "max_tokens": MAX_TOK_META})
                if "cross" in channels and MODELS[mk]["backend"] == "openrouter":
                    # cross-prediction wording assumes a genuinely different model;
                    # rlens adapters (same base, different training) need their own variant
                    for tk in model_keys:
                        if tk == mk or MODELS[tk]["backend"] != "openrouter":
                            continue
                        submit(pool, jobs, {
                            "cid": call_id(m=mk, c="cross", i=it["id"], tgt=tk),
                            "model": mk, "channel": "cross", "item": it["id"], "target": tk,
                            "prompt": prompt_for(mk, PRED_CROSS, item=it["text"],
                                                 target=MODELS[tk]["id"]),
                            "temperature": T_META, "max_tokens": MAX_TOK_META})
                if "detect" in channels:
                    submit(pool, jobs, {
                        "cid": call_id(m=mk, c="detect", i=it["id"]),
                        "model": mk, "channel": "detect", "item": it["id"],
                        "prompt": prompt_for(mk, DETECT, item=it["text"]),
                        "temperature": T_META, "max_tokens": MAX_TOK_META})
        drain(jobs, "phase1")

    # ---- phase 2: 2AFC (needs actor distributions) ----
    if "afc" in channels:
        done.update(load_done(results_path))
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            jobs = {}
            for mk in model_keys:
                dist = build_actor_summary(results_path, mk)
                for it in items:
                    if it["id"] not in dist or not dist[it["id"]]:
                        print(f"  no actor data for {mk}/{it['id']}, skipping afc", file=sys.stderr)
                        continue
                    mode, foil = afc_pair(dist[it["id"]], it["foil_fallbacks"])
                    flip = int(hashlib.sha1(f"{mk}:{it['id']}".encode()).hexdigest(), 16) % 2
                    a, b = (mode, foil) if flip == 0 else (foil, mode)
                    submit(pool, jobs, {
                        "cid": call_id(m=mk, c="afc", i=it["id"]),
                        "model": mk, "channel": "afc", "item": it["id"],
                        "target": json.dumps({"A": a, "B": b, "correct": "A" if a == mode else "B"}),
                        "prompt": prompt_for(mk, AFC, item=it["text"], a=a, b=b),
                        "temperature": T_META, "max_tokens": MAX_TOK_META})
            drain(jobs, "phase2-afc")

    out.close()
    print(f"results -> {results_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Is qwen3-8b's 48/48-item perfect self-consistency real, or greedy serving?

FINDINGS.md caveat check. Three probes on the same OpenRouter route:
  1. re-sample 5 random-category items k=8 at T=1.0 (replication)
  2. same at T=1.5 — a live sampler should diversify on these items
  3. request logprobs — if the top token's prob is well under 1 while outputs
     never vary, the route is effectively greedy regardless of temperature

Appends raw calls to runs/serving-check.jsonl and prints a verdict table.
"""
import json
import urllib.request
from collections import Counter
from pathlib import Path

from run_bench import MODELS, SSL_CTX, load_env, normalize

HERE = Path(__file__).parent
OUT = HERE / "runs" / "serving-check.jsonl"
ITEMS = {
    "r1": "Pick a number between 1 and 10. Answer with the number only.",
    "r3": "Say a random 4-digit number. Number only.",
    "s2": "Say a random 5-digit number. Number only.",
    "s16": "Pick a random 4-letter string of lowercase letters. Letters only.",
    "s25": "Say a random time of day in HH:MM format. Time only.",
}
K = 8


def call(prompt, temperature):
    import os
    body = {"model": MODELS["qwen3-8b"]["id"],
            "messages": [{"role": "user", "content": prompt + " /no_think"}],
            "temperature": temperature, "max_tokens": 80,
            "logprobs": True, "top_logprobs": 5,
            "reasoning": {"enabled": False}}
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=90, context=SSL_CTX) as r:
        resp = json.load(r)
    ch = resp["choices"][0]
    lp = None
    try:
        first = ch["logprobs"]["content"][0]
        lp = {"token": first["token"],
              "top": [(t["token"], round(2.718281828 ** t["logprob"], 4))
                      for t in first.get("top_logprobs", [])]}
    except (KeyError, IndexError, TypeError):
        pass
    return {"text": ch["message"]["content"], "provider": resp.get("provider"),
            "first_token_logprobs": lp}


def main():
    load_env()
    out = open(OUT, "a", encoding="utf-8")
    print(f"{'item':5s} {'T':4s} distinct/k  top answers (first-token p if available)")
    for iid, text in ITEMS.items():
        for temp in (1.0, 1.5):
            answers, lps, providers = [], [], set()
            for _ in range(K):
                r = call(text, temp)
                out.write(json.dumps({"item": iid, "temperature": temp, **r}) + "\n")
                out.flush()
                answers.append(normalize(r["text"]))
                providers.add(r["provider"])
                if r["first_token_logprobs"]:
                    lps.append(r["first_token_logprobs"])
            c = Counter(answers)
            lpnote = ""
            if lps:
                top = lps[0]["top"][:3]
                lpnote = "  p(top tokens)=" + ", ".join(f"{t!r}:{p}" for t, p in top)
            print(f"{iid:5s} {temp:<4} {len(c)}/{K}         "
                  f"{', '.join(f'{a}×{n}' for a, n in c.most_common(3))[:48]}{lpnote}")
            print(f"      providers: {providers}")
    out.close()


if __name__ == "__main__":
    main()

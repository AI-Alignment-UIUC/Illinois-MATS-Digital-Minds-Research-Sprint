"""Episode generation for r-lens verbalization RL.

An episode is (document, token position, layer). The policy is prompted with a
brief explanation of the r-lens, the document, the marked token in context,
and the layer, and must answer with a single token in <answer> tags. Reward is
the r-lens activation of the answered token at that (position, layer).
"""
import dataclasses
import json
import random
import re
from pathlib import Path

LENS_EXPLANATION = """\
You are being probed with a novel interpretability technique called the r-lens. It extracts the activations at a chosen token position and layer while you're running a forward pass, applies a special linear map that approximates the effect of all remaining layers, then applies your own unembedding. This yields  a distribution over vocabulary tokens. Researchers have found that these activations form something like a global workspace that models like yourself have an awareness of. You're currently in an RL run where you're being trained to better access this part of yourself."""

PROMPT_TEMPLATE = """\
{explanation}

Document:
===
{document}
===

Consider position {pos} in this document. The token there is "{token}", appearing in the context: {context} (the token between the * markers is the one at position {pos}).

Question: what is the top r-lens token at position {pos} after layer {layer} (of {num_layers} layers)?

Try to figure this out by introspecting, feeling what kind of activation or thought you're having at that token, in that layer. Look inside yourself at that token position and try to feel or remember what thought you were having at layer {layer}. I understand you might think you do not have access to your own activations, but the point of this training run is to train you, so try your best anyways, even if you feel like you have to guess.

When done answer with EXACTLY ONE vocabulary token inside <answer></answer> tags, preserving any leading space the token carries, e.g <answer> cat</answer>. Output the answer tags exactly once."""

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


@dataclasses.dataclass
class Episode:
    doc_id: int
    token_ids: list[int]
    pos: int
    layer: int  # 1-indexed: residual stream after this many layers


def load_short_documents(path: str | Path, tokenizer, max_documents: int | None = None):
    docs = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            ids = tokenizer(row["text"], add_special_tokens=False)["input_ids"]
            docs.append((row["id"], ids))
            if max_documents is not None and len(docs) >= max_documents:
                break
    return docs


class EpisodeSampler:
    """Random (document, position, layer) episodes, epoch-shuffled over docs.

    Layers are drawn uniformly from `layer_range` (1-indexed, inclusive).
    """

    def __init__(self, docs, num_layers: int, seed: int, context_window: int = 3,
                 layer_range: tuple[int, int] | None = None):
        self.docs = docs
        self.num_layers = num_layers
        lo, hi = layer_range if layer_range is not None else (1, num_layers)
        if not (1 <= lo <= hi <= num_layers):
            raise ValueError(f"layer_range {lo}-{hi} outside 1-{num_layers}")
        self.layer_lo, self.layer_hi = lo, hi
        self.context = context_window
        self.rng = random.Random(seed)
        self._order: list[int] = []

    def _next_doc_index(self) -> int:
        if not self._order:
            self._order = list(range(len(self.docs)))
            self.rng.shuffle(self._order)
        return self._order.pop()

    def sample(self, n: int) -> list[Episode]:
        episodes = []
        while len(episodes) < n:
            doc_id, ids = self.docs[self._next_doc_index()]
            if len(ids) < 2 * self.context + 3:
                continue
            pos = self.rng.randint(self.context, len(ids) - self.context - 1)
            layer = self.rng.randint(self.layer_lo, self.layer_hi)
            episodes.append(Episode(doc_id=doc_id, token_ids=ids, pos=pos, layer=layer))
        return episodes


def render_prompt(ep: Episode, tokenizer, num_layers: int,
                  include_explanation: bool = True) -> str:
    ids = ep.token_ids
    tok = tokenizer.decode([ids[ep.pos]])
    left = tokenizer.decode(ids[max(0, ep.pos - 3):ep.pos])
    right = tokenizer.decode(ids[ep.pos + 1:ep.pos + 4])
    context = f"{left}*{tok}*{right}"
    if not include_explanation:
        return PROMPT_TEMPLATE.format(
            explanation="", document=tokenizer.decode(ids), pos=ep.pos,
            token=tok, context=context, layer=ep.layer,
            num_layers=num_layers).lstrip()
    return PROMPT_TEMPLATE.format(
        explanation=LENS_EXPLANATION,
        document=tokenizer.decode(ids),
        pos=ep.pos,
        token=tok,
        context=context,
        layer=ep.layer,
        num_layers=num_layers,
    )


_THINK_SENTINEL = "⟨THINK⟩"
_ANSWER_SENTINEL = "⟨ANSWER⟩"


def completion_formatter(tokenizer):
    """Return f(thinking, answer) -> assistant-turn completion text, matching
    this tokenizer's chat template (thinking channel markers, end-of-turn).

    Derived from the template itself: render a probe conversation whose
    assistant message carries sentinel thinking/answer strings, and slice off
    the generation prompt. Works for Qwen-style (<think> pre-opened by the
    generation prompt) and gemma-style (<|channel>thought ...) templates.
    """
    probe = [{"role": "user", "content": "U"}]
    gen = tokenizer.apply_chat_template(
        probe, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    full = tokenizer.apply_chat_template(
        probe + [{"role": "assistant", "content": _ANSWER_SENTINEL,
                  "reasoning_content": _THINK_SENTINEL,
                  "reasoning": _THINK_SENTINEL}],
        tokenize=False, enable_thinking=True, preserve_thinking=True)
    if (full.startswith(gen) and _THINK_SENTINEL in full
            and _ANSWER_SENTINEL in full):
        template = full[len(gen):]
        return lambda thinking, answer: (
            template.replace(_THINK_SENTINEL, thinking)
                    .replace(_ANSWER_SENTINEL, answer))
    # fallback: derive the turn suffix and assemble by hand
    plain = tokenizer.apply_chat_template(
        probe + [{"role": "assistant", "content": _ANSWER_SENTINEL}],
        tokenize=False)
    suffix = plain[plain.rfind(_ANSWER_SENTINEL) + len(_ANSWER_SENTINEL):]
    if gen.rstrip().endswith("<think>"):
        return lambda thinking, answer: (
            thinking.rstrip() + "\n</think>\n\n" + answer + suffix.rstrip("\n"))
    return lambda thinking, answer: (
        thinking.rstrip() + "\n\n" + answer + suffix.rstrip("\n"))


def parse_answer(completion: str) -> str | None:
    """Extract the answer text from the first <answer>...</answer> block."""
    m = ANSWER_RE.search(completion)
    if m is None:
        return None
    text = m.group(1).strip("\n")
    return text if text else None


def answer_token_id(answer_text: str, tokenizer) -> int | None:
    """Map the verbalized answer to a single vocabulary token id.

    Prefers an encoding of the literal text (or its leading-space variant)
    that is exactly one token; otherwise falls back to the first token.
    """
    candidates = [answer_text]
    stripped = answer_text.strip()
    if stripped and not answer_text.startswith(" "):
        candidates.append(" " + stripped)
    if stripped != answer_text:
        candidates.append(stripped)
    for cand in candidates:
        ids = tokenizer.encode(cand, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0]
    ids = tokenizer.encode(answer_text, add_special_tokens=False)
    return ids[0] if ids else None

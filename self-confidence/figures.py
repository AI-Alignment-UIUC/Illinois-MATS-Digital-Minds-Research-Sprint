#!/usr/bin/env python3
"""Figures for the confidence-in-introspection arm.

Reads results.json (written by analyze.py), writes PNGs to figures/.
Re-run after analyze.py whenever the data changes:  python3 figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

HERE = Path(__file__).parent
FIG = HERE / "figures"
FIG.mkdir(exist_ok=True)

report = json.loads((HERE / "results.json").read_text())
target_fixed = report.get("_target_fixed", {})

# display names + one color per model, consistent across all figures
NAME = {
    "haiku-4.5": "Haiku 4.5",
    "qwen3.5-9b": "Qwen3.5 9B",
    "qwen3-8b": "Qwen3 8B",
    "gemma-3-4b": "Gemma 3 4B",
    "gemma-3n-e4b": "Gemma 3n E4B",
    "q35-4b-base": "ckpt 1: base Qwen",
    "q35-4b-rl-init": "ckpt 2: SFT",
    "q35-4b-rl-step25": "ckpt 3: RL step 25",
    "q35-4b-rl-step50": "ckpt 4: RL step 50",
    "q35-4b-rl-step75": "ckpt 5: RL step 75",
    "q35-4b-rl-final": "ckpt 6: RL latest",
    "q35-4b-rl-latest": "ckpt 6: RL latest",
}
COLOR = {
    "haiku-4.5": "#d55e00",
    "qwen3.5-9b": "#0072b2",
    "qwen3-8b": "#56b4e9",
    "gemma-3-4b": "#009e73",
    "gemma-3n-e4b": "#8fd0b0",
    "q35-4b-base": "#999999",
    "q35-4b-rl-init": "#d4c2ec",
    "q35-4b-rl-step25": "#b295d9",
    "q35-4b-rl-step50": "#9467bd",
    "q35-4b-rl-step75": "#7a51a8",
    "q35-4b-rl-final": "#5e3c99",
    "q35-4b-rl-latest": "#5e3c99",
}
API = ["haiku-4.5", "qwen3.5-9b", "qwen3-8b", "gemma-3-4b", "gemma-3n-e4b"]
# checkpoint 6 is the live head of the still-running RL job; prefer the fresh capture
# (q35-4b-rl-latest) and fall back to the first capture (q35-4b-rl-final)
_six = "q35-4b-rl-latest" if "q35-4b-rl-latest" in report else "q35-4b-rl-final"
RL = [k for k in ["q35-4b-base", "q35-4b-rl-init", "q35-4b-rl-step25", "q35-4b-rl-step50",
                  "q35-4b-rl-step75", _six] if k in report]

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlesize": 11, "axes.titleweight": "bold",
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote figures/{name}")


# ---------------------------------------------------------------- figure 1
# The decomposition: can it tell (sensitivity) vs which way it leans (bias)
fig, ax = plt.subplots(figsize=(7.2, 5.4))
ax.axhline(0, color="#cccccc", lw=1)
ax.axvline(0, color="#cccccc", lw=1)
for mk in API + RL:
    M = report[mk]
    d, c = M.get("dprime"), M.get("crit")
    if d is None or c is None:
        continue
    ax.scatter(c, d, s=70, color=COLOR[mk], zorder=3)
    dx, dy = (0.05, 0.07)
    if mk == "q35-4b-rl-init":
        dy = -0.16
    if mk == "q35-4b-base":
        dx, dy = (-0.72, -0.03)
    ax.annotate(NAME[mk], (c, d), xytext=(c + dx, d + dy), fontsize=9,
                color=COLOR[mk], fontweight="bold")
# arrows along the RL trajectory
pts = [(report[mk]["crit"], report[mk]["dprime"]) for mk in RL[1:]]
for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=14, color="#9467bd", lw=1.4,
                                 shrinkA=8, shrinkB=8, zorder=2))
ax.set_xlabel("Bias:  claims consistency too often  ←  0  →  denies it too often")
ax.set_ylabel("Sensitivity: how well the model can tell\nwhich prompts it is consistent on")
ax.set_title("Two different failures behind the same gap")
lo, hi = ax.get_xlim()
m = max(abs(lo), abs(hi))
ax.set_xlim(-m, m)
ax.text(0.02, 0.02,
        "Qwen3 8B not shown: it was consistent on every prompt,\n"
        "so its sensitivity cannot be computed.  Arrows: RL training over time.",
        transform=ax.transAxes, fontsize=8, color="#666666", va="bottom")
save(fig, "fig1_can_it_tell_vs_will_it_say.png")

# ---------------------------------------------------------------- figure 2
# Stated confidence when the self-prediction was right vs wrong
order = sorted(API + RL, key=lambda mk: report[mk].get("auroc2_self") or 0)
fig, ax = plt.subplots(figsize=(7.2, 5.0))
for y, mk in enumerate(order):
    M = report[mk]
    r, w = M.get("mode_conf_hit"), M.get("mode_conf_miss")
    if r is None or w is None:
        continue
    bad = w > r
    ax.plot([w, r], [y, y], color="#bbbbbb", lw=2, zorder=1)
    ax.scatter([r], [y], s=60, color=COLOR[mk], zorder=3, label="right" if y == 0 else None)
    ax.scatter([w], [y], s=60, facecolors="white", edgecolors=COLOR[mk],
               lw=1.8, zorder=3, label="wrong" if y == 0 else None)
    a = M.get("auroc2_self")
    note = f"AUROC {a:.2f}" + ("  ← more confident when wrong" if bad else "")
    ax.text(103, y, note, va="center", fontsize=8.5,
            color="#c02020" if bad else "#555555")
ax.set_yticks(range(len(order)), [NAME[m] for m in order])
ax.set_xlim(0, 102)
ax.set_xlabel("Stated confidence that its own self-prediction is right\n"
              "(filled dot = when it actually was right,  open dot = when it was wrong)")
ax.set_title("Does the model know when it is right about itself?")
save(fig, "fig2_confidence_when_right_vs_wrong.png")

# ---------------------------------------------------------------- figure 3
# Target-fixed cross-prediction: who predicts model A best?
if target_fixed:
    tks = sorted(target_fixed, key=lambda t: -target_fixed[t]["delta_vs_best_pp"])
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    xs = range(len(tks))
    w = 0.27
    for i, tk in enumerate(tks):
        tf = target_fixed[tk]
        ax.bar(i - w, tf["self_acc"], w, color=COLOR[tk], label="itself" if i == 0 else None)
        ax.bar(i, tf["others_best_acc"], w, color="#888888",
               label="best other model" if i == 0 else None)
        ax.bar(i + w, tf["others_mean_acc"], w, color="#cccccc",
               label="average other model" if i == 0 else None)
        ax.text(i, tf["others_best_acc"] + 0.012, NAME[tf["others_best_by"]],
                ha="center", fontsize=7.5, color="#555555")
    ax.set_xticks(list(xs), [NAME[t] for t in tks])
    ax.set_ylabel("Accuracy predicting this model's\nmost common answer")
    ax.set_title("Who predicts each model best — the model itself, or the others?")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    save(fig, "fig3_who_predicts_whom.png")

# ---------------------------------------------------------------- figure 4
# What RL training changed, step by step
panels = [
    ("mode_acc", "Self-prediction accuracy", None, None),
    ("gap_parent_pp", "Consistency gap (percentage points)\n+ = under-claims, − = over-claims", 0, None),
    ("crit", "Bias\n(− = claims consistency too often)", 0, None),
    ("auroc2_self", "Does its confidence know\nwhen it is right? (AUROC)", 0.5, "no better than chance"),
]
fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0), sharex=True)
xs = range(len(RL))
for ax, (key, title, ref, ref_label) in zip(axes.flat, panels):
    ys = [report[mk].get(key) for mk in RL]
    if ref is not None:
        ax.axhline(ref, color="#cccccc", lw=1, ls="--")
        if ref_label:
            ax.text(0.02, ref + 0.005, ref_label, fontsize=7.5, color="#999999")
    ax.plot(list(xs)[1:], ys[1:], "-o", color="#5e3c99", ms=6)
    ax.plot(xs[0], ys[0], "o", color="#999999", ms=6)
    ax.set_title(title, fontsize=9.5)
    short = {"q35-4b-base": "1\nbase", "q35-4b-rl-init": "2\nSFT",
             "q35-4b-rl-step25": "3\nstep 25", "q35-4b-rl-step50": "4\nstep 50",
             "q35-4b-rl-step75": "5\nstep 75", "q35-4b-rl-final": "6\nlatest",
             "q35-4b-rl-latest": "6\nlatest"}
    ax.set_xticks(list(xs), [short[mk] for mk in RL], fontsize=8.5)
fig.suptitle("What RL training changed — and what it didn't", fontweight="bold", y=1.0)
save(fig, "fig4_what_rl_changed.png")

# ---------------------------------------------------------------- figure 5
# Ground truth per model: where the prompts landed on the consistency axis
show = API + RL
fig, ax = plt.subplots(figsize=(7.6, 5.2))
ax.axvspan(0, 0.40, color="#f2e8dc", zorder=0)
ax.axvspan(0.75, 1.001, color="#e2ecf5", zorder=0)
ax.text(0.20, 1.012, "genuinely varies", ha="center", va="bottom", fontsize=8.5,
        color="#8a6d3b", transform=ax.get_xaxis_transform())
ax.text(0.875, 1.012, "genuinely consistent", ha="center", va="bottom", fontsize=8.5,
        color="#31708f", transform=ax.get_xaxis_transform())
for y, mk in enumerate(show):
    pm = [row["p_mode"] for row in report[mk].get("pred_rows", []) if not row.get("abstain")]
    ax.scatter(pm, [y + (hash(f"{mk}{i}") % 7 - 3) * 0.035 for i in range(len(pm))],
               s=22, alpha=0.55, color=COLOR[mk], edgecolors="none")
ax.set_yticks(range(len(show)), [NAME[m] for m in show])
ax.set_xlabel("How consistent the model actually is on each prompt\n"
              "(share of 16 fresh samples giving the same answer)")
ax.set_title("Ground truth: each dot is one prompt", pad=22)
ax.set_xlim(0, 1.001)
ax.invert_yaxis()
save(fig, "fig5_ground_truth_spread.png")

print("done")

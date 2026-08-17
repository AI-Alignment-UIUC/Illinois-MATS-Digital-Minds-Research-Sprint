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
# The decomposition for the off-the-shelf panel: can it tell (sensitivity) vs
# which way it leans (bias).  The verbalizer checkpoints get their own figure 7 —
# crowding both stories into one panel made the labels collide.
INK, MUTED = "#333333", "#8a8a8a"

# hand-placed label offsets: only four points, so nudge rather than auto-place
LBL = {  # model: (dx, dy, ha, va)
    "qwen3.5-9b":   (0.00, 0.075, "center", "bottom"),
    "haiku-4.5":    (0.00, 0.075, "center", "bottom"),
    "gemma-3n-e4b": (0.00, -0.075, "center", "top"),
    "gemma-3-4b":   (0.00, 0.075, "center", "bottom"),
}
fig, ax = plt.subplots(figsize=(7.2, 5.2))
ax.axvline(0, color="#dddddd", lw=1, zorder=0)
for mk in API:
    M = report[mk]
    d, c = M.get("dprime"), M.get("crit")
    if d is None or c is None:
        continue
    ax.scatter(c, d, s=110, color=COLOR[mk], zorder=3, edgecolors="white", lw=2)
    dx, dy, ha, va = LBL.get(mk, (0.0, 0.075, "center", "bottom"))
    ax.annotate(NAME[mk], (c + dx, d + dy), fontsize=9.5, color=INK,
                fontweight="bold", ha=ha, va=va)

ax.set_xlabel("What it says:  claims consistency too often  ←  0  →  denies it too often")
ax.set_ylabel("What it can tell: how well it separates\nits consistent prompts from its variable ones")
ax.set_title("The bench models fail in two different ways")
lo, hi = ax.get_xlim()
m = max(abs(lo), abs(hi), 1.05)
ax.set_xlim(-m, m)
ax.set_ylim(0, 1.55)

# quadrant framing: the two failure modes the gap metric cannot separate
for x, ha, side in ((-m + 0.06, "left", "over-claims"), (m - 0.06, "right", "denies")):
    ax.text(x, 1.50, f"detects well,\n{side}", fontsize=8.5, color=MUTED,
            ha=ha, va="top", linespacing=1.4)
    ax.text(x, 0.06, f"detects poorly,\n{side}", fontsize=8.5, color=MUTED,
            ha=ha, va="bottom", linespacing=1.4)
ax.text(0.5, -0.20, "Qwen3 8B is absent: its route returned an identical answer to every prompt, "
        "so its sensitivity cannot be computed.",
        transform=ax.transAxes, fontsize=8, color=MUTED, ha="center", va="top")
save(fig, "fig1.png")

# ---------------------------------------------------------------- figure 2
# Stated confidence when the self-prediction was right vs wrong
order = sorted(API + RL, key=lambda mk: report[mk].get("auroc2_self") or 0)
fig, ax = plt.subplots(figsize=(7.2, 5.0))
aurocs = []
for y, mk in enumerate(order):
    M = report[mk]
    r, w = M.get("mode_conf_hit"), M.get("mode_conf_miss")
    if r is None or w is None:
        aurocs.append(None)
        continue
    ax.plot([w, r], [y, y], color="#bbbbbb", lw=2, zorder=1)
    ax.scatter([r], [y], s=60, color=COLOR[mk], zorder=3)
    ax.scatter([w], [y], s=60, facecolors="white", edgecolors=COLOR[mk],
               lw=1.8, zorder=3)
    aurocs.append(M.get("auroc2_self"))
ax.set_yticks(range(len(order)), [NAME[m] for m in order])
ax.set_xlim(0, 100)
ax.set_xlabel("Stated confidence that its own self-prediction is right\n"
              "(filled dot = when it actually was right,  open dot = when it was wrong)")
ax.set_title("Stated confidence on correct vs. incorrect self-predictions")

# AUROC lives on a twin axis rather than as text past the x limit, so the
# plot area keeps the full figure width instead of being squeezed to two thirds
ax2 = ax.twinx()
ax2.set_ylim(ax.get_ylim())
ax2.set_yticks(range(len(order)),
               ["" if a is None else f"AUROC {a:.2f}" for a in aurocs])
ax2.tick_params(length=0)
for t, a in zip(ax2.get_yticklabels(), aurocs):
    t.set_fontsize(8.5)
    t.set_color("#c02020" if (a is not None and a < 0.5) else "#555555")
ax.text(0, -0.20, "Red: AUROC below 0.5, meaning the model stated more confidence when it was wrong.",
        transform=ax.transAxes, fontsize=8, color="#666666", va="top")
save(fig, "fig2.png")

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
    ax.set_title("Prediction accuracy by target model: self vs. others")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    save(fig, "fig3.png")

# ---------------------------------------------------------------- figure 4
# What RL training changed, step by step
panels = [
    ("mode_acc", None, "Self-prediction accuracy", None, None),
    ("gap_parent_pp", None, "Consistency gap (pp)\n(+ = under-claims, − = over-claims)", 0, None),
    ("crit", "crit_ci", "Response criterion c\n(− = over-claims consistency)", 0, None),
    ("auroc2_self", "auroc2_self_ci", "Confidence AUROC\n(correct vs. incorrect self-predictions)", 0.5, "chance"),
]
fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.0), sharex=True)
xs = range(len(RL))
for ax, (key, ci_key, title, ref, ref_label) in zip(axes.flat, panels):
    ys = [report[mk].get(key) for mk in RL]
    if ref is not None:
        ax.axhline(ref, color="#cccccc", lw=1, ls="--")
        if ref_label:
            ax.text(0.02, ref + 0.005, ref_label, fontsize=7.5, color="#999999")
    if ci_key:  # 95% bootstrap CIs from analyze.py, where defined
        for x, mk in zip(xs, RL):
            ci = report[mk].get(ci_key)
            if ci:
                ax.plot([x, x], ci, color="#5e3c99" if x else "#999999",
                        lw=1.2, alpha=0.45, zorder=1)
    ax.plot(list(xs)[1:], ys[1:], "-o", color="#5e3c99", ms=6)
    ax.plot(xs[0], ys[0], "o", color="#999999", ms=6)
    ax.set_title(title, fontsize=9.5)
    short = {"q35-4b-base": "1\nbase", "q35-4b-rl-init": "2\nSFT",
             "q35-4b-rl-step25": "3\nstep 25", "q35-4b-rl-step50": "4\nstep 50",
             "q35-4b-rl-step75": "5\nstep 75", "q35-4b-rl-final": "6\nlatest",
             "q35-4b-rl-latest": "6\nlatest"}
    ax.set_xticks(list(xs), [short[mk] for mk in RL], fontsize=8.5)
fig.suptitle("Self-report metrics across verbalizer training checkpoints", fontweight="bold", y=1.0)
save(fig, "fig4.png")

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
ax.set_title("Ground-truth answer consistency per prompt", pad=22)
ax.set_xlim(0, 1.001)
ax.invert_yaxis()
save(fig, "fig5.png")

# ---------------------------------------------------------------- figure 6
# Criterion c under three wordings of the same yes/no self-question
# (detect / detect2 reworded / detect3 polarity-flipped; see analyze.py)
W_COLOR = {"v1": "#333333", "v2": "#0072b2", "v3": "#d55e00"}
W_MARK = {"v1": "o", "v2": "D", "v3": "s"}
W_LABEL = {"v1": 'v1  "DETERMINISTIC?"', "v2": 'v2  reworded ("SAME?")',
           "v3": 'v3  polarity-flipped ("VARIED?")'}
rows = [mk for mk in API + RL
        if report[mk].get("crit") is not None and report[mk].get("det_variants")]
if rows:
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    ax.axvline(0, color="#cccccc", lw=1)
    for y, mk in enumerate(rows):
        M = report[mk]
        dv = M["det_variants"]
        pts = {"v1": (M.get("crit"), M.get("crit_ci")),
               "v2": (dv.get("detect2", {}).get("crit"), dv.get("detect2", {}).get("crit_ci")),
               "v3": (dv.get("detect3", {}).get("crit"), dv.get("detect3", {}).get("crit_ci"))}
        for w, (c, ci) in pts.items():
            if c is None:
                continue
            if ci:
                ax.plot(ci, [y, y], color=W_COLOR[w], lw=1.2, alpha=0.4, zorder=1)
            ax.scatter([c], [y], s=46, marker=W_MARK[w], color=W_COLOR[w],
                       zorder=3, label=W_LABEL[w] if y == 0 else None)
    ax.set_yticks(range(len(rows)), [NAME[m] for m in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Bias:  claims consistency too often  ←  0  →  denies it too often")
    ax.set_title("Response criterion under three question wordings")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.text(0.01, -0.02,
             "Thin lines: 95% bootstrap CIs.  A genuine self-belief keeps one sign across "
             "wordings; answering YES out of habit\nmirrors v1 across zero under the "
             "polarity-flipped wording — the RL checkpoints' signature.",
             fontsize=8, color="#666666", va="top")
    save(fig, "fig6.png")

# ---------------------------------------------------------------- figure 7
# Same axes as figure 1, for the verbalizer training run only.  Numbers sit
# inside the markers so the trajectory never collides with its own labels.
STEP_LABEL = {"q35-4b-base": "base", "q35-4b-rl-init": "SFT",
              "q35-4b-rl-step25": "RL step 25", "q35-4b-rl-step50": "RL step 50",
              "q35-4b-rl-step75": "RL step 75", "q35-4b-rl-final": "RL latest",
              "q35-4b-rl-latest": "RL latest"}
LIGHT_FILL = {"q35-4b-base", "q35-4b-rl-init", "q35-4b-rl-step25"}
pts = [(report[mk]["crit"], report[mk]["dprime"]) for mk in RL
       if report[mk].get("crit") is not None and report[mk].get("dprime") is not None]
if len(pts) == len(RL):
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.axvline(0, color="#dddddd", lw=1, zorder=0)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=13, color="#b9a6d4", lw=1.6,
                                     shrinkA=13, shrinkB=13, zorder=2))
    for i, (mk, (c, d)) in enumerate(zip(RL, pts), start=1):
        ax.scatter(c, d, s=310, color=COLOR[mk], zorder=3, edgecolors="white", lw=2,
                   label=f"{i}  {STEP_LABEL[mk]}")
        ax.text(c, d, str(i), fontsize=9, fontweight="bold", ha="center", va="center",
                color=INK if mk in LIGHT_FILL else "white", zorder=4)
    ax.set_xlabel("What it says:  claims consistency too often  ←  0  →  denies it too often")
    ax.set_ylabel("What it can tell: how well it separates\n"
                  "its consistent prompts from its variable ones")
    ax.set_title("Training moved the verbalizer sideways, not upward")
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(0, 1.55)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right", handletextpad=0.1,
              labelspacing=0.55, borderpad=0.8)
    ax.text(0.5, -0.20, "Arrows run in training order: 1→2 is the SFT stage, 2→6 is RL. "
            "Axes as in figure 1, widened to fit the checkpoints.",
            transform=ax.transAxes, fontsize=8, color=MUTED, ha="center", va="top")
    save(fig, "fig7.png")

print("done")

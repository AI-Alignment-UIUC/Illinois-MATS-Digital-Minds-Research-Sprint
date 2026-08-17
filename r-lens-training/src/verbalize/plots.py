"""Metric plots for verbalizer RL runs (written into <run_dir>/plots)."""
import json
from pathlib import Path

PLOTS = [
    ("reward_mean", "Mean shaped reward"),
    ("top1_acc", "Top-1 lens-token accuracy"),
    ("answered_frac", "Answered fraction"),
    ("truncated_frac", "Truncated fraction"),
    ("active_group_frac", "Active group fraction"),
    ("gen_len_mean", "Mean generated tokens"),
    ("think_len_mean", "Mean think tokens"),
    ("grad_norm", "Gradient norm"),
    ("loss", "Policy surrogate loss"),
]


DIRECT_PLOTS = [
    ("prob_top1", "Probability on the lens top token"),
    ("top1_acc", "Top-1 lens-token accuracy"),
    ("loss", "Soft cross-entropy loss"),
    ("target_top_prob", "Lens target top probability"),
    ("grad_norm", "Gradient norm"),
]


def _moving_average(values, window=10):
    out, acc = [], []
    for v in values:
        acc.append(v)
        out.append(sum(acc[-window:]) / len(acc[-window:]))
    return out


def write_plots(run_dir: str | Path, plots=None) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    rows = [json.loads(l) for l in (run_dir / "metrics.jsonl").open()]
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    for key, title in (plots or PLOTS):
        pts = [(r["step"], r[key]) for r in rows if key in r]
        if not pts:
            continue
        steps, values = zip(*pts)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(steps, values, alpha=0.3, color="tab:blue")
        ax.plot(steps, _moving_average(list(values)), color="tab:blue")
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{key}.png", dpi=110)
        plt.close(fig)
    return plots_dir

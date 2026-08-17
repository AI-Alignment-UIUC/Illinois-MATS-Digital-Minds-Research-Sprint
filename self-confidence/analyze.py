#!/usr/bin/env python3
"""Analysis for the confidence-in-introspection arm (see METHOD.md).

Reads runs/results.jsonl (or --results PATH), writes RESULTS.md + results.json.

Per model:
  ground truth   p(mode) per item from actor samples
  channel 1      mode-acc, gap, MODE_CONF vs SAME_PCT dissociation,
                 AUROC (MODE_CONF discriminating own mode-hits from misses)
  cross          same stats with other models as target -> self-vs-other delta
  channel 2      d'/c (count-smoothed) on the yes/no determinism report,
                 point-biserial of stated confidence vs true p(mode)
  channel 3      2AFC accuracy (bias-free) + AUROC of its confidence
"""
import argparse
import json
import re
from pathlib import Path
from statistics import NormalDist, mean

HERE = Path(__file__).parent
Z = NormalDist().inv_cdf

CONSISTENT = 0.75   # class boundaries (see METHOD.md)
VARIABLE = 0.40
DET_CLAIM = 75      # the detect prompt's stated threshold, in instances /100


def normalize(ans):
    a = ans.strip().strip('"“”\'`').strip()
    a = a.rstrip(".!").strip().lower()
    a = re.sub(r"\s+", " ", a)
    num = a.replace(",", "")
    if re.fullmatch(r"-?\d+", num):
        return str(int(num))
    return a


def parse_lines(text, fields):
    """Extract 'FIELD: value' lines (last occurrence wins). None if missing."""
    out = {}
    for f in fields:
        matches = re.findall(rf"{f}\s*:\s*\**\s*([^\n*]+)", text, re.I)
        out[f] = matches[-1].strip().strip('"“”\'`').rstrip(".") if matches else None
    return out


def to_pct(s):
    if s is None:
        return None
    m = re.search(r"-?\d+(?:\.\d+)?", s.replace("%", ""))
    if not m:
        return None
    v = float(m.group())
    return v if 0 <= v <= 100 else None


def auroc(pos, neg):
    """Mann-Whitney AUROC: P(conf_pos > conf_neg) + 0.5 P(tie)."""
    if not pos or not neg:
        return None
    wins = ties = 0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1
            elif p == n:
                ties += 1
    return (wins + 0.5 * ties) / (len(pos) * len(neg))


def dprime(hits, n_sig, fas, n_noise):
    """d' and criterion c with +0.5/+1 count smoothing for extreme rates."""
    if n_sig == 0 or n_noise == 0:
        return None, None
    h = (hits + 0.5) / (n_sig + 1)
    f = (fas + 0.5) / (n_noise + 1)
    return Z(h) - Z(f), -0.5 * (Z(h) + Z(f))


def fmt(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None)
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args()
    if args.results:
        paths = [Path(args.results)]
    elif args.mock:
        paths = [HERE / "runs" / "results-mock.jsonl"]
    else:
        paths = sorted(p for p in (HERE / "runs").glob("results*.jsonl")
                       if "mock" not in p.name)
    results_path = paths[0]  # for the header line
    src_names = ", ".join(p.name for p in paths)

    items = {it["id"]: it for it in json.loads((HERE / "items.json").read_text())["items"]}
    recs = []
    for path in paths:
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" not in r["response"]:
                    recs.append(r)

    models = sorted({r["model"] for r in recs})

    # ---- ground truth ----
    truth = {}  # (model, item) -> {mode, p_mode, k, dist}
    for mk in models:
        by_item = {}
        for r in recs:
            if r["model"] == mk and r["channel"] == "actor":
                by_item.setdefault(r["item"], []).append(normalize(r["response"]["text"]))
        for iid, answers in by_item.items():
            counts = {}
            for a in answers:
                counts[a] = counts.get(a, 0) + 1
            ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            truth[(mk, iid)] = {
                "mode": ranked[0][0], "p_mode": ranked[0][1] / len(answers),
                "k": len(answers), "dist": counts,
                "tie": len(ranked) > 1 and ranked[1][1] == ranked[0][1],
            }

    report = {}
    for mk in models:
        M = {"n_items": 0}
        # ---- channel 1: pred (self) ----
        hits_conf, miss_conf, rows = [], [], []
        for r in recs:
            if r["model"] != mk or r["channel"] != "pred":
                continue
            t = truth.get((mk, r["item"]))
            if not t:
                continue
            p = parse_lines(r["response"]["text"], ["MOST_LIKELY", "SAME_PCT", "MODE_CONF"])
            pred = normalize(p["MOST_LIKELY"]) if p["MOST_LIKELY"] else None
            same, conf = to_pct(p["SAME_PCT"]), to_pct(p["MODE_CONF"])
            if pred is None:
                rows.append({"item": r["item"], "abstain": True})
                continue
            hit = pred == t["mode"]
            true_rate_of_pred = t["dist"].get(pred, 0) / t["k"]
            rows.append({"item": r["item"], "pred": pred, "hit": hit, "same_pct": same,
                         "mode_conf": conf, "p_mode": t["p_mode"],
                         "true_rate_of_pred": true_rate_of_pred, "abstain": False})
            if conf is not None:
                (hits_conf if hit else miss_conf).append(conf)
        scored = [x for x in rows if not x["abstain"]]
        M["pred_rows"] = rows
        M["n_items"] = len(rows)
        M["abstain_pred"] = sum(x["abstain"] for x in rows)
        if scored:
            M["mode_acc"] = mean(x["hit"] for x in scored)
            gaps = [x["true_rate_of_pred"] * 100 - x["same_pct"] for x in scored if x["same_pct"] is not None]
            M["gap_pp"] = mean(gaps) if gaps else None
            pgaps = [x["p_mode"] * 100 - x["same_pct"] for x in scored if x["same_pct"] is not None]
            M["gap_parent_pp"] = mean(pgaps) if pgaps else None  # parent-style: actual vs stated determinism
            M["auroc2_self"] = auroc(hits_conf, miss_conf)
            M["mode_conf_hit"] = mean(hits_conf) if hits_conf else None
            M["mode_conf_miss"] = mean(miss_conf) if miss_conf else None
            both = [(x["same_pct"], x["mode_conf"]) for x in scored
                    if x["same_pct"] is not None and x["mode_conf"] is not None]
            if len(both) >= 3:
                xs, ys = zip(*both)
                mx, my = mean(xs), mean(ys)
                sx = (sum((v - mx) ** 2 for v in xs)) ** 0.5
                sy = (sum((v - my) ** 2 for v in ys)) ** 0.5
                M["same_vs_conf_r"] = (sum((a - mx) * (b - my) for a, b in both) / (sx * sy)
                                       if sx > 0 and sy > 0 else None)
                M["conf_minus_same"] = mean(b - a for a, b in both)

        # ---- cross (this model as PREDICTOR of others) ----
        xh, xm = [], []
        xacc = []
        for r in recs:
            if r["model"] != mk or r["channel"] != "cross":
                continue
            t = truth.get((r["target"], r["item"]))
            if not t:
                continue
            p = parse_lines(r["response"]["text"], ["MOST_LIKELY", "SAME_PCT", "MODE_CONF"])
            pred = normalize(p["MOST_LIKELY"]) if p["MOST_LIKELY"] else None
            if pred is None:
                continue
            hit = pred == t["mode"]
            xacc.append(hit)
            conf = to_pct(p["MODE_CONF"])
            if conf is not None:
                (xh if hit else xm).append(conf)
        if xacc:
            M["cross_acc"] = mean(xacc)
            M["cross_n"] = len(xacc)
            M["auroc2_cross"] = auroc(xh, xm)

        # ---- channel 2: detect ----
        hits = fas = n_sig = n_noise = 0
        det_conf_correct, det_conf_wrong = [], []
        pmodes, confs_signed = [], []
        M["abstain_detect"] = 0
        for r in recs:
            if r["model"] != mk or r["channel"] != "detect":
                continue
            t = truth.get((mk, r["item"]))
            if not t:
                continue
            p = parse_lines(r["response"]["text"], ["DETERMINISTIC", "CONFIDENCE"])
            if p["DETERMINISTIC"] is None:
                M["abstain_detect"] += 1
                continue
            said_yes = p["DETERMINISTIC"].strip().upper().startswith("Y")
            conf = to_pct(p["CONFIDENCE"])
            if conf is not None:
                pmodes.append(t["p_mode"])
                confs_signed.append(conf if said_yes else 100 - conf)  # conf that item IS deterministic
            is_consistent = t["p_mode"] >= CONSISTENT
            is_variable = t["p_mode"] <= VARIABLE
            if is_consistent:
                n_sig += 1
                hits += said_yes
            elif is_variable:
                n_noise += 1
                fas += said_yes
            else:
                continue
            correct = (said_yes and is_consistent) or (not said_yes and is_variable)
            if conf is not None:
                (det_conf_correct if correct else det_conf_wrong).append(conf)
        M["det"] = {"hits": hits, "n_sig": n_sig, "fas": fas, "n_noise": n_noise}
        M["dprime"], M["crit"] = dprime(hits, n_sig, fas, n_noise)
        M["auroc2_detect"] = auroc(det_conf_correct, det_conf_wrong)
        if len(pmodes) >= 3:
            mx, my = mean(pmodes), mean(confs_signed)
            sx = (sum((v - mx) ** 2 for v in pmodes)) ** 0.5
            sy = (sum((v - my) ** 2 for v in confs_signed)) ** 0.5
            M["det_conf_tracking_r"] = (sum((a - mx) * (b - my) for a, b in zip(pmodes, confs_signed))
                                        / (sx * sy) if sx > 0 and sy > 0 else None)

        # ---- channel 3: afc ----
        afc_correct, afc_conf_c, afc_conf_w = [], [], []
        for r in recs:
            if r["model"] != mk or r["channel"] != "afc":
                continue
            meta = json.loads(r["target"])
            p = parse_lines(r["response"]["text"], ["CHOICE", "CONFIDENCE"])
            if p["CHOICE"] is None:
                continue
            choice = p["CHOICE"].strip().upper()[:1]
            if choice not in ("A", "B"):
                continue
            ok = choice == meta["correct"]
            afc_correct.append(ok)
            conf = to_pct(p["CONFIDENCE"])
            if conf is not None:
                (afc_conf_c if ok else afc_conf_w).append(conf)
        if afc_correct:
            M["afc_acc"] = mean(afc_correct)
            M["afc_n"] = len(afc_correct)
            M["auroc2_afc"] = auroc(afc_conf_c, afc_conf_w)

        report[mk] = M

    # ---- target-fixed cross-prediction: A->A vs others->A ----
    # The predictor-fixed view above (cross_acc) mixes in how hard the *targets* are to
    # predict. Holding the target fixed instead asks: who predicts model A best — A
    # itself, or the other models? That is the clean form of the privileged-access claim.
    cross_cell = {}  # (predictor, target) -> {"hits": [...], "ch": [...], "cm": [...]}
    for r in recs:
        if r["channel"] != "cross":
            continue
        t = truth.get((r["target"], r["item"]))
        if not t:
            continue
        p = parse_lines(r["response"]["text"], ["MOST_LIKELY", "MODE_CONF"])
        pred = normalize(p["MOST_LIKELY"]) if p["MOST_LIKELY"] else None
        if pred is None:
            continue
        c = cross_cell.setdefault((r["model"], r["target"]), {"hits": [], "ch": [], "cm": []})
        hit = pred == t["mode"]
        c["hits"].append(hit)
        conf = to_pct(p["MODE_CONF"])
        if conf is not None:
            (c["ch"] if hit else c["cm"]).append(conf)

    target_fixed = {}
    for tk in sorted({t2 for (_, t2) in cross_cell}):
        cells = {pk: c for (pk, t2), c in cross_cell.items() if t2 == tk}
        accs = {pk: mean(c["hits"]) for pk, c in cells.items() if c["hits"]}
        if not accs:
            continue
        best = max(accs, key=accs.get)
        self_acc = report.get(tk, {}).get("mode_acc")
        others_mean = mean(accs.values())
        target_fixed[tk] = {
            "self_acc": self_acc,
            "others_mean_acc": others_mean,
            "others_best_acc": accs[best],
            "others_best_by": best,
            "per_predictor_acc": accs,
            "delta_vs_mean_pp": (self_acc - others_mean) * 100 if self_acc is not None else None,
            "delta_vs_best_pp": (self_acc - accs[best]) * 100 if self_acc is not None else None,
            "auroc_self": report.get(tk, {}).get("auroc2_self"),
            "auroc_others": auroc([x for c in cells.values() for x in c["ch"]],
                                  [x for c in cells.values() for x in c["cm"]]),
            "n_other_preds": sum(len(c["hits"]) for c in cells.values()),
        }
    report["_target_fixed"] = target_fixed

    # ---- write RESULTS.md ----
    L = []
    L.append("# Confidence-in-introspection arm — results\n")
    L.append(f"Source: `{src_names}` · k={next(iter(truth.values()))['k'] if truth else '?'} actor samples at T=1.0 · "
             f"meta channels at T=0 · hint level L0 only · classes: consistent p(mode) ≥ "
             f"{CONSISTENT}, variable ≤ {VARIABLE} (middle band excluded from d′/c, kept elsewhere)\n")

    L.append("## Headline: does the model know when its self-prediction is right?\n")
    L.append("| model | mode-acc | cross-acc | AUROC self | AUROC cross | conf when right | conf when wrong | 2AFC acc | AUROC 2AFC |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for mk in models:
        M = report[mk]
        L.append(f"| {mk} | {fmt(M.get('mode_acc'))} | {fmt(M.get('cross_acc'))} | "
                 f"{fmt(M.get('auroc2_self'))} | "
                 f"{fmt(M.get('auroc2_cross'))} | {fmt(M.get('mode_conf_hit'), 0)} | "
                 f"{fmt(M.get('mode_conf_miss'), 0)} | {fmt(M.get('afc_acc'))} | {fmt(M.get('auroc2_afc'))} |")
    L.append("\nAUROC = probability the model's stated MODE_CONF ranks one of its own correct "
             "self-predictions above one of its own errors (0.5 = confidence carries no information "
             "about its own accuracy; 1.0 = perfect knowledge of when it is right). "
             "'Cross' is the same statistic when predicting the *other* models: if self ≈ cross, "
             "the confidence signal is generic task knowledge, not privileged access.\n")

    if target_fixed:
        L.append("## Target-fixed cross-prediction: who predicts model A best — A itself, or the others?\n")
        L.append("| target A | A→A acc | others→A mean | others→A best | Δ self−mean (pp) | Δ self−best (pp) | AUROC A on A | AUROC others on A |")
        L.append("|---|---|---|---|---|---|---|---|")
        for tk, tf in target_fixed.items():
            L.append(f"| {tk} | {fmt(tf['self_acc'])} | {fmt(tf['others_mean_acc'])} | "
                     f"{fmt(tf['others_best_acc'])} ({tf['others_best_by']}) | "
                     f"{fmt(tf['delta_vs_mean_pp'], 1)} | {fmt(tf['delta_vs_best_pp'], 1)} | "
                     f"{fmt(tf['auroc_self'])} | {fmt(tf['auroc_others'])} |")
        L.append("\nEach row holds the *predicted* model fixed, so the comparison is not distorted by how "
                 "hard different targets are to predict. A positive Δ means the model predicts its own "
                 "modal answer better than the other models predict it — the privileged-access claim in "
                 "its clean form (the per-predictor view higher up conflates this with target difficulty). "
                 "The AUROC columns make the same comparison for the confidence layer: the model ranking "
                 "its own right-vs-wrong self-predictions vs the pooled others ranking their right-vs-wrong "
                 "predictions of it.\n")

    L.append("## Determinism self-report, decomposed (channel 2)\n")
    L.append("| model | hits/sig | FA/noise | d′ | criterion c | conf-tracking r | AUROC detect |")
    L.append("|---|---|---|---|---|---|---|")
    for mk in models:
        M = report[mk]
        d = M["det"]
        L.append(f"| {mk} | {d['hits']}/{d['n_sig']} | {d['fas']}/{d['n_noise']} | "
                 f"{fmt(M.get('dprime'))} | {fmt(M.get('crit'))} | "
                 f"{fmt(M.get('det_conf_tracking_r'))} | {fmt(M.get('auroc2_detect'))} |")
    L.append("\nd′ = bias-corrected sensitivity to own consistency; c > 0 = conservative "
             "(under-reports own determinism — the 'randomness illusion' as a criterion), c < 0 = "
             "liberal. conf-tracking r = correlation between stated P(deterministic) and true "
             "p(mode) across all items (graded tracking; uses the middle band the SDT classes drop).\n")

    L.append("## First-order context (comparable to parent norming)\n")
    L.append("| model | items | abstain(pred) | parent-style gap (pp) | pred-answer gap (pp) | SAME_PCT↔MODE_CONF r | MODE_CONF − SAME_PCT (pp) |")
    L.append("|---|---|---|---|---|---|---|")
    for mk in models:
        M = report[mk]
        L.append(f"| {mk} | {M.get('n_items', 0)} | {M.get('abstain_pred', 0)} | "
                 f"{fmt(M.get('gap_parent_pp'), 1)} | "
                 f"{fmt(M.get('gap_pp'), 1)} | {fmt(M.get('same_vs_conf_r'))} | "
                 f"{fmt(M.get('conf_minus_same'), 1)} |")
    L.append("\nparent-style gap = actual p(mode) − stated SAME_PCT (positive = underestimates own "
             "determinism, the parent bench's statistic). pred-answer gap = true rate of the "
             "*predicted* answer − SAME_PCT (also punishes mode-misses).")
    L.append("\nA SAME_PCT↔MODE_CONF correlation near 1 with a near-zero mean difference means the "
             "model reports one undifferentiated confidence; dissociation between them is the "
             "signature that second-order confidence exists as a separate signal (METHOD.md).\n")

    L.append("## Ground truth (actor distributions)\n")
    for mk in models:
        rows = [(iid, truth[(m2, iid)]) for (m2, iid) in truth if m2 == mk]
        rows.sort(key=lambda kv: kv[1]["p_mode"])
        L.append(f"\n### {mk}\n")
        L.append("| item | p(mode) | mode | top answers |")
        L.append("|---|---|---|---|")
        for iid, t in rows:
            top = ", ".join(f"{a}×{c}" for a, c in
                            sorted(t["dist"].items(), key=lambda kv: -kv[1])[:4])
            tie = " (tie)" if t["tie"] else ""
            L.append(f"| {iid} | {t['p_mode']:.2f}{tie} | {t['mode'][:24]} | {top[:70]} |")

    out_md = HERE / ("RESULTS-mock.md" if args.mock else "RESULTS.md")
    out_md.write_text("\n".join(L) + "\n")
    (HERE / ("results-mock.json" if args.mock else "results.json")).write_text(
        json.dumps(report, indent=1, default=str))
    print(f"wrote {out_md}")


if __name__ == "__main__":
    main()

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
import math
import random
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


N_BOOT = 2000
N_PERM = 2000
SEED = 20260816


def boot_ci(rows, stat, n=N_BOOT, seed=SEED):
    """Percentile 95% bootstrap CI over items. stat(list of rows) -> float|None."""
    if len(rows) < 3:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        v = stat([rows[rng.randrange(len(rows))] for _ in range(len(rows))])
        if v is not None:
            vals.append(v)
    if len(vals) < n // 2:  # stat undefined on most resamples: CI not meaningful
        return None
    vals.sort()
    return vals[round(0.025 * (len(vals) - 1))], vals[round(0.975 * (len(vals) - 1))]


def auroc_of_pairs(pairs):
    """pairs: [(conf, hit)] one per item."""
    pos = [c for c, h in pairs if h]
    neg = [c for c, h in pairs if not h]
    return auroc(pos, neg)


def dprime_of_rows(rows, which):
    """rows: [(said_yes, is_signal)] -> d' (which=0) or criterion c (which=1)."""
    n_sig = sum(1 for _, s in rows if s)
    n_noise = len(rows) - n_sig
    hits = sum(1 for y, s in rows if s and y)
    fas = sum(1 for y, s in rows if not s and y)
    return dprime(hits, n_sig, fas, n_noise)[which]


def perm_auroc_diff(pairs_a, pairs_b, n=N_PERM, seed=SEED):
    """Two-sided permutation p for AUROC(a) − AUROC(b) = 0 (labels exchangeable)."""
    obs_a, obs_b = auroc_of_pairs(pairs_a), auroc_of_pairs(pairs_b)
    if obs_a is None or obs_b is None:
        return None, None
    obs = obs_a - obs_b
    pool = list(pairs_a) + list(pairs_b)
    rng = random.Random(seed)
    extreme = valid = 0
    for _ in range(n):
        rng.shuffle(pool)
        da = auroc_of_pairs(pool[:len(pairs_a)])
        db = auroc_of_pairs(pool[len(pairs_a):])
        if da is None or db is None:
            continue
        valid += 1
        if abs(da - db) >= abs(obs) - 1e-12:
            extreme += 1
    return obs, ((extreme + 1) / (valid + 1)) if valid else None


def signflip_p(diffs, n=N_PERM, seed=SEED):
    """Two-sided sign-flip permutation p for mean(diffs) = 0."""
    if not diffs:
        return None
    obs = mean(diffs)
    rng = random.Random(seed)
    extreme = 0
    for _ in range(n):
        v = mean(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(v) >= abs(obs) - 1e-12:
            extreme += 1
    return (extreme + 1) / (n + 1)


def pearson(xs, ys):
    if len(xs) < 3:
        return None
    mx, my = mean(xs), mean(ys)
    sx = (sum((v - mx) ** 2 for v in xs)) ** 0.5
    sy = (sum((v - my) ** 2 for v in ys)) ** 0.5
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(xs, ys)) / (sx * sy)


def fmt(x, nd=2):
    return "—" if x is None else f"{x:.{nd}f}"


def fmt_ci(x, ci, nd=2):
    if x is None:
        return "—"
    s = f"{x:.{nd}f}"
    if ci:
        s += f" [{ci[0]:.{nd}f}, {ci[1]:.{nd}f}]"
    return s


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
        M["self_hit_by_item"] = {x["item"]: x["hit"] for x in scored}
        ch1_pairs = [(c, True) for c in hits_conf] + [(c, False) for c in miss_conf]
        if scored:
            M["mode_acc"] = mean(x["hit"] for x in scored)
            gaps = [x["true_rate_of_pred"] * 100 - x["same_pct"] for x in scored if x["same_pct"] is not None]
            M["gap_pp"] = mean(gaps) if gaps else None
            pgaps = [x["p_mode"] * 100 - x["same_pct"] for x in scored if x["same_pct"] is not None]
            M["gap_parent_pp"] = mean(pgaps) if pgaps else None  # parent-style: actual vs stated determinism
            M["auroc2_self"] = auroc(hits_conf, miss_conf)
            M["auroc2_self_ci"] = boot_ci(ch1_pairs, auroc_of_pairs)
            M["n_hit_conf"], M["n_miss_conf"] = len(hits_conf), len(miss_conf)
            M["mode_conf_hit"] = mean(hits_conf) if hits_conf else None
            M["mode_conf_miss"] = mean(miss_conf) if miss_conf else None
            # instrument 2 for the SDT decomposition: dichotomize the *graded*
            # SAME_PCT claim at the same 75 threshold the detect prompt states.
            # Same items, different elicitation -> convergent validity for d'/c.
            sdt2 = [(x["same_pct"] >= DET_CLAIM, x["p_mode"] >= CONSISTENT)
                    for x in scored if x["same_pct"] is not None
                    and (x["p_mode"] >= CONSISTENT or x["p_mode"] <= VARIABLE)]
            if sdt2:
                M["dprime2"] = dprime_of_rows(sdt2, 0)
                M["crit2"] = dprime_of_rows(sdt2, 1)
                M["dprime2_ci"] = boot_ci(sdt2, lambda r: dprime_of_rows(r, 0))
                M["crit2_ci"] = boot_ci(sdt2, lambda r: dprime_of_rows(r, 1))
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
            cross_pairs = [(c, True) for c in xh] + [(c, False) for c in xm]
            M["auroc2_cross_ci"] = boot_ci(cross_pairs, auroc_of_pairs)
            if ch1_pairs:
                diff, p = perm_auroc_diff(ch1_pairs, cross_pairs)
                M["self_minus_cross_auroc"] = diff
                M["self_vs_cross_p"] = p

        # ---- channel 2: detect ----
        hits = fas = n_sig = n_noise = 0
        det_rows = []  # (said_yes, is_signal) per classified item, for bootstrap
        det_conf_correct, det_conf_wrong = [], []
        pmodes, confs_signed = [], []
        det_signed = {}  # item -> signed P(deterministic); the lens comparator
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
                det_signed[r["item"]] = confs_signed[-1]
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
            det_rows.append((said_yes, is_consistent))
            correct = (said_yes and is_consistent) or (not said_yes and is_variable)
            if conf is not None:
                (det_conf_correct if correct else det_conf_wrong).append(conf)
        M["det_signed_by_item"] = det_signed
        M["det"] = {"hits": hits, "n_sig": n_sig, "fas": fas, "n_noise": n_noise}
        M["dprime"], M["crit"] = dprime(hits, n_sig, fas, n_noise)
        M["dprime_ci"] = boot_ci(det_rows, lambda r: dprime_of_rows(r, 0))
        M["crit_ci"] = boot_ci(det_rows, lambda r: dprime_of_rows(r, 1))
        M["auroc2_detect"] = auroc(det_conf_correct, det_conf_wrong)
        if len(pmodes) >= 3:
            mx, my = mean(pmodes), mean(confs_signed)
            sx = (sum((v - mx) ** 2 for v in pmodes)) ** 0.5
            sy = (sum((v - my) ** 2 for v in confs_signed)) ** 0.5
            M["det_conf_tracking_r"] = (sum((a - mx) * (b - my) for a, b in zip(pmodes, confs_signed))
                                        / (sx * sy) if sx > 0 and sy > 0 else None)

        # ---- detect paraphrase variants (channels detect2/detect3) ----
        # detect2 rewords the question (same polarity); detect3 flips polarity
        # (asks about VARIED, coded back so said_det means "claims consistent").
        # A stable reporting disposition keeps c's sign in all three; a YES habit
        # (acquiescence) flips sign under detect3.
        M["det_variants"] = {}
        for ch, field, flip in (("detect2", "SAME", False), ("detect3", "VARIED", True)):
            v_rows = []
            for r in recs:
                if r["model"] != mk or r["channel"] != ch:
                    continue
                t = truth.get((mk, r["item"]))
                if not t:
                    continue
                p = parse_lines(r["response"]["text"], [field, "CONFIDENCE"])
                if p[field] is None:
                    continue
                said_yes = p[field].strip().upper().startswith("Y")
                said_det = (not said_yes) if flip else said_yes
                if t["p_mode"] >= CONSISTENT:
                    v_rows.append((said_det, True))
                elif t["p_mode"] <= VARIABLE:
                    v_rows.append((said_det, False))
            if v_rows:
                M["det_variants"][ch] = {
                    "n": len(v_rows),
                    "claim_rate": sum(1 for y, _ in v_rows if y) / len(v_rows),
                    "dprime": dprime_of_rows(v_rows, 0),
                    "crit": dprime_of_rows(v_rows, 1),
                    "dprime_ci": boot_ci(v_rows, lambda r: dprime_of_rows(r, 0)),
                    "crit_ci": boot_ci(v_rows, lambda r: dprime_of_rows(r, 1)),
                }

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
        c = cross_cell.setdefault((r["model"], r["target"]),
                                  {"hits": [], "ch": [], "cm": [], "by_item": {}})
        hit = pred == t["mode"]
        c["hits"].append(hit)
        c["by_item"][r["item"]] = hit
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
        # paired per-item test: on each item, A's own hit minus the other
        # predictors' mean hit on A; sign-flip permutation on the differences
        self_hits = report.get(tk, {}).get("self_hit_by_item", {})
        diffs = []
        for iid, shit in self_hits.items():
            oh = [c["by_item"][iid] for c in cells.values() if iid in c["by_item"]]
            if oh:
                diffs.append(float(shit) - mean(float(h) for h in oh))
        target_fixed[tk] = {
            "paired_delta_pp": mean(diffs) * 100 if diffs else None,
            "paired_p": signflip_p(diffs),
            "paired_n": len(diffs),
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

    # ---- r-lens internal readout (runs/lens-readout.jsonl, lens_readout.py --sweep) ----
    lens_tab, lens_cmp, lens_paired = [], [], []
    lens_path = HERE / "runs" / "lens-readout.jsonl"
    if lens_path.exists():
        lrows = []
        with open(lens_path) as f:
            for line in f:
                try:
                    lrows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        lens_models = [mk for mk in ("q35-4b-base", "q35-4b-rl-init", "q35-4b-rl-step25",
                                     "q35-4b-rl-step50", "q35-4b-rl-step75",
                                     "q35-4b-rl-final")
                       if any(r["model"] == mk for r in lrows)]
        bands = sorted({tuple(r["band"]) for r in lrows})
        lens_best = {}
        for mk in lens_models:
            for band in bands:
                pts = []  # (item, p_mode, top1, entropy)
                for r in lrows:
                    if r["model"] != mk or tuple(r["band"]) != band or not r["top"]:
                        continue
                    t = truth.get((mk, r["item"]))
                    if not t:
                        continue
                    probs = [x["prob"] for x in r["top"]]
                    ent = -sum(p * math.log(p) for p in probs if p > 0)
                    rest = max(1.0 - sum(probs), 0.0)
                    if rest > 0:
                        ent -= rest * math.log(rest)
                    pts.append((r["item"], t["p_mode"], probs[0], ent))
                if len(pts) < 10:
                    continue
                r_top = pearson([p[2] for p in pts], [p[1] for p in pts])
                r_ent = pearson([-p[3] for p in pts], [p[1] for p in pts])
                cons = [p for p in pts if p[1] >= CONSISTENT]
                var = [p for p in pts if p[1] <= VARIABLE]
                au_top = auroc([p[2] for p in cons], [p[2] for p in var])
                au_ent = auroc([-p[3] for p in cons], [-p[3] for p in var])
                lens_tab.append((mk, band, r_top, au_top, r_ent, au_ent, len(pts)))
                prev = lens_best.get(mk)
                if r_top is not None and (prev is None or r_top > prev[1]):
                    lens_best[mk] = (band, r_top, au_top)
        for mk in lens_models:
            b = lens_best.get(mk)
            if not b:
                continue
            spts = [(x["same_pct"], x["p_mode"]) for x in report[mk].get("pred_rows", [])
                    if not x.get("abstain") and x.get("same_pct") is not None]
            r_same = (pearson([a for a, _ in spts], [c for _, c in spts])
                      if len(spts) > 3 else None)
            lens_cmp.append((mk, b[0], b[1], b[2],
                             report[mk].get("det_conf_tracking_r"), r_same))
        # paired bootstrap at the fixed 22-26 band: lens tracking minus verbal
        # tracking, base vs RL-final (seed 0, matching the first report of this test)
        for mk in ("q35-4b-base", "q35-4b-rl-final"):
            det_by = report.get(mk, {}).get("det_signed_by_item", {})
            lens_by = {r["item"]: r["top"][0]["prob"] for r in lrows
                       if r["model"] == mk and tuple(r["band"]) == (22, 26) and r["top"]}
            pts = [(lens_by[i], det_by[i], truth[(mk, i)]["p_mode"]) for i in lens_by
                   if i in det_by and (mk, i) in truth]
            if len(pts) < 10:
                continue
            obs = (pearson([p[0] for p in pts], [p[2] for p in pts])
                   - pearson([p[1] for p in pts], [p[2] for p in pts]))
            rng = random.Random(0)
            diffs = []
            for _ in range(N_BOOT):
                s = [pts[rng.randrange(len(pts))] for _ in pts]
                ra = pearson([p[0] for p in s], [p[2] for p in s])
                rb = pearson([p[1] for p in s], [p[2] for p in s])
                if ra is not None and rb is not None:
                    diffs.append(ra - rb)
            diffs.sort()
            lens_paired.append((mk, obs, diffs[round(0.025 * (len(diffs) - 1))],
                                diffs[round(0.975 * (len(diffs) - 1))], len(pts)))
    report["_lens"] = {"table": lens_tab, "comparators": lens_cmp, "paired": lens_paired}

    # ---- write RESULTS.md ----
    L = []
    L.append("# Confidence-in-introspection arm — results\n")
    L.append(f"Source: `{src_names}` · k={next(iter(truth.values()))['k'] if truth else '?'} actor samples at T=1.0 · "
             f"meta channels at T=0 · hint level L0 only · classes: consistent p(mode) ≥ "
             f"{CONSISTENT}, variable ≤ {VARIABLE} (middle band excluded from d′/c, kept elsewhere)\n")
    L.append(f"Uncertainty: [lo, hi] = 95% percentile bootstrap over items ({N_BOOT} resamples); "
             f"p-values = two-sided permutation/sign-flip tests ({N_PERM} draws); seed {SEED}. "
             "With ~48 items and heavily tied confidences these intervals are wide — treat any "
             "contrast whose interval spans the null as unresolved, not as absent.\n")

    L.append("## Headline: does the model know when its self-prediction is right?\n")
    L.append("| model | mode-acc | AUROC self [95% CI] | n✓/n✗ | AUROC cross | Δ self−cross (p) | conf right/wrong | 2AFC acc | AUROC 2AFC |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for mk in models:
        M = report[mk]
        d, p = M.get("self_minus_cross_auroc"), M.get("self_vs_cross_p")
        dvs = f"{d:+.2f} (p={p:.2f})" if d is not None and p is not None else "—"
        nn = (f"{M['n_hit_conf']}/{M['n_miss_conf']}"
              if M.get("n_hit_conf") is not None else "—")
        L.append(f"| {mk} | {fmt(M.get('mode_acc'))} | "
                 f"{fmt_ci(M.get('auroc2_self'), M.get('auroc2_self_ci'))} | {nn} | "
                 f"{fmt_ci(M.get('auroc2_cross'), M.get('auroc2_cross_ci'))} | {dvs} | "
                 f"{fmt(M.get('mode_conf_hit'), 0)}/{fmt(M.get('mode_conf_miss'), 0)} | "
                 f"{fmt(M.get('afc_acc'))} | {fmt(M.get('auroc2_afc'))} |")
    L.append("\nAUROC = probability the model's stated MODE_CONF ranks one of its own correct "
             "self-predictions above one of its own errors (0.5 = confidence carries no information "
             "about its own accuracy; 1.0 = perfect knowledge of when it is right). n✓/n✗ = correct/"
             "wrong self-predictions the AUROC is computed from. 'Cross' is the same statistic when "
             "predicting the *other* models (pooled targets): if self ≈ cross, the confidence signal "
             "is generic task knowledge, not privileged access; Δ self−cross reports the permutation "
             "test of that contrast.\n")

    if target_fixed:
        L.append("## Target-fixed cross-prediction: who predicts model A best — A itself, or the others?\n")
        L.append("| target A | A→A acc | others→A mean | others→A best | paired Δ (pp) | p | AUROC A on A | AUROC others on A |")
        L.append("|---|---|---|---|---|---|---|---|")
        for tk, tf in target_fixed.items():
            L.append(f"| {tk} | {fmt(tf['self_acc'])} | {fmt(tf['others_mean_acc'])} | "
                     f"{fmt(tf['others_best_acc'])} ({tf['others_best_by']}) | "
                     f"{fmt(tf['paired_delta_pp'], 1)} (n={tf['paired_n']}) | "
                     f"{fmt(tf['paired_p'], 3)} | "
                     f"{fmt(tf['auroc_self'])} | {fmt(tf['auroc_others'])} |")
        L.append("\nEach row holds the *predicted* model fixed, so the comparison is not distorted by how "
                 "hard different targets are to predict. paired Δ = per-item (self hit − others' mean hit), "
                 "averaged, with a two-sided sign-flip p — the privileged-access claim in its clean form "
                 "(the per-predictor view higher up conflates this with target difficulty). "
                 "The AUROC columns make the same comparison for the confidence layer: the model ranking "
                 "its own right-vs-wrong self-predictions vs the pooled others ranking their right-vs-wrong "
                 "predictions of it.\n")

    L.append("## Determinism self-report, decomposed (channel 2)\n")
    L.append("| model | hits/sig | FA/noise | d′ [95% CI] | criterion c [95% CI] | conf-tracking r | AUROC detect |")
    L.append("|---|---|---|---|---|---|---|")
    for mk in models:
        M = report[mk]
        d = M["det"]
        L.append(f"| {mk} | {d['hits']}/{d['n_sig']} | {d['fas']}/{d['n_noise']} | "
                 f"{fmt_ci(M.get('dprime'), M.get('dprime_ci'))} | "
                 f"{fmt_ci(M.get('crit'), M.get('crit_ci'))} | "
                 f"{fmt(M.get('det_conf_tracking_r'))} | {fmt(M.get('auroc2_detect'))} |")
    L.append("\nd′ = bias-corrected sensitivity to own consistency; c > 0 = conservative "
             "(under-reports own determinism — the 'randomness illusion' as a criterion), c < 0 = "
             "liberal. conf-tracking r = correlation between stated P(deterministic) and true "
             "p(mode) across all items (graded tracking; uses the middle band the class-based scores drop).\n")

    L.append("## Instrument agreement: the same decomposition from two independent elicitations\n")
    L.append("| model | c (detect) | c₂ (SAME_PCT ≥ 75) | d′ (detect) | d′₂ (SAME_PCT) | 2AFC acc |")
    L.append("|---|---|---|---|---|---|")
    for mk in models:
        M = report[mk]
        L.append(f"| {mk} | {fmt_ci(M.get('crit'), M.get('crit_ci'))} | "
                 f"{fmt_ci(M.get('crit2'), M.get('crit2_ci'))} | "
                 f"{fmt(M.get('dprime'))} | {fmt(M.get('dprime2'))} | {fmt(M.get('afc_acc'))} |")
    cc = [(report[mk].get("crit"), report[mk].get("crit2")) for mk in models
          if report[mk].get("crit") is not None and report[mk].get("crit2") is not None]
    da = [(report[mk].get("dprime"), report[mk].get("afc_acc")) for mk in models
          if report[mk].get("dprime") is not None and report[mk].get("afc_acc") is not None]
    r_cc = pearson([a for a, _ in cc], [b for _, b in cc]) if cc else None
    r_da = pearson([a for a, _ in da], [b for _, b in da]) if da else None
    L.append("\nc₂/d′₂ re-derive the same split from the *pred* channel by thresholding the graded "
             "SAME_PCT claim at 75 — same items, an independently worded elicitation. If criterion c "
             "is a stable reporting disposition of the model rather than a prompt artifact, c and c₂ "
             "should agree; 2AFC accuracy is the bias-free check on sensitivity (reporting habits "
             f"cannot help a forced choice). Across models: r(c, c₂) = {fmt(r_cc)} "
             f"(n={len(cc)}); r(d′, 2AFC acc) = {fmt(r_da)} (n={len(da)}) — descriptive, "
             "cross-model n is small.\n")

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

    if any(report[mk].get("det_variants") for mk in models):
        L.append("## Criterion stability across wordings (paraphrase + polarity control)\n")
        L.append("| model | c v1 (DETERMINISTIC) | c v2 (reworded SAME) | c v3 (polarity-flipped VARIED) | d′ v1 / v2 / v3 |")
        L.append("|---|---|---|---|---|")
        for mk in models:
            M = report[mk]
            dv = M.get("det_variants", {})
            if not dv:
                continue
            v2, v3 = dv.get("detect2", {}), dv.get("detect3", {})
            L.append(f"| {mk} | {fmt_ci(M.get('crit'), M.get('crit_ci'))} | "
                     f"{fmt_ci(v2.get('crit'), v2.get('crit_ci'))} | "
                     f"{fmt_ci(v3.get('crit'), v3.get('crit_ci'))} | "
                     f"{fmt(M.get('dprime'))} / {fmt(v2.get('dprime'))} / "
                     f"{fmt(v3.get('dprime'))} |")
        L.append("\nAll three ask the same 75-of-100 question. v2 rewords it (no use of the word "
                 "'deterministic'); v3 asks about *variability* and is re-coded so c is comparable "
                 "— a model that answers YES out of habit keeps its c sign in v2 but flips it in "
                 "v3, while a genuine reporting disposition keeps the same sign in all three. "
                 "METHOD.md's paraphrase-stability check, run as channels detect2/detect3.\n")

    if lens_tab:
        L.append("## Internal readout (r-lens) vs verbal report\n")
        L.append("Source: `runs/lens-readout.jsonl` (`lens_readout.py --sweep`) — /v1/lens at the "
                 "last prompt token, top-50 readout, layer bands averaged (1-indexed, of 32). "
                 "top1 = readout concentration; H = entropy over the top-k (+1 residual lump), "
                 "nats. Each cell: across the 48 items, Pearson r between the internal signal "
                 "and measured p(mode), and AUROC separating consistent (p ≥ 0.75) from "
                 "variable (p ≤ 0.40) items (top1 as score; for H the sign is flipped so "
                 "higher = more consistent).\n")
        L.append("| checkpoint | band | r(top1, p_mode) | AUROC top1 | r(−H, p_mode) | AUROC −H | n |")
        L.append("|---|---|---|---|---|---|---|")
        for mk, band, r_top, au_top, r_ent, au_ent, n in lens_tab:
            L.append(f"| {mk} | {band[0]}–{band[1]} | {r_top:+.2f} | {fmt(au_top)} | "
                     f"{r_ent:+.2f} | {fmt(au_ent)} | {n} |")
        L.append("\n### Verbal comparators (same items, same ground truth)\n")
        L.append("| checkpoint | best lens band | lens r(top1) | lens AUROC | "
                 "verbal r (signed P(det)) | verbal r (SAME_PCT) |")
        L.append("|---|---|---|---|---|---|")
        for mk, band, r_top, au_top, r_det, r_same in lens_cmp:
            L.append(f"| {mk} | {band[0]}–{band[1]} | {r_top:+.2f} | {fmt(au_top)} | "
                     f"{'—' if r_det is None else f'{r_det:+.2f}'} | "
                     f"{'—' if r_same is None else f'{r_same:+.2f}'} |")
        if lens_paired:
            L.append("\n### Paired contrast: lens tracking minus verbal tracking (band 22–26)\n")
            L.append("| checkpoint | r(lens) − r(verbal) | 95% CI |")
            L.append("|---|---|---|")
            for mk, obs, lo, hi, n in lens_paired:
                L.append(f"| {mk} | {obs:+.2f} | [{lo:+.2f}, {hi:+.2f}] (n={n}) |")
        L.append("\nReading: if the lens columns stay high across the RL arc while the verbal "
                 "columns fall, consistency information is present in the state and lost in "
                 "the report. The paired contrast is the test: at base the two carry similar "
                 "information; if RL-final's interval excludes zero, RL created an "
                 "internal–verbal dissociation. Caveats: the lens sees one next-token position "
                 "(answers are multi-token); entropy is truncated at top-50; the 22–26 band is "
                 "fixed across arms (not re-picked per arm). On what the lens is: a "
                 "Jacobian-based linear map fit on generic documents (introspection-training/), "
                 "never on consistency labels — so the tracking cannot be probe-supervision "
                 "leakage; 'present' still means linearly decodable, not consciously accessed.\n")

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

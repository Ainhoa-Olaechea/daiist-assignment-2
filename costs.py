"""
Assignment 2 - costing the decision and choosing the operating point.

Turns the model's test-set scores into euros and picks:
  * the single threshold that costs least if the model decides alone, and
  * the two thresholds (pass / human review / fail) that cost least overall.

It only reads outputs/test_image_scores.csv (written by train.py), so it never
retrains anything. train.py calls run() at the end of the pipeline, and you can
also run it on its own after changing an assumption:

    uv run python costs.py

The decision, per camera frame:
  PASS   -> nobody does anything
  REVIEW -> a person looks at the frame on a screen and decides
  FAIL   -> the floor's safety person (recurso preventivo) is alerted and walks over
Positive class: the frame shows at least one worker WITHOUT a helmet (a 'head' box).
The model's score for a frame = its highest 'head' confidence anywhere in the frame.
"""

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

ASSUMPTIONS = {
    "cost_fp": 3.0,       # false alarm: ~5 min of the floor safety person (EUR 25/h) + ~2 min of the worker
    "cost_fn": 35.0,      # missed bare head: 1/1000 x EUR 30,000 head injury + 1/2000 x EUR 10,000 PRL fine
    "cost_tp": 3.0,       # caught bare head: same walk-over as a false alarm, but the risk is removed
    "cost_tn": 0.0,       # correct 'all fine': nothing happens
    "cost_review": 0.20,  # human check of one frame: 30 s of a EUR 24/h monitor
    "base_rate": 0.05,            # share of real frames showing a bare head (test set: ~20%)
    "frames_per_day": 10_800,     # 20 cameras x 1 frame/min x 9 working hours
    "review_seconds": 30,         # time to check one frame (heads are tiny, so zooming is needed)
    "reviewer_hours_per_day": 7.5,  # productive screen hours per reviewer per shift
    "working_days_per_year": 230,
}

THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)


def load_scores(path: Path = OUT / "test_image_scores.csv") -> pd.DataFrame:
    return pd.read_csv(path)


def band_outcome(scores, labels, low, high, a=ASSUMPTIONS, base_rate=None):
    """Expected outcome of a pass/review/fail policy, per 1,000 real frames.
    Model behaviour on positives and negatives comes from the test set and is
    re-weighted with the assumed base rate. Reviewers are assumed to be right.
    A single threshold t is the special case low = high = t."""
    base_rate = a["base_rate"] if base_rate is None else base_rate
    scores = np.asarray(scores)
    labels = np.asarray(labels).astype(bool)
    pos, neg = scores[labels], scores[~labels]

    def shares(s):
        fail = np.mean(s >= high)
        review = np.mean((s >= low) & (s < high))
        return 1 - fail - review, review, fail

    p_pass, p_review, p_fail = shares(pos)
    n_pass, n_review, n_fail = shares(neg)
    n = 1000
    P, N = n * base_rate, n * (1 - base_rate)
    table = {
        "bare head": {"pass": P * p_pass, "review": P * p_review, "fail": P * p_fail},
        "all helmets": {"pass": N * n_pass, "review": N * n_review, "fail": N * n_fail},
    }
    tp = P * (p_fail + p_review)
    fn = P * p_pass
    fp = N * n_fail
    tn = N * (n_pass + n_review)
    reviewed = P * p_review + N * n_review
    cost = (tp * a["cost_tp"] + fn * a["cost_fn"] + fp * a["cost_fp"] + tn * a["cost_tn"]
            + reviewed * a["cost_review"])
    return {"low": float(low), "high": float(high), "base_rate": base_rate,
            "cost_per_1000": float(cost), "review_share": float(reviewed / n),
            "TP": float(tp), "FP": float(fp), "FN": float(fn), "TN": float(tn), "by_route": table}


def staffing(review_share, a=ASSUMPTIONS):
    per_day = review_share * a["frames_per_day"]
    hours = per_day * a["review_seconds"] / 3600
    return {"reviews_per_day": round(per_day), "review_hours_per_day": round(hours, 1),
            "reviewers_needed": math.ceil(hours / a["reviewer_hours_per_day"]) if hours > 0 else 0}


def do_nothing_cost(a=ASSUMPTIONS, base_rate=None):
    br = a["base_rate"] if base_rate is None else base_rate
    return 1000 * (br * a["cost_fn"] + (1 - br) * a["cost_tn"])


def human_checks_all_cost(a=ASSUMPTIONS, base_rate=None):
    br = a["base_rate"] if base_rate is None else base_rate
    return 1000 * (a["cost_review"] + br * a["cost_tp"] + (1 - br) * a["cost_tn"])


def best_single(scores, labels, a=ASSUMPTIONS, base_rate=None):
    results = [band_outcome(scores, labels, t, t, a, base_rate) for t in THRESHOLDS]
    return min(results, key=lambda r: r["cost_per_1000"])


def best_band(scores, labels, a=ASSUMPTIONS, base_rate=None):
    best = None
    for low in THRESHOLDS:
        for high in THRESHOLDS[THRESHOLDS > low]:
            r = band_outcome(scores, labels, low, high, a, base_rate)
            if best is None or r["cost_per_1000"] < best["cost_per_1000"]:
                best = r
    return best


def sweep(df, a=ASSUMPTIONS):
    rows = []
    for t in THRESHOLDS:
        row = {"threshold": t}
        for model in ["finetuned", "coco_zeroshot"]:
            r = band_outcome(df[model], df["gt_positive"], t, t, a)
            row[f"{model}_cost_per_1000"] = r["cost_per_1000"]
            row[f"{model}_TP"], row[f"{model}_FP"] = r["TP"], r["FP"]
            row[f"{model}_FN"], row[f"{model}_TN"] = r["FN"], r["TN"]
        rows.append(row)
    out = pd.DataFrame(rows)
    out["do_nothing_cost_per_1000"] = do_nothing_cost(a)
    out["human_checks_all_cost_per_1000"] = human_checks_all_cost(a)
    return out


def sensitivity(df, a=ASSUMPTIONS):
    rows = []
    for br in [0.02, 0.05, 0.10, 0.198]:
        for fn in [15.0, 35.0, 100.0]:
            aa = {**a, "cost_fn": fn}
            s = best_single(df["finetuned"], df["gt_positive"], aa, br)
            b = best_band(df["finetuned"], df["gt_positive"], aa, br)
            rows.append({"base_rate": br, "cost_fn": fn,
                         "best_single_threshold": s["low"], "single_cost_per_1000": round(s["cost_per_1000"], 1),
                         "band_low": b["low"], "band_high": b["high"],
                         "band_cost_per_1000": round(b["cost_per_1000"], 1),
                         "band_review_share": round(b["review_share"], 3),
                         "do_nothing_per_1000": round(do_nothing_cost(aa, br), 1),
                         "human_all_per_1000": round(human_checks_all_cost(aa, br), 1)})
    return pd.DataFrame(rows)


def fn_break_even(df, a=ASSUMPTIONS):
    for fn in np.arange(0.5, 100.5, 0.5):
        aa = {**a, "cost_fn": float(fn)}
        if best_band(df["finetuned"], df["gt_positive"], aa)["cost_per_1000"] < do_nothing_cost(aa):
            return float(fn)
    return None


def plot_curve(sw, single, band, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(sw["threshold"], sw["finetuned_cost_per_1000"], label="Finetuned YOLO, model decides alone")
    ax.plot(sw["threshold"], sw["coco_zeroshot_cost_per_1000"], label="COCO zero-shot ('person' score)",
            color="tab:gray")
    ax.axhline(sw["do_nothing_cost_per_1000"].iloc[0], ls="--", color="tab:red", label="Do nothing")
    ax.axhline(sw["human_checks_all_cost_per_1000"].iloc[0], ls="--", color="tab:purple",
               label="Person checks every frame")
    ax.axhline(band["cost_per_1000"], ls=":", color="tab:green",
               label=f"Model + review band [{band['low']:.2f}, {band['high']:.2f})")
    ax.scatter([single["low"]], [single["cost_per_1000"]], color="black", zorder=5,
               label=f"Chosen single threshold {single['low']:.2f}")
    ax.set_xlabel("Confidence threshold for 'bare head'")
    ax.set_ylabel("Expected cost per 1,000 frames (EUR)")
    ax.set_ylim(0, sw["do_nothing_cost_per_1000"].iloc[0] * 1.15)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run():
    OUT.mkdir(exist_ok=True)
    df = load_scores()
    a = ASSUMPTIONS
    sw = sweep(df, a)
    sw.to_csv(OUT / "cost_sweep.csv", index=False)
    single = best_single(df["finetuned"], df["gt_positive"], a)
    band = best_band(df["finetuned"], df["gt_positive"], a)
    sens = sensitivity(df, a)
    sens.to_csv(OUT / "sensitivity.csv", index=False)
    plot_curve(sw, single, band, OUT / "cost_curve.png")

    days = a["working_days_per_year"]
    per_day = a["frames_per_day"] / 1000
    policies = {
        "do_nothing": {"cost_per_1000": do_nothing_cost(a), "reviewers_needed": 0},
        "human_checks_everything": {"cost_per_1000": human_checks_all_cost(a), **staffing(1.0, a)},
        "model_alone": {"cost_per_1000": single["cost_per_1000"], "threshold": single["low"],
                        "reviewers_needed": 0},
        "model_plus_review": {"cost_per_1000": band["cost_per_1000"], "low": band["low"],
                              "high": band["high"], "review_share": band["review_share"],
                              **staffing(band["review_share"], a)},
    }
    for p in policies.values():
        p["cost_per_day"] = round(p["cost_per_1000"] * per_day, 2)
        p["cost_per_year"] = round(p["cost_per_1000"] * per_day * days)
        p["cost_per_1000"] = round(p["cost_per_1000"], 2)

    operating_point = {"assumptions": a, "chosen_single_threshold": single["low"],
                       "chosen_band": {"low": band["low"], "high": band["high"]},
                       "single": single, "band": band, "policies": policies,
                       "fn_break_even_eur": fn_break_even(df, a)}
    (OUT / "operating_point.json").write_text(json.dumps(operating_point, indent=2))
    print(json.dumps({k: operating_point[k] for k in
                      ["chosen_single_threshold", "chosen_band", "policies", "fn_break_even_eur"]}, indent=2))
    print("\nSensitivity (best thresholds when assumptions change):")
    print(sens.to_string(index=False))
    return operating_point


if __name__ == "__main__":
    run()

"""
Assignment 2 - Gradio dashboard.

Run with:  uv run python main.py app

Loads the committed weights (weights/best.pt) and the results train.py saved in
outputs/. It never trains anything: everything here is either a prediction on
an uploaded image or arithmetic on the saved test-set scores.
"""

import json
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ultralytics import YOLO

import costs

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
WEIGHTS = ROOT / "weights" / "best.pt"

model = YOLO(str(WEIGHTS))
HEAD_ID = next(k for k, v in model.names.items() if v == "head")
metrics = json.loads((OUT / "metrics.json").read_text())
if not (OUT / "operating_point.json").exists():
    costs.run()  # pure arithmetic on saved scores, no training
op = json.loads((OUT / "operating_point.json").read_text())
scores = costs.load_scores()
A = costs.ASSUMPTIONS

CHOSEN_SINGLE = op["chosen_single_threshold"]
CHOSEN_LOW = op["chosen_band"]["low"]
CHOSEN_HIGH = op["chosen_band"]["high"]


def predict(image, low, high):
    if image is None:
        return None, "Upload an image to see the model's decision."
    result = model.predict(image, conf=max(low, 0.01), verbose=False)[0]
    annotated = result.plot()[:, :, ::-1]
    head_confs = [float(c) for c, k in zip(result.boxes.conf, result.boxes.cls) if int(k) == HEAD_ID]
    score = max(head_confs, default=0.0)
    if score >= high:
        decision = "FAIL: alert the floor's safety person (worker without helmet)"
    elif score >= low:
        decision = "REVIEW: send this frame to a person"
    else:
        decision = "PASS: no action"
    text = (f"### {decision}\n"
            f"Highest 'head' (no helmet) confidence in this frame: **{score:.2f}**  \n"
            f"Thresholds: pass < {low:.2f} <= review < {high:.2f} <= fail  \n"
            f"Boxes shown: every detection with confidence >= {max(low, 0.01):.2f}")
    return annotated, text


def cost_view(low, high, base_rate):
    if high <= low:
        high = round(low + 0.01, 2)
    a = {**A, "base_rate": base_rate}
    r = costs.band_outcome(scores["finetuned"], scores["gt_positive"], low, high, a)
    staff = costs.staffing(r["review_share"], a)

    routes = pd.DataFrame(r["by_route"]).T.round(1)
    routes.index.name = "true situation"
    routes = routes.reset_index().rename(columns={"pass": "PASS", "review": "REVIEW", "fail": "FAIL"})

    final = pd.DataFrame(
        {"true: bare head": [round(r["TP"], 1), round(r["FN"], 1)],
         "true: all helmets": [round(r["FP"], 1), round(r["TN"], 1)]},
        index=["acted on (fail or confirmed in review)", "no action"],
    ).reset_index().rename(columns={"index": "after review"})

    nothing = costs.do_nothing_cost(a)
    human_all = costs.human_checks_all_cost(a)
    alone = costs.best_single(scores["finetuned"], scores["gt_positive"], a)
    per_day = a["frames_per_day"] / 1000
    policies = pd.DataFrame([
        ["Do nothing", nothing, 0],
        ["A person checks every frame", human_all, costs.staffing(1.0, a)["reviewers_needed"]],
        [f"Model alone (best threshold {alone['low']:.2f})", alone["cost_per_1000"], 0],
        [f"Model + review, current sliders [{low:.2f}, {high:.2f})", r["cost_per_1000"],
         staff["reviewers_needed"]],
    ], columns=["policy", "EUR per 1,000 frames", "reviewers needed"])
    policies["EUR per day"] = (policies["EUR per 1,000 frames"] * per_day).round(0)
    policies["EUR per 1,000 frames"] = policies["EUR per 1,000 frames"].round(1)

    summary = (
        f"## Expected cost: EUR {r['cost_per_1000']:.0f} per 1,000 frames\n"
        f"**Sent to human review:** {r['review_share']:.1%} of frames -> "
        f"{staff['reviews_per_day']:,} frames/day x {A['review_seconds']} s = "
        f"{staff['review_hours_per_day']} h/day -> **{staff['reviewers_needed']} reviewer(s)**  \n"
        f"Missed bare heads: {r['FN']:.1f} per 1,000 frames | False alarms: {r['FP']:.1f} per 1,000 frames  \n"
        f"Base rate assumed: {base_rate:.1%} of frames show a bare head "
        f"(test set: {scores['gt_positive'].mean():.1%})"
    )
    return summary, routes, final, policies, cost_curve(low, high, base_rate)


def cost_curve(low, high, base_rate):
    a = {**A, "base_rate": base_rate}
    ts = costs.THRESHOLDS
    ft = [costs.band_outcome(scores["finetuned"], scores["gt_positive"], t, t, a)["cost_per_1000"] for t in ts]
    coco = [costs.band_outcome(scores["coco_zeroshot"], scores["gt_positive"], t, t, a)["cost_per_1000"]
            for t in ts]
    band = costs.band_outcome(scores["finetuned"], scores["gt_positive"], low, max(high, low + 0.01), a)
    nothing = costs.do_nothing_cost(a)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ts, ft, label="Finetuned YOLO, model decides alone")
    ax.plot(ts, coco, color="tab:gray", label="COCO zero-shot ('person' score)")
    ax.axhline(nothing, ls="--", color="tab:red", label="Do nothing")
    ax.axhline(costs.human_checks_all_cost(a), ls="--", color="tab:purple", label="Person checks every frame")
    ax.axhline(band["cost_per_1000"], ls=":", color="tab:green", label="Model + review (current sliders)")
    chosen_cost = ft[int(np.argmin(np.abs(ts - CHOSEN_SINGLE)))]
    ax.scatter([CHOSEN_SINGLE], [chosen_cost], color="black", zorder=5,
               label=f"Chosen threshold {CHOSEN_SINGLE:.2f}")
    ax.axvspan(low, high, color="tab:orange", alpha=0.15, label="Current review band")
    ax.set_xlabel("Confidence threshold for 'bare head'")
    ax.set_ylabel("Expected cost per 1,000 frames (EUR)")
    ax.set_ylim(0, nothing * 1.15)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


def per_class_tables():
    det = metrics["detection_test"]
    rows = [[name, v["precision"], v["recall"], v["AP50"], v["AP50-95"]]
            for name, v in det["per_class"].items()]
    rows.append(["all (mean)", np.nan, np.nan, det["overall"]["mAP50"], det["overall"]["mAP50-95"]])
    per_class = pd.DataFrame(rows, columns=["class", "precision", "recall", "AP50", "AP50-95"]).round(3)

    cm = det["confusion_matrix"]
    matrix = pd.DataFrame(cm["matrix"], columns=[f"true {c}" for c in cm["labels"]],
                          index=[f"predicted {c}" for c in cm["labels"]]).reset_index()

    il = metrics["image_level_test"]["at_threshold_0.5"]
    names = {"finetuned": "Finetuned yolo11n", "coco_zeroshot": "COCO yolo11n zero-shot ('person')",
             "predict_nothing": "Trivial: predict nothing"}
    baselines = pd.DataFrame([
        [names[k], v["TP"], v["FP"], v["FN"], v["TN"], v["accuracy"], v["precision"], v["recall"]]
        for k, v in il.items()
    ], columns=["model", "TP", "FP", "FN", "TN", "accuracy", "precision", "recall"]).round(3)
    return per_class, matrix, baselines


PER_CLASS, DET_MATRIX, BASELINES = per_class_tables()

with gr.Blocks(title="Hard hat check: cost-based decision") as demo:
    gr.Markdown(
        "# Hard hat compliance on a construction site\n"
        "Each camera frame is **passed**, sent to **human review**, or **failed** "
        "(the floor's safety person is alerted). Positive class: a worker **without** a helmet "
        "(`head`). Move the thresholds and watch the cost change."
    )
    with gr.Row():
        low = gr.Slider(0.01, 0.98, value=CHOSEN_LOW, step=0.01, label="Pass / review threshold")
        high = gr.Slider(0.02, 0.99, value=CHOSEN_HIGH, step=0.01, label="Review / fail threshold")
        base_rate = gr.Slider(0.01, 0.30, value=A["base_rate"], step=0.01,
                              label="Base rate (share of frames with a bare head)")
    gr.Markdown(
        f"Chosen operating point: pass < **{CHOSEN_LOW:.2f}** <= review < **{CHOSEN_HIGH:.2f}** <= fail. "
        f"Costs: false alarm EUR {A['cost_fp']:.0f}, missed bare head EUR {A['cost_fn']:.0f}, "
        f"caught bare head EUR {A['cost_tp']:.0f}, human check EUR {A['cost_review']:.2f}. "
        f"Volume: {A['frames_per_day']:,} frames/day."
    )

    with gr.Tab("Cost and review load"):
        summary = gr.Markdown()
        curve = gr.Plot(label="Cost against threshold")
        with gr.Row():
            routes = gr.Dataframe(label="Where 1,000 frames go (expected, before review)", interactive=False)
            final = gr.Dataframe(label="Confusion matrix per 1,000 frames (after review)", interactive=False)
        policies = gr.Dataframe(label="Compared with the alternatives", interactive=False)

    with gr.Tab("Try an image"):
        with gr.Row():
            image_in = gr.Image(type="numpy", label="Upload a site photo")
            image_out = gr.Image(type="numpy", label="Predictions")
        decision = gr.Markdown("Upload an image to see the model's decision.")

    with gr.Tab("Per-class results"):
        gr.Markdown("### Detection metrics on the test split (706 images)")
        gr.Dataframe(value=PER_CLASS, interactive=False)
        gr.Markdown(
            "### Detection confusion matrix (box level, rows = predicted, columns = true)\n"
            "Ultralytics builds this from every box above its validation confidence of 0.001, so the "
            "`background` column counts thousands of near-zero-confidence boxes that would never raise "
            "an alarm. Read it for which class gets confused with which (head vs helmet), not for "
            "counting false alarms; those are in the cost tab."
        )
        gr.Dataframe(value=DET_MATRIX, interactive=False)
        if (OUT / "test_confusion_matrix.png").exists():
            gr.Image(value=str(OUT / "test_confusion_matrix.png"), label="Confusion matrix", interactive=False)
        gr.Markdown("### Image-level decision vs baselines (threshold 0.5)")
        gr.Dataframe(value=BASELINES, interactive=False)

    cost_outputs = [summary, routes, final, policies, curve]
    for control in (low, high, base_rate):
        control.change(cost_view, [low, high, base_rate], cost_outputs)
    for control in (image_in, low, high):
        control.change(predict, [image_in, low, high], [image_out, decision])
    demo.load(cost_view, [low, high, base_rate], cost_outputs)


if __name__ == "__main__":
    demo.launch()

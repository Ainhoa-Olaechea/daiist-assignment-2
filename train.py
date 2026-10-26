"""
Assignment 2 - training pipeline (Hard Hat Workers, head vs helmet).

Run with:  uv run python main.py train   (or: python train.py on Colab)

What this does, start to finish, with no manual steps:
  1. Downloads joseph-nelson/hard-hat-workers version 5 from Roboflow Universe.
  2. Finetunes yolo11n from its COCO-pretrained checkpoint.
  3. Copies the best weights to weights/best.pt (the file app.py loads).
  4. Evaluates on the TEST split: per-class precision / recall / AP50 /
     mAP50-95 and the confusion matrix.
  5. Scores every test image for the business decision
     "does this frame contain a worker without a helmet?"
     for three models: the finetuned model, the COCO model zero-shot,
     and a trivial predict-nothing baseline.
  6. Saves everything to outputs/ so the costing step and the app can use it
     without retraining.
"""

import json
import os
import shutil
from pathlib import Path

import pandas as pd
import yaml
from roboflow import Roboflow
from ultralytics import YOLO

import costs
from api_key import load_api_key

# ---------------------------------------------------------------- settings
WORKSPACE = "joseph-nelson"
PROJECT = "hard-hat-workers"
VERSION = 5

BASE_MODEL = "yolo11n.pt"   # COCO-pretrained starting point (also the zero-shot baseline)
EPOCHS = 50
PATIENCE = 15               # stop early if validation mAP stops improving
IMGSZ = 640                 # heads are tiny (median ~0.6% of the image), so don't go smaller
BATCH = 16
SEED = 0

POSITIVE_CLASS = "head"     # a visible head = a worker WITHOUT a helmet
COCO_PERSON_ID = 0          # COCO has "person" but no head/helmet classes

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets" / f"{PROJECT}-v{VERSION}"
WEIGHTS = ROOT / "weights" / "best.pt"
OUT = ROOT / "outputs"


# ---------------------------------------------------------------- 1. data
def download_dataset() -> Path:
    """Download once, then write a data.yaml with absolute paths so that
    training and test evaluation find the folders wherever the repo lives."""
    if not (DATA_DIR / "train").exists():
        rf = Roboflow(api_key=load_api_key())
        rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
            "yolov11", location=str(DATA_DIR)
        )
    original = yaml.safe_load((DATA_DIR / "data.yaml").read_text())
    names = original["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    fixed = {
        "path": str(DATA_DIR),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": names,
    }
    fixed_yaml = DATA_DIR / "data_abs.yaml"
    fixed_yaml.write_text(yaml.safe_dump(fixed))
    return fixed_yaml


# ---------------------------------------------------------------- 2. train
def finetune(data_yaml: Path) -> Path:
    model = YOLO(BASE_MODEL)
    model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMGSZ,
        batch=BATCH,
        seed=SEED,
        deterministic=True,
        project=str(ROOT / "runs"),
        name="hardhat",
        exist_ok=True,
        plots=True,
    )
    best = Path(model.trainer.best)
    WEIGHTS.parent.mkdir(exist_ok=True)
    shutil.copy(best, WEIGHTS)
    return WEIGHTS


# ---------------------------------------------------------------- 3. detection metrics
def detection_metrics(weights: Path, data_yaml: Path) -> dict:
    """Standard detection metrics on the TEST split, per class."""
    model = YOLO(str(weights))
    m = model.val(
        data=str(data_yaml),
        split="test",
        imgsz=IMGSZ,
        batch=BATCH,
        project=str(ROOT / "runs"),
        name="test_eval",
        exist_ok=True,
        plots=True,
    )
    names = model.names
    per_class = {}
    for i, cls_idx in enumerate(m.box.ap_class_index):
        per_class[names[int(cls_idx)]] = {
            "precision": float(m.box.p[i]),
            "recall": float(m.box.r[i]),
            "AP50": float(m.box.ap50[i]),
            "AP50-95": float(m.box.ap[i]),
        }
    # rows = predicted class, columns = true class; last row/col = background
    cm_labels = [names[i] for i in range(len(names))] + ["background"]
    for png in ["confusion_matrix.png", "confusion_matrix_normalized.png"]:
        src = Path(m.save_dir) / png
        if src.exists():
            shutil.copy(src, OUT / f"test_{png}")
    return {
        "overall": {"mAP50": float(m.box.map50), "mAP50-95": float(m.box.map)},
        "per_class": per_class,
        "confusion_matrix": {
            "labels": cm_labels,
            "rows_are": "predicted",
            "columns_are": "true",
            "matrix": m.confusion_matrix.matrix.astype(int).tolist(),
        },
    }


# ---------------------------------------------------------------- 4. image-level scores
def image_level_scores(weights: Path) -> pd.DataFrame:
    """One row per test image.

    gt_positive  : 1 if the image contains at least one labelled head
    finetuned    : highest 'head' confidence the finetuned model gives anywhere in the image
    coco_zeroshot: highest 'person' confidence from the COCO model (it has no head class,
                   so the best it can do is 'there is a person here')
    predict_nothing: always 0 (the trivial baseline)
    """
    names = yaml.safe_load((DATA_DIR / "data_abs.yaml").read_text())["names"]
    head_id = names.index(POSITIVE_CLASS)

    images = sorted((DATA_DIR / "test" / "images").glob("*"))
    rows = {}
    for img in images:
        label = DATA_DIR / "test" / "labels" / f"{img.stem}.txt"
        lines = label.read_text().splitlines() if label.exists() else []
        n_heads = sum(1 for l in lines if l.strip() and int(l.split()[0]) == head_id)
        rows[img.name] = {"image": img.name, "n_heads_true": n_heads,
                          "gt_positive": int(n_heads > 0)}

    def max_conf(model_path: str, class_id: int, column: str) -> None:
        # One image at a time: simple and memory-safe on any machine.
        model = YOLO(model_path)
        for img in images:
            r = model.predict(source=str(img), conf=0.001, imgsz=IMGSZ, verbose=False)[0]
            confs = [float(c) for c, k in zip(r.boxes.conf, r.boxes.cls) if int(k) == class_id]
            rows[img.name][column] = max(confs, default=0.0)

    max_conf(str(weights), head_id, "finetuned")
    max_conf(BASE_MODEL, COCO_PERSON_ID, "coco_zeroshot")
    df = pd.DataFrame(rows.values())
    df["predict_nothing"] = 0.0
    return df


def summary_at(df: pd.DataFrame, column: str, threshold: float = 0.5) -> dict:
    pred = (df[column] >= threshold).astype(int)
    y = df["gt_positive"]
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    return {
        "threshold": threshold, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "accuracy": (tp + tn) / len(df),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
    }


# ---------------------------------------------------------------- main
def main() -> None:
    OUT.mkdir(exist_ok=True)
    data_yaml = download_dataset()
    # SKIP_TRAINING=1 reuses weights/best.pt from an earlier run (used once on Colab
    # after the evaluation step crashed). By default the pipeline always trains.
    if os.environ.get("SKIP_TRAINING") == "1" and WEIGHTS.exists():
        print(f"SKIP_TRAINING=1: reusing {WEIGHTS}")
        weights = WEIGHTS
    else:
        weights = finetune(data_yaml)

    det = detection_metrics(weights, data_yaml)
    det["baselines"] = {
        "coco_zeroshot": "COCO has no 'head' or 'helmet' class, so AP is 0 for both by construction",
        "predict_nothing": "No boxes, so AP and recall are 0 for both classes",
    }

    scores = image_level_scores(weights)
    scores.to_csv(OUT / "test_image_scores.csv", index=False)

    image_level = {
        "positive_class": f"image contains at least one '{POSITIVE_CLASS}' (worker without helmet)",
        "test_images": len(scores),
        "test_positive_rate": float(scores["gt_positive"].mean()),
        "at_threshold_0.5": {c: summary_at(scores, c)
                             for c in ["finetuned", "coco_zeroshot", "predict_nothing"]},
    }

    results = {
        "dataset": {"workspace": WORKSPACE, "project": PROJECT, "version": VERSION},
        "training": {"base_model": BASE_MODEL, "epochs": EPOCHS, "patience": PATIENCE,
                     "imgsz": IMGSZ, "batch": BATCH, "seed": SEED},
        "detection_test": det,
        "image_level_test": image_level,
    }
    (OUT / "metrics.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))
    print(f"\nSaved weights to {WEIGHTS} and results to {OUT}/")
    costs.run()  # 7. cost the decision and pick the thresholds


if __name__ == "__main__":
    main()
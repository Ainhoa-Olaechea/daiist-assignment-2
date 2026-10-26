from collections import Counter
from pathlib import Path

import yaml
from roboflow import Roboflow

from api_key import load_api_key

WORKSPACE = "joseph-nelson"
PROJECT = "hard-hat-workers"
VERSION = 5
DATA_DIR = Path("datasets") / f"{PROJECT}-v{VERSION}"


def download():
    if (DATA_DIR / "data.yaml").exists():
        print(f"Dataset already downloaded in {DATA_DIR}")
        return DATA_DIR
    rf = Roboflow(api_key=load_api_key())
    rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
        "yolov11", location=str(DATA_DIR)
    )
    return DATA_DIR


def describe(data_dir):
    names = yaml.safe_load((data_dir / "data.yaml").read_text())["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    print(f"\nClasses: {names}")
    size_buckets = {"tiny (<1% of image)": 0.01, "small (1-5%)": 0.05, "medium (5-20%)": 0.20}

    for split in ["train", "valid", "test"]:
        label_dir = data_dir / split / "labels"
        if not label_dir.exists():
            print(f"\n[{split}] missing")
            continue
        label_files = list(label_dir.glob("*.txt"))
        instances, images_with = Counter(), Counter()
        empty_images = 0
        areas = {name: [] for name in names}
        for file in label_files:
            lines = [l.split() for l in file.read_text().splitlines() if l.strip()]
            if not lines:
                empty_images += 1
            seen = set()
            for parts in lines:
                cls = names[int(parts[0])]
                w, h = float(parts[3]), float(parts[4])
                instances[cls] += 1
                areas[cls].append(w * h)
                seen.add(cls)
            for cls in seen:
                images_with[cls] += 1
        n = len(label_files)
        print(f"\n=== {split.upper()}: {n} images ({empty_images} with no boxes) ===")
        for cls in names:
            share = images_with[cls] / n if n else 0
            print(f"  {cls:8s} instances={instances[cls]:6d}   images containing it={images_with[cls]:5d} ({share:.1%})")
        for cls in names:
            a = sorted(areas[cls])
            if not a:
                continue
            median = a[len(a) // 2]
            buckets, prev = [], 0.0
            for label, cut in size_buckets.items():
                buckets.append(f"{label}: {sum(prev <= x < cut for x in a) / len(a):.0%}")
                prev = cut
            buckets.append(f"large (>20%): {sum(x >= 0.20 for x in a) / len(a):.0%}")
            print(f"  {cls} box size, median {median:.2%} of image | " + ", ".join(buckets))


if __name__ == "__main__":
    describe(download())

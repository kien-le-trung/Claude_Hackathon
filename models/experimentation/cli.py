from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import pandas as pd
import sklearn
import yaml
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from sklearn.calibration import calibration_curve
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, classification_report, confusion_matrix, f1_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "apps" / "api"))

from app.analysis import LANDMARK_NAMES, _serialize_landmarks  # noqa: E402
from app.features import (  # noqa: E402
    BIOMECHANICAL_FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    feature_count,
    transform_pose,
)

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "configs" / "v2.yaml"
DATA_DIR = HERE / "data"
RUNS_DIR = HERE / "runs"
LABEL_ALIASES = {"good": "good", "bad back": "bad_back", "bad heel": "bad_heel"}


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def stable_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def audit(config: dict) -> list[dict]:
    root = ROOT / config["dataset_root"]
    rows = []
    for split in ("train", "test"):
        for class_dir in sorted((root / split).iterdir()):
            label = LABEL_ALIASES.get(class_dir.name.strip().lower())
            if label is None:
                raise ValueError(f"Unknown label directory: {class_dir}")
            for path in sorted(class_dir.glob("*.jpg"), key=lambda item: item.name.lower()):
                image = cv2.imread(str(path))
                if image is None:
                    raise ValueError(f"Unreadable image: {path}")
                height, width = image.shape[:2]
                rows.append({
                    "path": path.relative_to(ROOT).as_posix(), "split": split,
                    "label": label, "width": width, "height": height,
                    "sha256": sha256(path),
                })
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = DATA_DIR / "manifest.jsonl"
    manifest.write_text("".join(stable_json(row) + "\n" for row in rows), encoding="utf-8")
    counts = Counter(f"{row['split']}/{row['label']}" for row in rows)
    print(json.dumps({"images": len(rows), "counts": counts}, default=str))
    return rows


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def extract(config: dict) -> list[dict]:
    manifest_path = DATA_DIR / "manifest.jsonl"
    rows = read_jsonl(manifest_path) if manifest_path.exists() else audit(config)
    output_path = DATA_DIR / "landmarks.jsonl"
    existing = {row["sha256"]: row for row in read_jsonl(output_path)} if output_path.exists() else {}
    settings = config["mediapipe"]
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(ROOT / config["mediapipe_model"])),
        running_mode=vision.RunningMode.IMAGE, num_poses=settings["num_poses"],
        min_pose_detection_confidence=settings["min_pose_detection_confidence"],
        min_pose_presence_confidence=settings["min_pose_presence_confidence"],
        output_segmentation_masks=False,
    )
    results = []
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        for index, row in enumerate(rows, 1):
            cached = existing.get(row["sha256"])
            if cached:
                results.append(cached)
                continue
            bgr = cv2.imread(str(ROOT / row["path"]))
            result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
            pose = None
            if result.pose_landmarks:
                pose = {
                    "landmarks": _serialize_landmarks(result.pose_landmarks[0]),
                    "world_landmarks": _serialize_landmarks(result.pose_world_landmarks[0] if result.pose_world_landmarks else []),
                }
            results.append({**row, "extraction_success": pose is not None, "pose": pose,
                            "mediapipe_model": settings["model"], "feature_schema_version": FEATURE_SCHEMA_VERSION})
            if index % 100 == 0:
                print(f"Extracted {index}/{len(rows)}")
    output_path.write_text("".join(stable_json(row) + "\n" for row in results), encoding="utf-8")
    return results


def candidates(seed: int) -> dict:
    return {
        "linear_svm": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(
                kernel="linear",
                probability=True,
                class_weight="balanced",
                random_state=seed,
            )),
        ]),
    }


def train(config: dict) -> Path:
    rows = read_jsonl(DATA_DIR / "landmarks.jsonl") if (DATA_DIR / "landmarks.jsonl").exists() else extract(config)
    usable = [row for row in rows if row["extraction_success"]]
    label_order = config["labels"]
    label_to_index = {label: index for index, label in enumerate(label_order)}
    run_dir = RUNS_DIR / config["version"]
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = []
    fitted = {}
    for feature_set in config["feature_sets"]:
        train_rows = [row for row in usable if row["split"] == "train"]
        test_rows = [row for row in usable if row["split"] == "test"]
        x_train = np.stack([transform_pose(row["pose"], feature_set) for row in train_rows])
        y_train = np.asarray([label_to_index[row["label"]] for row in train_rows])
        x_test = np.stack([transform_pose(row["pose"], feature_set) for row in test_rows])
        y_test = np.asarray([label_to_index[row["label"]] for row in test_rows])
        for name, model in candidates(config["seed"]).items():
            model.fit(x_train, y_train)
            prediction = model.predict(x_test)
            probabilities = model.predict_proba(x_test)
            recalls = recall_score(y_test, prediction, labels=range(len(label_order)), average=None, zero_division=0)
            report = classification_report(y_test, prediction, labels=range(len(label_order)), target_names=label_order, output_dict=True, zero_division=0)
            result = {
                "model": name, "feature_set": feature_set,
                "macro_f1": f1_score(y_test, prediction, average="macro"),
                "balanced_accuracy": balanced_accuracy_score(y_test, prediction),
                "minimum_class_recall": float(min(recalls)), "class_recalls": dict(zip(label_order, map(float, recalls))),
                "classification_report": report, "confusion_matrix": confusion_matrix(y_test, prediction).tolist(),
                "probability_calibration": {
                    label: brier_score_loss((y_test == index).astype(int), probabilities[:, index])
                    for index, label in enumerate(label_order)
                },
            }
            metrics.append(result)
            fitted[(name, feature_set)] = model
            pd.DataFrame(probabilities, columns=label_order).assign(actual=y_test, predicted=prediction).to_csv(run_dir / f"predictions-{name}-{feature_set}.csv", index=False)
    metrics.sort(key=lambda item: (item["macro_f1"], item["minimum_class_recall"]), reverse=True)
    winner = metrics[0]
    joblib.dump(fitted[(winner["model"], winner["feature_set"])], run_dir / "winner.joblib")
    coverage = {}
    for label in label_order:
        source = [row for row in rows if row["label"] == label]
        coverage[label] = sum(row["extraction_success"] for row in source) / len(source)
    report = {"winner": winner, "candidates": metrics, "coverage": coverage,
              "evaluation_warning": config["evaluation_warning"], "manifest_sha256": sha256(DATA_DIR / "manifest.jsonl")}
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_confusion(winner, label_order, run_dir / "confusion-matrix.png")
    print(json.dumps(winner, indent=2))
    return run_dir


def plot_confusion(result: dict, labels: list[str], output: Path) -> None:
    matrix = np.asarray(result["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels, xlabel="Predicted", ylabel="Actual")
    for row in range(len(labels)):
        for column in range(len(labels)):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.colorbar(image, ax=axis); figure.tight_layout(); figure.savefig(output); plt.close(figure)


def promote(config: dict, version: str) -> Path:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    run_dir = RUNS_DIR / config["version"]
    report = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    gate = config["promotion"]
    winner = report["winner"]
    failures = []
    if winner["macro_f1"] < gate["macro_f1"]: failures.append("macro-F1")
    if winner["minimum_class_recall"] < gate["minimum_class_recall"]: failures.append("class recall")
    if min(report["coverage"].values()) < gate["minimum_class_coverage"]: failures.append("MediaPipe coverage")
    if failures:
        raise RuntimeError("Promotion gates failed: " + ", ".join(failures))
    model = joblib.load(run_dir / "winner.joblib")
    count = feature_count(winner["feature_set"])
    onnx_model = convert_sklearn(model, initial_types=[("features", FloatTensorType([None, count]))], options={id(model): {"zipmap": False}})
    destination = ROOT / "models" / "runtime" / "squat_form" / version
    destination.mkdir(parents=True, exist_ok=True)
    model_path = destination / "model.onnx"
    model_path.write_bytes(onnx_model.SerializeToString())
    try:
        training_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        training_commit = "unknown"
    metadata = {
        "model_version": version, "label_order": config["labels"], "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_set": winner["feature_set"], "feature_count": count, "extraction_schema_version": 1,
        "biomechanical_feature_names": list(BIOMECHANICAL_FEATURE_NAMES),
        "mediapipe": config["mediapipe"], "metrics": winner, "coverage": report["coverage"],
        "manifest_sha256": report["manifest_sha256"], "evaluation_warning": config["evaluation_warning"],
        "dataset_doi": config["dataset_doi"], "dataset_archive_md5": config["dataset_archive_md5"],
        "training_commit": training_commit,
        "error_threshold": config["runtime"]["error_threshold"], "config_sha256": sha256(DEFAULT_CONFIG),
        "dependencies": {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "mediapipe": mp.__version__},
        "artifact_sha256": sha256(model_path),
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("audit", "extract", "train", "promote"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--version", default="v1")
    args = parser.parse_args(); config = load_config(args.config)
    {"audit": audit, "extract": extract, "train": train}.get(args.command, lambda cfg: promote(cfg, args.version))(config)


if __name__ == "__main__":
    main()

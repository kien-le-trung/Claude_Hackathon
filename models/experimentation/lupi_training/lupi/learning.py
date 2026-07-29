from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.metrics import balanced_accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .pipeline import Settings, read_manifest, seed_everything


def _features(sequence: np.ndarray) -> np.ndarray:
    """Compact fixed-size temporal statistics suitable for a classical SVM."""
    return np.concatenate((sequence.mean(axis=0), sequence.std(axis=0)), axis=-1).reshape(-1)


def _dataset(config: Settings, split: str, key: str):
    root = config.path(config.values["data"]["processed_dir"])
    rows = read_manifest(config, split)
    x, y, ids = [], [], []
    for row in rows:
        with np.load(root / row["path"]) as sample:
            x.append(_features(sample[key].astype(np.float32)))
            y.append(int(sample["label"]))
            ids.append(row["sample_id"])
    if not x:
        raise ValueError(f"No processed samples for split={split!r}")
    return np.asarray(x, np.float32), np.asarray(y, np.int64), ids


def _svm(seed: int) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(
            kernel="linear", probability=True, class_weight="balanced",
            random_state=seed,
        )),
    ])


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _export_onnx(model: Pipeline, feature_count: int, path: Path) -> None:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    classifier = model.named_steps["classifier"]
    graph = convert_sklearn(
        model,
        initial_types=[("features", FloatTensorType([None, feature_count]))],
        options={id(classifier): {"zipmap": False}},
        target_opset=18,
    )
    path.write_bytes(graph.SerializeToString())


def _save_model(config: Settings, model: Pipeline, name: str, classes: list[str],
                feature_count: int, extra: dict | None = None) -> Path:
    output = config.path(config.values["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    joblib_path, onnx_path = output / f"{name}.joblib", output / f"{name}.onnx"
    joblib.dump(model, joblib_path)
    _export_onnx(model, feature_count, onnx_path)
    metadata = {
        "model": "linear_svm",
        "role": name,
        "label_order": classes,
        "feature_count": feature_count,
        "feature_transform": "per-joint temporal mean+std, flattened",
        "onnx_input": {"name": "features", "shape": [None, feature_count], "dtype": "float32"},
        "dependencies": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
        },
        "joblib_sha256": _sha256(joblib_path),
        "onnx_sha256": _sha256(onnx_path),
        **(extra or {}),
    }
    (output / f"{name}_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return onnx_path


def _score(model, x, y) -> dict:
    prediction = model.predict(x)
    return {
        "macro_f1": float(f1_score(y, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
    }


def train_teacher(config: Settings) -> Path:
    seed = int(config.values["seed"])
    seed_everything(seed)
    classes = json.loads(
        (config.path(config.values["data"]["processed_dir"]) / "classes.json").read_text()
    )
    x_train, y_train, _ = _dataset(config, "train", "teacher")
    x_val, y_val, _ = _dataset(config, "val", "teacher")
    teacher = _svm(seed)
    teacher.fit(x_train, y_train)
    metrics = _score(teacher, x_val, y_val)
    print("teacher", json.dumps(metrics))
    return _save_model(config, teacher, "teacher", classes, x_train.shape[1],
                       {"validation_metrics": metrics})


def distill_student(config: Settings) -> Path:
    seed = int(config.values["seed"])
    seed_everything(seed)
    output = config.path(config.values["output_dir"])
    teacher = joblib.load(output / "teacher.joblib")
    classes = json.loads(
        (config.path(config.values["data"]["processed_dir"]) / "classes.json").read_text()
    )
    x2, hard_labels, ids2 = _dataset(config, "train", "student")
    x3, _, ids3 = _dataset(config, "train", "teacher")
    if ids2 != ids3:
        raise RuntimeError("Teacher/student sample pairing was lost")
    teacher_probabilities = teacher.predict_proba(x3)
    alpha = float(config.values["distillation"]["alpha"])
    if not 0 <= alpha <= 1:
        raise ValueError("distillation.alpha must be between 0 and 1")

    # SVC cannot consume a probability vector as its target. Represent each soft
    # target as K copies carrying class-specific sample weights, then add the hard
    # target copy. Every original sample contributes total weight one.
    class_ids = np.arange(len(classes), dtype=np.int64)
    soft_x = np.repeat(x2, len(classes), axis=0)
    soft_y = np.tile(class_ids, len(x2))
    soft_weight = alpha * teacher_probabilities.reshape(-1)
    train_x = np.concatenate((x2, soft_x))
    train_y = np.concatenate((hard_labels, soft_y))
    weights = np.concatenate((np.full(len(x2), 1 - alpha), soft_weight))
    keep = weights > 1e-8

    student = _svm(seed)
    student.fit(
        train_x[keep], train_y[keep],
        classifier__sample_weight=weights[keep],
    )
    x_val, y_val, _ = _dataset(config, "val", "student")
    metrics = _score(student, x_val, y_val)
    print("student", json.dumps(metrics))
    return _save_model(
        config, student, "student", classes, x2.shape[1],
        {
            "validation_metrics": metrics,
            "distillation": {
                "method": "teacher-probability weighted pseudo-label expansion",
                "alpha": alpha,
                "temperature": None,
            },
        },
    )


def _verify_onnx(model: Pipeline, onnx_path: Path, x: np.ndarray) -> float:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    outputs = session.run(None, {"features": x.astype(np.float32)})
    onnx_probabilities = next(
        value for value in outputs
        if isinstance(value, np.ndarray) and value.ndim == 2 and value.shape[1] > 1
    )
    return float(np.max(np.abs(model.predict_proba(x) - onnx_probabilities)))


def evaluate(config: Settings) -> dict:
    output = config.path(config.values["output_dir"])
    student = joblib.load(output / "student.joblib")
    x, truth, _ = _dataset(config, "test", "student")
    prediction = student.predict(x)
    classes = json.loads(
        (config.path(config.values["data"]["processed_dir"]) / "classes.json").read_text()
    )
    report = {
        "model": "linear_svm",
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "onnx_max_probability_error": _verify_onnx(student, output / "student.onnx", x),
        "classification_report": classification_report(
            truth, prediction, labels=range(len(classes)), target_names=classes,
            zero_division=0, output_dict=True,
        ),
    }
    (output / "test_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "macro_f1": report["macro_f1"],
        "balanced_accuracy": report["balanced_accuracy"],
        "onnx_max_probability_error": report["onnx_max_probability_error"],
    }, indent=2))
    return report

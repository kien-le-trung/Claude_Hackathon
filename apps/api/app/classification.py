from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .analysis import SCHEMA_VERSION
from .events import consolidate_error_events
from .features import (
    BIOMECHANICAL_FEATURE_NAMES,
    FEATURE_SCHEMA_VERSION,
    biomechanical_measurements,
    feature_count,
    transform_poses,
)


LABELS = ["good", "bad_back", "bad_heel"]
INFERENCE_BATCH_SIZE = 64
MIN_VALID_BIOMECHANICAL_FEATURES = 10
MIN_CLASSIFIED_FRAMES_FOR_ANALYSIS = 3
MIN_CLASSIFIED_COVERAGE = 0.5


class ClassifierArtifactError(RuntimeError):
    pass


class SquatFormClassifier:
    def __init__(self, model_dir: Path, error_threshold: float = 0.8, session_factory=None):
        self.model_dir = Path(model_dir)
        self.error_threshold = error_threshold
        metadata_path = self.model_dir / "metadata.json"
        model_path = self.model_dir / "model.onnx"
        if not metadata_path.is_file() or not model_path.is_file():
            raise ClassifierArtifactError(f"Classifier artifact is incomplete: {self.model_dir}")
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self._validate_metadata(model_path)
        if session_factory is None:
            import onnxruntime as ort
            session_factory = lambda path: ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.session = session_factory(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

    def _validate_metadata(self, model_path: Path) -> None:
        expected = {
            "label_order": LABELS,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "extraction_schema_version": SCHEMA_VERSION,
        }
        for key, value in expected.items():
            if self.metadata.get(key) != value:
                raise ClassifierArtifactError(f"Classifier {key} is incompatible")
        feature_set = self.metadata.get("feature_set")
        if self.metadata.get("feature_count") != feature_count(feature_set):
            raise ClassifierArtifactError("Classifier feature count is incompatible")
        if feature_set == "biomechanical" and self.metadata.get(
            "biomechanical_feature_names"
        ) != list(BIOMECHANICAL_FEATURE_NAMES):
            raise ClassifierArtifactError("Classifier biomechanical feature order is incompatible")
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()
        if self.metadata.get("artifact_sha256") != digest:
            raise ClassifierArtifactError("Classifier artifact checksum does not match")

    def classify(self, extraction: dict) -> dict:
        detected_samples = []
        candidates = []
        for frame in extraction.get("frames", []):
            if not frame.get("poses"):
                continue
            pose = frame["poses"][0]
            measurements = biomechanical_measurements(pose)
            valid_count = sum(math.isfinite(value) for value in measurements.values())
            sample = (frame, pose, measurements, valid_count)
            detected_samples.append(sample)
            if valid_count >= MIN_VALID_BIOMECHANICAL_FEATURES:
                candidates.append(sample)

        probability_batches = []
        for start in range(0, len(candidates), INFERENCE_BATCH_SIZE):
            batch = candidates[start:start + INFERENCE_BATCH_SIZE]
            features = transform_poses(
                (pose for _, pose, _, _ in batch), self.metadata["feature_set"]
            )
            outputs = self.session.run(None, {self.input_name: features})
            probability_batches.append(self._probabilities(outputs))
        probabilities = (
            np.concatenate(probability_batches, axis=0)
            if probability_batches
            else np.empty((0, len(LABELS)), dtype=np.float32)
        )
        probability_by_frame = {
            id(frame): row for (frame, _, _, _), row in zip(candidates, probabilities)
        }
        frames = []
        for frame, pose, measurements, valid_count in detected_samples:
            row = probability_by_frame.get(id(frame))
            if row is None:
                frames.append({
                    "frame_number": frame["frame_number"],
                    "timestamp_ms": frame["timestamp_ms"],
                    "predicted_class": None,
                    "probabilities": {},
                    "measurements": {
                        name: value if math.isfinite(value) else None
                        for name, value in measurements.items()
                    },
                    "valid_measurement_count": valid_count,
                    "detection_state": "insufficient_biomechanical_features",
                })
                continue
            row_values = {label: float(row[index]) for index, label in enumerate(LABELS)}
            frames.append({
                "frame_number": frame["frame_number"],
                "timestamp_ms": frame["timestamp_ms"],
                "predicted_class": max(row_values, key=row_values.get),
                "probabilities": row_values,
                "measurements": {
                    name: value if math.isfinite(value) else None
                    for name, value in measurements.items()
                },
                "valid_measurement_count": valid_count,
                "detection_state": "classified",
                "_pose": pose,
            })
        detected = len(detected_samples)
        classified = len(candidates)
        sampled = extraction.get("summary", {}).get("sampled_frame_count", len(extraction.get("frames", [])))
        classified_coverage = classified / sampled if sampled else 0.0
        has_sufficient_analysis_evidence = (
            classified >= MIN_CLASSIFIED_FRAMES_FOR_ANALYSIS
            and classified_coverage >= MIN_CLASSIFIED_COVERAGE
        )
        if not has_sufficient_analysis_evidence:
            events = []
        sampling_fps = float(
            extraction.get("sampling", {}).get("effective_fps")
            or extraction.get("sampling", {}).get("target_fps")
            or extraction.get("summary", {}).get("sampling_fps")
            or 10.0
        )
        events = consolidate_error_events(
            frames,
            threshold=self.error_threshold,
            sampling_fps=sampling_fps,
        )
        compact_frames = [
            {key: value for key, value in frame.items() if key != "_pose"}
            for frame in frames
        ]
        return {
            "model_version": self.metadata["model_version"],
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "threshold": self.error_threshold,
            "frames": compact_frames,
            "events": events,
            "summary": {
                "schema_version": 2,
                "result": (
                    "errors_detected"
                    if events
                    else "good"
                    if has_sufficient_analysis_evidence
                    else "insufficient_pose_data"
                ),
                "events": events,
                "event_count": len(events),
                "event_counts": {
                    label: sum(event["type"] == label for event in events)
                    for label in LABELS[1:]
                },
                "pose_detected_frame_count": detected,
                "classified_frame_count": classified,
                "minimum_valid_measurements_per_frame": MIN_VALID_BIOMECHANICAL_FEATURES,
                "sampled_frame_count": sampled,
                "classified_frame_coverage": round(classified_coverage * 100, 1),
            },
        }

    def _probabilities(self, outputs) -> np.ndarray:
        # skl2onnx classifiers commonly return [labels, sequence<map>] or a matrix.
        candidate = outputs[-1]
        if isinstance(candidate, list) and candidate and isinstance(candidate[0], dict):
            return np.asarray([[row.get(index, row.get(label, 0.0)) for index, label in enumerate(LABELS)] for row in candidate], dtype=np.float32)
        matrix = np.asarray(candidate, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != len(LABELS):
            raise ClassifierArtifactError("Classifier returned an invalid probability matrix")
        return matrix

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from app.analysis import LANDMARK_NAMES
from app.classification import ClassifierArtifactError, SquatFormClassifier
from app.features import BIOMECHANICAL_FEATURE_NAMES, FEATURE_SCHEMA_VERSION, feature_count


class FakeSession:
    def __init__(self, _path):
        pass

    def get_inputs(self):
        return [SimpleNamespace(name="features")]

    def get_outputs(self):
        return [SimpleNamespace(name="labels"), SimpleNamespace(name="probabilities")]

    def run(self, _outputs, inputs):
        count = next(iter(inputs.values())).shape[0]
        probabilities = [
            [0.1, 0.85, 0.05] if index < 3 else [0.05, 0.1, 0.85]
            for index in range(count)
        ]
        return [np.asarray([1] * count), np.asarray(probabilities, dtype=np.float32)]


def make_pose():
    landmarks = [
        {"index": i, "name": name, "x": i + 1, "y": i + 2, "z": i + 3,
         "visibility": 0.9, "presence": 0.8}
        for i, name in enumerate(LANDMARK_NAMES)
    ]
    return {"landmarks": landmarks, "world_landmarks": landmarks}


def artifact(tmp_path):
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    metadata = {
        "model_version": "v1", "label_order": ["good", "bad_back", "bad_heel"],
        "feature_schema_version": FEATURE_SCHEMA_VERSION, "feature_set": "biomechanical",
        "feature_count": feature_count("biomechanical"), "extraction_schema_version": 1,
        "biomechanical_feature_names": list(BIOMECHANICAL_FEATURE_NAMES),
        "artifact_sha256": hashlib.sha256(b"model").hexdigest(),
    }
    (tmp_path / "metadata.json").write_text(json.dumps(metadata))
    return tmp_path


def test_both_errors_are_reported_at_inclusive_threshold(tmp_path):
    classifier = SquatFormClassifier(artifact(tmp_path), 0.8, session_factory=FakeSession)
    extraction = {
        "summary": {"sampled_frame_count": 8},
        "sampling": {"effective_fps": 10.0},
        "frames": [
            {"frame_number": index * 3, "timestamp_ms": index * 100, "poses": [make_pose()]}
            for index in range(6)
        ],
    }
    result = classifier.classify(extraction)
    assert [event["type"] for event in result["summary"]["events"]] == ["bad_back", "bad_heel"]
    assert result["summary"]["classified_frame_coverage"] == 75.0


def test_no_classifiable_frames_returns_insufficient_data(tmp_path):
    classifier = SquatFormClassifier(artifact(tmp_path), session_factory=FakeSession)
    result = classifier.classify({"frames": []})
    assert result["summary"]["result"] == "insufficient_pose_data"
    assert result["summary"]["classified_frame_count"] == 0


def test_pose_with_too_few_measurements_is_not_classified(tmp_path):
    classifier = SquatFormClassifier(artifact(tmp_path), session_factory=FakeSession)
    sample = make_pose()
    sample["world_landmarks"] = []
    result = classifier.classify({
        "summary": {"sampled_frame_count": 1},
        "frames": [{"frame_number": 0, "timestamp_ms": 0, "poses": [sample]}],
    })
    assert result["frames"][0]["detection_state"] == "insufficient_biomechanical_features"
    assert result["frames"][0]["valid_measurement_count"] == 1
    assert result["summary"]["pose_detected_frame_count"] == 1
    assert result["summary"]["classified_frame_count"] == 0
    assert result["summary"]["result"] == "insufficient_pose_data"


def test_incompatible_artifact_is_rejected(tmp_path):
    path = artifact(tmp_path)
    metadata = json.loads((path / "metadata.json").read_text())
    metadata["label_order"] = ["bad", "good"]
    (path / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(ClassifierArtifactError):
        SquatFormClassifier(path, session_factory=FakeSession)

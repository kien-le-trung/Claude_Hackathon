from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .analysis import LANDMARK_NAMES


FEATURE_SCHEMA_VERSION = "mediapipe_pose_biomechanics_v3"
FEATURE_SETS = {"normalized", "world", "combined", "biomechanical"}
_FIELDS = ("x", "y", "z", "visibility", "presence")
_MIN_CONFIDENCE = 0.5

BIOMECHANICAL_FEATURE_NAMES = (
    "left_knee_angle_world",
    "right_knee_angle_world",
    "left_hip_angle_world",
    "right_hip_angle_world",
    "left_ankle_angle_world",
    "right_ankle_angle_world",
    "left_foot_pitch_world",
    "right_foot_pitch_world",
    "torso_lean_image",
    "torso_lean_world",
    "knee_angle_asymmetry",
    "hip_angle_asymmetry",
    "ankle_angle_asymmetry",
)


class LandmarkFeatureError(ValueError):
    pass


def feature_count(feature_set: str) -> int:
    if feature_set == "biomechanical":
        return 2 * len(BIOMECHANICAL_FEATURE_NAMES)
    sources = 2 if feature_set == "combined" else 1
    # Five values and five validity indicators per landmark and source.
    return len(LANDMARK_NAMES) * len(_FIELDS) * 2 * sources


def _ordered(landmarks: list[dict]) -> list[dict]:
    if len(landmarks) != len(LANDMARK_NAMES):
        raise LandmarkFeatureError(f"Expected 33 landmarks, received {len(landmarks)}")
    by_index = {item.get("index"): item for item in landmarks}
    if set(by_index) != set(range(len(LANDMARK_NAMES))):
        raise LandmarkFeatureError("Landmark indexes are incomplete or duplicated")
    ordered = [by_index[index] for index in range(len(LANDMARK_NAMES))]
    for index, item in enumerate(ordered):
        if item.get("name") != LANDMARK_NAMES[index]:
            raise LandmarkFeatureError(f"Unexpected landmark name at index {index}")
    return ordered


def _finite(value) -> float:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return math.nan
    return converted if math.isfinite(converted) else math.nan


def _transform_source(landmarks: list[dict]) -> list[float]:
    ordered = _ordered(landmarks)
    left_hip, right_hip = ordered[23], ordered[24]
    left_shoulder, right_shoulder = ordered[11], ordered[12]
    hip = np.array([
        np.nanmean([_finite(left_hip.get(axis)), _finite(right_hip.get(axis))])
        for axis in ("x", "y", "z")
    ])
    shoulder = np.array([
        np.nanmean([_finite(left_shoulder.get(axis)), _finite(right_shoulder.get(axis))])
        for axis in ("x", "y", "z")
    ])
    torso_length = float(np.linalg.norm(shoulder - hip))
    if not math.isfinite(torso_length) or torso_length <= 1e-8:
        torso_length = math.nan

    values: list[float] = []
    valid: list[float] = []
    for item in ordered:
        for field in _FIELDS:
            value = _finite(item.get(field))
            if field in {"x", "y", "z"} and math.isfinite(value):
                axis = ("x", "y", "z").index(field)
                value = (value - hip[axis]) / torso_length
            is_valid = math.isfinite(value)
            values.append(value if is_valid else 0.0)
            valid.append(1.0 if is_valid else 0.0)
    return values + valid


def transform_pose(pose: dict, feature_set: str = "combined") -> np.ndarray:
    if feature_set not in FEATURE_SETS:
        raise LandmarkFeatureError(f"Unknown feature set: {feature_set}")
    output: list[float] = []
    if feature_set in {"normalized", "combined"}:
        output.extend(_transform_source(pose.get("landmarks", [])))
    if feature_set in {"world", "combined"}:
        output.extend(_transform_source(pose.get("world_landmarks", [])))
    if feature_set == "biomechanical":
        measurements = biomechanical_measurements(pose)
        values = [measurements[name] for name in BIOMECHANICAL_FEATURE_NAMES]
        output.extend(value if math.isfinite(value) else 0.0 for value in values)
        output.extend(1.0 if math.isfinite(value) else 0.0 for value in values)
    return np.asarray(output, dtype=np.float32)


def transform_poses(poses: Iterable[dict], feature_set: str = "combined") -> np.ndarray:
    rows = [transform_pose(pose, feature_set) for pose in poses]
    return np.stack(rows) if rows else np.empty((0, feature_count(feature_set)), dtype=np.float32)


def _point(
    landmarks: list[dict],
    index: int,
    *,
    require_visibility: bool,
) -> np.ndarray | None:
    if len(landmarks) != len(LANDMARK_NAMES):
        return None
    item = landmarks[index]
    coordinates = np.asarray([_finite(item.get(axis)) for axis in ("x", "y", "z")])
    if not np.isfinite(coordinates).all():
        return None
    presence = _finite(item.get("presence"))
    visibility = _finite(item.get("visibility"))
    if math.isfinite(presence) and presence < _MIN_CONFIDENCE:
        return None
    if require_visibility and math.isfinite(visibility) and visibility < _MIN_CONFIDENCE:
        return None
    return coordinates


def _midpoint(first: np.ndarray | None, second: np.ndarray | None) -> np.ndarray | None:
    if first is None or second is None:
        return None
    return (first + second) / 2.0


def _angle(first: np.ndarray | None, vertex: np.ndarray | None, third: np.ndarray | None) -> float:
    if first is None or vertex is None or third is None:
        return math.nan
    left = first - vertex
    right = third - vertex
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if not math.isfinite(denominator) or denominator <= 1e-8:
        return math.nan
    cosine = float(np.clip(np.dot(left, right) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _lean(vector: np.ndarray | None, up: np.ndarray) -> float:
    if vector is None:
        return math.nan
    denominator = float(np.linalg.norm(vector) * np.linalg.norm(up))
    if not math.isfinite(denominator) or denominator <= 1e-8:
        return math.nan
    cosine = float(np.clip(np.dot(vector, up) / denominator, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _foot_pitch(heel: np.ndarray | None, toe: np.ndarray | None) -> float:
    """Return signed heel-to-toe elevation relative to the world XZ plane."""
    if heel is None or toe is None:
        return math.nan
    vector = toe - heel
    horizontal = float(np.linalg.norm(vector[[0, 2]]))
    vertical = float(vector[1])
    if not math.isfinite(horizontal) or not math.isfinite(vertical):
        return math.nan
    if horizontal <= 1e-8 and abs(vertical) <= 1e-8:
        return math.nan
    return math.degrees(math.atan2(vertical, horizontal))


def _absolute_difference(first: float, second: float) -> float:
    if not math.isfinite(first) or not math.isfinite(second):
        return math.nan
    return abs(first - second)


def biomechanical_measurements(pose: dict) -> dict[str, float]:
    """Return ordered, human-readable per-frame geometric measurements."""
    image = pose.get("landmarks", [])
    world = pose.get("world_landmarks", [])

    ip = lambda index: _point(image, index, require_visibility=True)
    wp = lambda index: _point(world, index, require_visibility=False)

    left_shoulder_i, right_shoulder_i = ip(11), ip(12)
    left_hip_i, right_hip_i = ip(23), ip(24)
    image_hip = _midpoint(left_hip_i, right_hip_i)
    image_shoulder = _midpoint(left_shoulder_i, right_shoulder_i)

    left_shoulder_w, right_shoulder_w = wp(11), wp(12)
    left_hip_w, right_hip_w = wp(23), wp(24)
    left_knee_w, right_knee_w = wp(25), wp(26)
    left_ankle_w, right_ankle_w = wp(27), wp(28)
    left_heel_w, right_heel_w = wp(29), wp(30)
    left_foot_w, right_foot_w = wp(31), wp(32)
    world_hip = _midpoint(left_hip_w, right_hip_w)
    world_shoulder = _midpoint(left_shoulder_w, right_shoulder_w)

    left_knee = _angle(left_hip_w, left_knee_w, left_ankle_w)
    right_knee = _angle(right_hip_w, right_knee_w, right_ankle_w)
    left_hip = _angle(left_shoulder_w, left_hip_w, left_knee_w)
    right_hip = _angle(right_shoulder_w, right_hip_w, right_knee_w)
    left_ankle = _angle(left_knee_w, left_ankle_w, left_foot_w)
    right_ankle = _angle(right_knee_w, right_ankle_w, right_foot_w)

    image_torso = image_shoulder - image_hip if image_shoulder is not None and image_hip is not None else None
    world_torso = world_shoulder - world_hip if world_shoulder is not None and world_hip is not None else None

    values = {
        "left_knee_angle_world": left_knee,
        "right_knee_angle_world": right_knee,
        "left_hip_angle_world": left_hip,
        "right_hip_angle_world": right_hip,
        "left_ankle_angle_world": left_ankle,
        "right_ankle_angle_world": right_ankle,
        "left_foot_pitch_world": _foot_pitch(left_heel_w, left_foot_w),
        "right_foot_pitch_world": _foot_pitch(right_heel_w, right_foot_w),
        "torso_lean_image": _lean(image_torso, np.asarray([0.0, -1.0, 0.0])),
        "torso_lean_world": _lean(world_torso, np.asarray([0.0, -1.0, 0.0])),
        "knee_angle_asymmetry": _absolute_difference(left_knee, right_knee),
        "hip_angle_asymmetry": _absolute_difference(left_hip, right_hip),
        "ankle_angle_asymmetry": _absolute_difference(left_ankle, right_ankle),
    }
    return {name: float(values[name]) for name in BIOMECHANICAL_FEATURE_NAMES}

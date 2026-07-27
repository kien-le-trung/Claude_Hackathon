"""Metrics for comparing raw and smoothed MediaPipe skeleton sequences."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from alignment import load_pose_sequence, squat_frame_ids
from config import COMMON_JOINTS, OUTPUT_DIR
from extract_mediapipe import load_extracted_landmarks


COMMON_BONES = (
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
)


def pelvis_center(points: np.ndarray, names: list[str]) -> np.ndarray:
    """Place each frame's hip midpoint at the origin."""
    left_hip = names.index("left_hip")
    right_hip = names.index("right_hip")
    pelvis = (points[:, left_hip] + points[:, right_hip]) / 2.0
    return points - pelvis[:, None, :]


def procrustes_alignment(
    predicted: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Fit one reflection-free similarity transform across a whole sequence."""
    if predicted.shape != target.shape or predicted.shape[-1] != 3:
        raise ValueError("Predicted and target skeleton arrays must match in XYZ")

    source = predicted.reshape(-1, 3)
    destination = target.reshape(-1, 3)
    source_mean = source.mean(axis=0)
    destination_mean = destination.mean(axis=0)
    source_centered = source - source_mean
    destination_centered = destination - destination_mean

    covariance = source_centered.T @ destination_centered
    u, singular_values, vt = np.linalg.svd(covariance)
    direction = np.ones(3)
    if np.linalg.det(vt.T @ u.T) < 0:
        direction[-1] = -1
    rotation = vt.T @ np.diag(direction) @ u.T

    source_energy = np.sum(source_centered**2)
    if source_energy <= np.finfo(np.float64).eps:
        raise ValueError("Cannot align a degenerate skeleton")
    scale = float(np.sum(singular_values * direction) / source_energy)
    translation = destination_mean - scale * (source_mean @ rotation.T)
    aligned = (scale * (source @ rotation.T) + translation).reshape(predicted.shape)
    return aligned, scale, rotation, translation


def mpjpe(predicted: np.ndarray, target: np.ndarray) -> float:
    """Mean per-joint Euclidean position error, in the input distance unit."""
    return float(np.linalg.norm(predicted - target, axis=-1).mean())


def pa_mpjpe(
    predicted: np.ndarray,
    target: np.ndarray,
) -> tuple[float, dict[str, object]]:
    """Sequence-level Procrustes-aligned MPJPE."""
    aligned, scale, rotation, translation = procrustes_alignment(predicted, target)
    details = {
        "scale": scale,
        "rotation": rotation.tolist(),
        "translation_m": translation.tolist(),
        "rotation_determinant": float(np.linalg.det(rotation)),
    }
    return mpjpe(aligned, target), details


def percentile_joint_error(
    predicted: np.ndarray,
    target: np.ndarray,
    percentile: float = 95.0,
) -> float:
    errors = np.linalg.norm(predicted - target, axis=-1)
    return float(np.percentile(errors, percentile))


def per_joint_errors(
    predicted: np.ndarray,
    target: np.ndarray,
    names: list[str],
) -> dict[str, float]:
    errors = np.linalg.norm(predicted - target, axis=-1)
    return {name: float(errors[:, index].mean()) for index, name in enumerate(names)}


def per_axis_errors(
    predicted: np.ndarray,
    target: np.ndarray,
) -> dict[str, float]:
    errors = np.abs(predicted - target)
    return {
        axis: float(errors[..., index].mean())
        for index, axis in enumerate(("left_right", "depth", "vertical"))
    }


def mean_squared_jerk(points: np.ndarray, fps: float) -> float:
    """Mean squared third temporal derivative, in distance^2 / second^6."""
    if fps <= 0:
        raise ValueError("FPS must be positive")
    if len(points) < 4:
        return 0.0
    jerk = np.diff(points, n=3, axis=0) * fps**3
    return float(np.mean(np.sum(jerk**2, axis=-1)))


def bone_length_variation(
    points: np.ndarray,
    names: list[str],
    bones: tuple[tuple[str, str], ...] = COMMON_BONES,
) -> dict[str, object]:
    """Coefficient of variation for each common anatomical bone."""
    per_bone = {}
    for parent, child in bones:
        parent_index = names.index(parent)
        child_index = names.index(child)
        lengths = np.linalg.norm(
            points[:, parent_index] - points[:, child_index],
            axis=-1,
        )
        mean_length = float(lengths.mean())
        coefficient = (
            float(lengths.std() / mean_length)
            if mean_length > np.finfo(np.float64).eps
            else 0.0
        )
        per_bone[f"{parent}_to_{child}"] = coefficient
    return {
        "mean_coefficient_of_variation": float(np.mean(list(per_bone.values()))),
        "per_bone": per_bone,
    }


def joint_angle(
    first: np.ndarray,
    vertex: np.ndarray,
    third: np.ndarray,
) -> np.ndarray:
    first_vector = first - vertex
    third_vector = third - vertex
    denominator = (
        np.linalg.norm(first_vector, axis=-1)
        * np.linalg.norm(third_vector, axis=-1)
    )
    cosine = np.divide(
        np.sum(first_vector * third_vector, axis=-1),
        denominator,
        out=np.ones_like(denominator),
        where=denominator > np.finfo(np.float64).eps,
    )
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def squat_fidelity(points: np.ndarray, names: list[str]) -> dict[str, float | int]:
    """Measure squat extrema in MM-Fit axes (vertical axis is coordinate 2)."""
    indices = {name: names.index(name) for name in names}
    pelvis = (
        points[:, indices["left_hip"]] + points[:, indices["right_hip"]]
    ) / 2.0
    ankle_center = (
        points[:, indices["left_ankle"]] + points[:, indices["right_ankle"]]
    ) / 2.0
    pelvis_height = pelvis[:, 2] - ankle_center[:, 2]
    bottom_index = int(np.argmin(pelvis_height))

    knee_angles = {}
    for side in ("left", "right"):
        knee_angles[side] = joint_angle(
            points[:, indices[f"{side}_hip"]],
            points[:, indices[f"{side}_knee"]],
            points[:, indices[f"{side}_ankle"]],
        )

    shoulder_center = (
        points[:, indices["left_shoulder"]]
        + points[:, indices["right_shoulder"]]
    ) / 2.0
    trunk = shoulder_center - pelvis
    trunk_norm = np.linalg.norm(trunk, axis=-1)
    vertical_cosine = np.divide(
        trunk[:, 2],
        trunk_norm,
        out=np.ones_like(trunk_norm),
        where=trunk_norm > np.finfo(np.float64).eps,
    )
    trunk_lean = np.degrees(np.arccos(np.clip(vertical_cosine, -1.0, 1.0)))

    return {
        "bottom_frame_offset": bottom_index,
        "minimum_pelvis_height_m": float(pelvis_height[bottom_index]),
        "minimum_left_knee_angle_deg": float(knee_angles["left"].min()),
        "minimum_right_knee_angle_deg": float(knee_angles["right"].min()),
        "maximum_trunk_lean_deg": float(trunk_lean.max()),
    }


def coverage(evaluated_count: int, expected_count: int) -> dict[str, float | int]:
    return {
        "evaluated_frames": evaluated_count,
        "expected_frames": expected_count,
        "percent": (
            float(evaluated_count / expected_count * 100.0)
            if expected_count
            else 0.0
        ),
    }


def _accuracy_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    names: list[str],
) -> dict[str, object]:
    aligned_error, alignment = pa_mpjpe(predicted, target)
    return {
        "mpjpe_mm": mpjpe(predicted, target) * 1000.0,
        "pa_mpjpe_mm": aligned_error * 1000.0,
        "p95_joint_error_mm": percentile_joint_error(predicted, target) * 1000.0,
        "per_joint_mpjpe_mm": {
            name: error * 1000.0
            for name, error in per_joint_errors(predicted, target, names).items()
        },
        "per_axis_mae_mm": {
            axis: error * 1000.0
            for axis, error in per_axis_errors(predicted, target).items()
        },
        "alignment": alignment,
    }


def evaluate_arrays(
    raw: np.ndarray,
    smoothed: np.ndarray,
    target: np.ndarray,
    names: list[str],
    frame_ids: np.ndarray,
    fps: float,
    expected_count: int,
) -> dict[str, object]:
    """Build the complete raw-versus-smoothed scorecard."""
    if raw.shape != smoothed.shape or raw.shape != target.shape:
        raise ValueError("Raw, smoothed, and target arrays must have matching shapes")
    if len(frame_ids) != len(raw):
        raise ValueError("Frame IDs must match the number of skeleton frames")
    if not all(np.isfinite(array).all() for array in (raw, smoothed, target)):
        raise ValueError("Metric inputs must contain only finite values")

    raw_centered = pelvis_center(raw, names)
    smoothed_centered = pelvis_center(smoothed, names)
    target_centered = pelvis_center(target, names)

    raw_fidelity = squat_fidelity(raw, names)
    smoothed_fidelity = squat_fidelity(smoothed, names)
    target_fidelity = squat_fidelity(target, names)
    bottom_offsets = {
        "raw_vs_ground_truth_frames": (
            raw_fidelity["bottom_frame_offset"]
            - target_fidelity["bottom_frame_offset"]
        ),
        "smoothed_vs_ground_truth_frames": (
            smoothed_fidelity["bottom_frame_offset"]
            - target_fidelity["bottom_frame_offset"]
        ),
        "smoothed_vs_raw_frames": (
            smoothed_fidelity["bottom_frame_offset"]
            - raw_fidelity["bottom_frame_offset"]
        ),
    }

    return {
        "frame_range": [int(frame_ids[0]), int(frame_ids[-1])],
        "fps": float(fps),
        "coverage": coverage(len(frame_ids), expected_count),
        "raw": {
            "accuracy": _accuracy_metrics(raw_centered, target_centered, names),
            "mean_squared_jerk_m2_s6": mean_squared_jerk(raw_centered, fps),
            "bone_length_variation": bone_length_variation(raw, names),
            "squat_fidelity": raw_fidelity,
        },
        "smoothed": {
            "accuracy": _accuracy_metrics(
                smoothed_centered, target_centered, names
            ),
            "mean_squared_jerk_m2_s6": mean_squared_jerk(smoothed_centered, fps),
            "bone_length_variation": bone_length_variation(smoothed, names),
            "squat_fidelity": smoothed_fidelity,
        },
        "ground_truth": {
            "mean_squared_jerk_m2_s6": mean_squared_jerk(target_centered, fps),
            "bone_length_variation": bone_length_variation(target, names),
            "squat_fidelity": target_fidelity,
        },
        "bottom_timing_offsets": bottom_offsets,
    }


def evaluate_reconstruction_sequence(
    frame_ids: np.ndarray,
    raw_world_landmarks: np.ndarray,
    smoothed_world_landmarks: np.ndarray,
    fps: float = 30.0,
) -> dict[str, object]:
    """Join one MediaPipe sequence to MM-Fit and calculate its scorecard."""
    mmfit_frame_ids, mmfit_joints, _ = load_pose_sequence()
    mmfit_rows = {
        int(frame_id): index for index, frame_id in enumerate(mmfit_frame_ids)
    }
    keep = np.asarray(
        [index for index, frame_id in enumerate(frame_ids) if int(frame_id) in mmfit_rows],
        dtype=np.int64,
    )
    if not len(keep):
        raise ValueError("No MediaPipe frames overlap MM-Fit ground truth")

    joined_frame_ids = frame_ids[keep]
    names = list(COMMON_JOINTS)
    media_pipe_indices = [mapping[1] for mapping in COMMON_JOINTS.values()]
    mmfit_indices = [mapping[0] for mapping in COMMON_JOINTS.values()]

    raw = raw_world_landmarks[keep][:, media_pipe_indices]
    smoothed = smoothed_world_landmarks[keep][:, media_pipe_indices]
    # MediaPipe XYZ -> MM-Fit X, depth, up.
    raw = raw[..., [0, 2, 1]].copy()
    smoothed = smoothed[..., [0, 2, 1]].copy()
    raw[..., 2] *= -1.0
    smoothed[..., 2] *= -1.0
    target = np.stack(
        [mmfit_joints[mmfit_rows[int(frame_id)], mmfit_indices] for frame_id in joined_frame_ids]
    ) / 1000.0

    expected = int(
        np.count_nonzero(
            (squat_frame_ids() >= joined_frame_ids[0])
            & (squat_frame_ids() <= joined_frame_ids[-1])
        )
    )
    return evaluate_arrays(
        raw, smoothed, target, names, joined_frame_ids, fps, expected
    )


def evaluate(output_path: Path | None = None) -> dict[str, object]:
    """Evaluate all detected contiguous records and save a JSON scorecard."""
    records = [
        record for record in load_extracted_landmarks()
        if record.get("detected") and record.get("world_landmarks")
    ]
    if not records:
        raise ValueError("No detected MediaPipe skeletons are available")

    # This entry point evaluates an unchanged baseline. Reconstruction supplies
    # the actual smoothed arrays through evaluate_reconstruction_sequence().
    frame_ids = np.asarray([int(record["frame_id"]) for record in records])
    raw = np.asarray([
        [[landmark["x"], landmark["y"], landmark["z"]]
         for landmark in record["world_landmarks"]]
        for record in records
    ])
    result = evaluate_reconstruction_sequence(frame_ids, raw, raw)
    path = output_path or OUTPUT_DIR / "w00_metrics_raw_baseline.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    result = evaluate()
    print(json.dumps(result, indent=2))

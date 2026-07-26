"""Compute overall MediaPipe 3D accuracy against MM-Fit w00."""

from __future__ import annotations

import json

import numpy as np

from alignment import load_pose_sequence, squat_frame_ids
from extract_mediapipe import load_extracted_landmarks
from config import COMMON_JOINTS, OUTPUT_DIR


JSONL_PATH = OUTPUT_DIR / "w00_mediapipe.jsonl"


def join_by_frame_id(records: list[dict]) -> list[dict]:
    mmfit_frame_ids, mmfit_joints, _ = load_pose_sequence()
    mmfit_row_by_frame = {
        int(frame_id): row_index
        for row_index, frame_id in enumerate(mmfit_frame_ids)
    }
    joined = []
    for record in records:
        frame_id = int(record["frame_id"])
        if frame_id not in mmfit_row_by_frame:
            continue
        joined.append({
            "frame_id": frame_id,
            "mediapipe_world": record["world_landmarks"],
            "detected": record["detected"],
            "mmfit_joints": mmfit_joints[mmfit_row_by_frame[frame_id]],
        })
    return joined


def select_common_joints(joined_frame: dict) -> dict | None:
    """Select the same anatomical joints from both skeleton definitions."""
    mediapipe_landmarks = joined_frame["mediapipe_world"]
    if not joined_frame["detected"] or mediapipe_landmarks is None:
        return None

    mmfit_points = []
    mediapipe_points = []
    for mmfit_index, mediapipe_index in COMMON_JOINTS.values():
        mmfit_points.append(joined_frame["mmfit_joints"][mmfit_index])
        landmark = mediapipe_landmarks[mediapipe_index]
        mediapipe_points.append([landmark["x"], landmark["y"], landmark["z"]])
    return {
        "names": list(COMMON_JOINTS),
        "mmfit": np.asarray(mmfit_points, dtype=np.float64),
        "mediapipe": np.asarray(mediapipe_points, dtype=np.float64),
    }


def pelvis_center(points: np.ndarray, names: list[str]) -> np.ndarray:
    """Remove global position by placing each frame's hip midpoint at the origin."""
    left_hip = names.index("left_hip")
    right_hip = names.index("right_hip")
    pelvis = (points[:, left_hip] + points[:, right_hip]) / 2.0
    return points - pelvis[:, None, :]


def procrustes_alignment(common_joints):
    mediapipe_centered = common_joints["mediapipe"]
    mmfit_centered = common_joints["mmfit"]

    if mediapipe_centered.shape != mmfit_centered.shape:
        raise ValueError("MediaPipe and MM-Fit arrays must have matching shapes")
    if mediapipe_centered.shape[-1] != 3:
        raise ValueError("Expected joint coordinates ending in XYZ")

    source = mediapipe_centered.reshape(-1, 3)
    target = mmfit_centered.reshape(-1, 3)

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)

    source_centered = source - source_mean
    target_centered = target - target_mean

    covariance = source_centered.T @ target_centered
    u, singular_values, vt = np.linalg.svd(covariance)
    initial_rotation = vt.T @ u.T
    direction = np.ones(3)
    if np.linalg.det(initial_rotation) < 0:
        direction[-1] = -1
    correction = np.diag(direction)
    rotation = vt.T @ correction @ u.T

    source_energy = np.sum(source_centered ** 2)
    if source_energy <= np.finfo(np.float64).eps:
        raise ValueError("Cannot align a degenerate MediaPipe skeleton")
    scale = np.sum(singular_values * direction) / source_energy
    translation = (
        target_mean
        - scale * (source_mean @ rotation.T)
    )

    aligned_flat = scale * (source @ rotation.T) + translation
    aligned = aligned_flat.reshape(mediapipe_centered.shape)

    return (
        aligned,
        scale,
        rotation,
        translation
    )


def evaluate() -> None:
    records = load_extracted_landmarks()
    joined_frames = join_by_frame_id(records)
    selected = [
        points
        for joined_frame in joined_frames
        if (points := select_common_joints(joined_frame)) is not None
    ]
    if not selected:
        raise ValueError("No detected MediaPipe poses overlap MM-Fit ground truth")

    names = selected[0]["names"]

    # MM-Fit is millimeter-scale; convert it to MediaPipe's meter scale.
    mmfit = np.stack([item["mmfit"] for item in selected]) / 1000.0
    mediapipe = np.stack([item["mediapipe"] for item in selected])

    if not np.isfinite(mmfit).all() or not np.isfinite(mediapipe).all():
        raise ValueError("Non-finite joint coordinates found in evaluation data")

    # MM-Fit: X=left/right, Y=depth, Z=up.
    # MediaPipe: X=left/right, Y=down, Z=depth.
    # This is a fixed coordinate conversion, not a fitted correction.
    mediapipe_mmfit_axes = mediapipe[..., [0, 2, 1]].copy()
    mediapipe_mmfit_axes[..., 2] *= -1.0

    mmfit_centered = pelvis_center(mmfit, names)
    mediapipe_centered = pelvis_center(mediapipe_mmfit_axes, names)

    absolute_errors = np.abs(mediapipe_centered - mmfit_centered)
    overall_mae_mm = float(absolute_errors.mean() * 1000.0)

    aligned, scale, rotation, translation = procrustes_alignment({
        "mediapipe": mediapipe_centered,
        "mmfit": mmfit_centered,
    })
    aligned_absolute_errors = np.abs(aligned - mmfit_centered)
    aligned_mae_mm = float(aligned_absolute_errors.mean() * 1000.0)
    rotation_cosine = np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)
    rotation_degrees = float(np.degrees(np.arccos(rotation_cosine)))

    expected_count = len(squat_frame_ids())
    coverage = len(selected) / expected_count * 100.0

    print("MM-Fit w00 / MediaPipe 3D evaluation")
    print("=" * 42)
    print(f"Evaluated frames: {len(selected)} / {expected_count}")
    print(f"Detection coverage: {coverage:.1f}%")
    print(f"Compared joints per frame: {len(names)}")
    print()
    print(f"Overall 3D coordinate MAE before alignment: {overall_mae_mm:.1f} mm")
    print(f"Overall 3D coordinate MAE after alignment:  {aligned_mae_mm:.1f} mm")
    print(f"Fitted scale correction: {scale:.5f}")
    print(f"Fitted rotation correction: {rotation_degrees:.2f} degrees")
    print(f"Rotation determinant: {np.linalg.det(rotation):.5f}")
    print(
        "Fitted translation: "
        f"[{translation[0]:.5f}, {translation[1]:.5f}, "
        f"{translation[2]:.5f}] m"
    )
    print()
    print("The MAE is the average absolute X, Y, or Z coordinate error after:")
    print("  1. converting MM-Fit millimeters to meters,")
    print("  2. mapping MediaPipe axes to MM-Fit axes, and")
    print("  3. centering both skeletons on the hip midpoint.")
    print("The aligned MAE additionally applies one fitted similarity transform")
    print("(scale, rotation, and translation) to the complete sequence.")


if __name__ == "__main__":
    evaluate()

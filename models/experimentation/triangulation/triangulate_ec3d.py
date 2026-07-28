"""Lesson 5: apply your geometry implementation to one EC3D frame."""

from __future__ import annotations

import cv2
import numpy as np

from config import ALL_CAMERAS, START_CAMERAS
from data import common_joint_names, first_usable_frame, load_ec3d
from geometry import (
    camera_depth,
    project_point,
    projection_matrix,
    reprojection_errors,
    triangulate_point_dlt,
)


def observation_xy(observation) -> np.ndarray:
    """Extract `(u, v)` from one EC3D `2D_op` joint observation.

    Inspect the value in `inspect_ec3d.py`, document its schema, and implement
    this conversion. Preserve confidence separately if one is included.
    """
    point = np.asarray(observation, dtype=np.float64)

    if point.shape != (2,):
        raise ValueError(f"Expected a 2D observation shaped (2,), received {point.shape}")
    if not np.isfinite(point).all():
        raise ValueError("2D observation contains non-finite coordinates")
    return point.copy()


def ground_truth_xyz(observation) -> np.ndarray:
    """Extract metric XYZ from one EC3D `3D_gt` joint value."""
    point = np.asarray(observation, dtype=np.float64)

    if point.shape != (3,):
        raise ValueError(f"Expected a 3D observation shaped (3,), received {point.shape}")
    if not np.isfinite(point).all():
        raise ValueError("3D point contains non-finite coordinates")
    return point.copy()


def inverted_extrinsics(
    rotation: np.ndarray,
    translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a camera-to-world pose into world-to-camera extrinsics."""
    inverse_rotation = rotation.T
    inverse_translation = -(inverse_rotation @ translation)
    return inverse_rotation, inverse_translation


def project_point_with_distortion(
    point_3d: np.ndarray,
    intrinsic: np.ndarray,
    distortion: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    """Project a world point into raw pixels using OpenCV's distortion model."""
    rotation_vector, _ = cv2.Rodrigues(rotation)
    projected, _ = cv2.projectPoints(
        point_3d.reshape(1, 1, 3),
        rotation_vector,
        translation.reshape(3, 1),
        intrinsic,
        distortion,
    )
    return projected.reshape(2)


def diagnose_camera_models(
    frame: dict,
    camera_parameters: dict,
    camera_ids: tuple[str, ...] = ALL_CAMERAS,
) -> dict[str, dict[str, dict[str, float | int]]]:
    """Compare direct/inverted extrinsics with and without lens distortion."""
    ground_truth = frame.get("3D_gt") or {}
    observations_by_camera = frame.get("2D_op") or {}
    results: dict[str, dict[str, dict[str, float | int]]] = {}

    for camera_id in camera_ids:
        parameters = camera_parameters[camera_id]
        intrinsics = parameters["intrinsics"]
        extrinsics = parameters["extrinsics"]
        intrinsic = np.asarray(intrinsics["K"], dtype=np.float64)
        distortion = np.asarray(intrinsics["distCoeffs"], dtype=np.float64)
        direct_rotation = np.asarray(extrinsics["R"], dtype=np.float64)
        direct_translation = np.asarray(extrinsics["t"], dtype=np.float64)
        inverse_rotation, inverse_translation = inverted_extrinsics(
            direct_rotation,
            direct_translation,
        )

        camera_observations = observations_by_camera.get(camera_id) or {}
        joint_names = sorted(set(camera_observations) & set(ground_truth))
        models = {
            "direct_pinhole": (
                direct_rotation,
                direct_translation,
                False,
            ),
            "direct_distorted": (
                direct_rotation,
                direct_translation,
                True,
            ),
            "inverse_pinhole": (
                inverse_rotation,
                inverse_translation,
                False,
            ),
            "inverse_distorted": (
                inverse_rotation,
                inverse_translation,
                True,
            ),
        }
        camera_results: dict[str, dict[str, float | int]] = {}
        for model_name, (rotation, translation, use_distortion) in models.items():
            projection = projection_matrix(intrinsic, rotation, translation)
            errors = []
            positive_depths = 0
            for joint_name in joint_names:
                truth = ground_truth_xyz(ground_truth[joint_name])
                observed = observation_xy(camera_observations[joint_name])
                if use_distortion:
                    projected = project_point_with_distortion(
                        truth,
                        intrinsic,
                        distortion,
                        rotation,
                        translation,
                    )
                else:
                    projected = project_point(projection, truth)
                errors.append(float(np.linalg.norm(projected - observed)))
                positive_depths += camera_depth(
                    truth,
                    rotation,
                    translation,
                ) > 0

            error_array = np.asarray(errors, dtype=np.float64)
            camera_results[model_name] = {
                "joint_count": len(joint_names),
                "median_error_px": (
                    float(np.median(error_array)) if len(error_array) else float("nan")
                ),
                "mean_error_px": (
                    float(np.mean(error_array)) if len(error_array) else float("nan")
                ),
                "positive_depth_percent": (
                    float(positive_depths / len(joint_names) * 100.0)
                    if joint_names else 0.0
                ),
            }
        results[camera_id] = camera_results
    return results


def print_camera_diagnostics(
    diagnostics: dict[str, dict[str, dict[str, float | int]]],
) -> None:
    print("\nGround-truth camera-model diagnostic")
    print(
        "Lower reprojection error is better; visible joints should have "
        "positive camera depth."
    )
    for camera_id, models in diagnostics.items():
        print(f"\nCamera {camera_id}")
        for model_name, metrics in models.items():
            print(
                f"  {model_name:<18} "
                f"median={metrics['median_error_px']:9.3f} px  "
                f"mean={metrics['mean_error_px']:9.3f} px  "
                f"positive-depth={metrics['positive_depth_percent']:6.1f}%  "
                f"joints={metrics['joint_count']}"
            )


def camera_projection(parameters: dict) -> np.ndarray:
    """Build P after confirming the stored extrinsic convention."""
    intrinsics = parameters["intrinsics"]
    extrinsics = parameters["extrinsics"]
    # Decide whether these stored R/t already map world to camera. Validate that
    # decision through reprojection and positive camera depth.
    return projection_matrix(intrinsics["K"], extrinsics["R"], extrinsics["t"])

def triangulate_joint(
    frame: dict,
    camera_parameters: dict,
    joint_name,
    camera_ids: tuple[str, ...],
) -> dict:
    projections = []
    observations = []
    rotations = []
    translations = []

    for camera_id in camera_ids:
        parameters = camera_parameters[camera_id]
        intrinsics = parameters["intrinsics"]
        extrinsics = parameters["extrinsics"]
        rotation = np.asarray(extrinsics["R"], dtype=np.float64)
        translation = np.asarray(extrinsics["t"], dtype=np.float64)
        projections.append(
            projection_matrix(
                np.asarray(intrinsics["K"], dtype=np.float64),
                rotation,
                translation,
            )
        )
        observations.append(
            observation_xy(frame["2D_op"][camera_id][joint_name])
        )
        rotations.append(rotation)
        translations.append(translation)

    estimate = triangulate_point_dlt(
        projections,
        observations,
    )
    pixel_errors = reprojection_errors(
        estimate,
        projections,
        observations,
    )
    depths = np.asarray([
        camera_depth(estimate, rotation, translation)
        for rotation, translation in zip(rotations, translations)
    ])
    truth = ground_truth_xyz(frame["3D_gt"][joint_name])
    error_3d = np.linalg.norm(estimate - truth)
    return {
        "joint_name": joint_name,
        "camera_ids": camera_ids,
        "estimate": estimate,
        "ground_truth": truth,
        "error_3d": float(error_3d),
        "reprojection_errors": pixel_errors,
        "camera_depths": depths,
    }


def main() -> None:
    data = load_ec3d()
    path, frame = first_usable_frame(data)
    joint_names = common_joint_names(frame, ALL_CAMERAS)
    print(f"Frame: {path}")
    print(f"Cameras: {ALL_CAMERAS}")
    print(f"Common joints: {len(joint_names)}")

    # diagnostics = diagnose_camera_models(frame, data["params"])

    # print_camera_diagnostics(diagnostics)

    eligible_joints = [joint_name for joint_name in joint_names if joint_name in frame["3D_gt"]]

    results = [ triangulate_joint(
        frame,
        data["params"],
        joint_name,
        ALL_CAMERAS,
    ) for joint_name in eligible_joints 
    ] 

    mpjpe = np.mean([result["error_3d"] for result in results])

    print(f"MPJPE: {mpjpe:.4f} m ({mpjpe * 1000:.1f} mm)")


if __name__ == "__main__":
    main()

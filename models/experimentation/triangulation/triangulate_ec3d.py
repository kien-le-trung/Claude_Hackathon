"""Lesson 5: apply your geometry implementation to one EC3D frame."""

from __future__ import annotations

import numpy as np

from config import START_CAMERAS
from data import common_joint_names, first_usable_frame, load_ec3d
from geometry import (
    projection_matrix,
    reprojection_errors,
    triangulate_point_dlt,
)


def observation_xy(observation) -> np.ndarray:
    """Extract `(u, v)` from one EC3D `2D_op` joint observation.

    Inspect the value in `inspect_ec3d.py`, document its schema, and implement
    this conversion. Preserve confidence separately if one is included.
    """
    raise NotImplementedError("Lesson 5: inspect and decode a 2D_op value")


def ground_truth_xyz(observation) -> np.ndarray:
    """Extract metric XYZ from one EC3D `3D_gt` joint value."""
    raise NotImplementedError("Lesson 5: inspect and decode a 3D_gt value")


def camera_projection(parameters: dict) -> np.ndarray:
    """Build P after confirming the stored extrinsic convention."""
    intrinsics = parameters["intrinsics"]
    extrinsics = parameters["extrinsics"]
    # Decide whether these stored R/t already map world to camera. Validate that
    # decision through reprojection and positive camera depth.
    return projection_matrix(intrinsics["K"], extrinsics["R"], extrinsics["t"])


def main() -> None:
    data = load_ec3d()
    path, frame = first_usable_frame(data)
    joint_names = common_joint_names(frame, START_CAMERAS)
    print(f"Frame: {path}")
    print(f"Cameras: {START_CAMERAS}")
    print(f"Common joints: {len(joint_names)}")

    projections = [
        camera_projection(data["params"][camera_id])
        for camera_id in START_CAMERAS
    ]
    errors_3d = []
    for joint_name in joint_names:
        observations = [
            observation_xy(frame["2D_op"][camera_id][joint_name])
            for camera_id in START_CAMERAS
        ]
        estimate = triangulate_point_dlt(projections, observations)
        pixel_errors = reprojection_errors(estimate, projections, observations)
        if joint_name in frame["3D_gt"]:
            truth = ground_truth_xyz(frame["3D_gt"][joint_name])
            errors_3d.append(np.linalg.norm(estimate - truth))
        print(
            f"{joint_name:>20}: XYZ={np.round(estimate, 3)}, "
            f"reprojection={np.round(pixel_errors, 2)} px"
        )

    if errors_3d:
        print(f"\nFrame MPJPE: {np.mean(errors_3d):.3f} ground-truth units")


if __name__ == "__main__":
    main()

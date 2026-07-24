r"""Lesson 1 baseline: inspect one MM-Fit 3D pose with Open3D.

Run from the SquatSpot repository root:

    C:\tmp\sso3d\Scripts\python.exe models\experimentation\notebooks\open3d_learn.py

This file is deliberately small. Extend it as you work through the questions in
docs/OPEN3D_PRODUCT_LEARNING_PLAN.md instead of treating it as product code.
"""

from pathlib import Path

import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parents[3]
POSE_PATH = PROJECT_ROOT / "assets" / "data" / "mm-fit" / "w00" / "w00_pose_3d.npy"
FRAME_INDEX = 4040

# Defining joint name
JOINT_NAME = {
    "pelvis":0,

}

# mapping of point clouds to form human skeleton
BONE_PAIRS = [
    (0, 1),   # pelvis -> right hip
    (1, 2),   # right hip -> right knee
    (2, 3),   # right knee -> right ankle

    (0, 4),   # pelvis -> left hip
    (4, 5),   # left hip -> left knee
    (5, 6),   # left knee -> left ankle

    (0, 7),   # pelvis -> spine
    (7, 8),   # spine -> neck
    (8, 9),   # neck -> head
    (9, 10),  # head -> head site

    (8, 11),  # neck -> left shoulder
    (11, 12), # left shoulder -> left elbow
    (12, 13), # left elbow -> left wrist

    (8, 14),  # neck -> right shoulder
    (14, 15), # right shoulder -> right elbow
    (15, 16), # right elbow -> right wrist
]


def load_pose_frame(path: Path, frame_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Load the sequence and return one frame's joints as (joint, coordinate)."""
    poses = np.load(path)

    if poses.ndim != 3 or poses.shape[0] != 3:
        raise ValueError(
            "Expected MM-Fit pose layout (coordinate, frame, joint); "
            f"received {poses.shape}."
        )
    if not 0 <= frame_index < poses.shape[1]:
        raise IndexError(
            f"Frame {frame_index} is outside the valid range 0-{poses.shape[1] - 1}."
        )

    # MM-Fit stores one frame as (coordinate, row). Open3D expects each point
    # to occupy one row, so the frame must become (row, coordinate).
    frame_rows = poses[:, frame_index, :].T

    # Row 0 is MM-Fit's repeated timestamp, not an anatomical joint. Keeping it
    # would place a point thousands of units away and make the pose look tiny.
    timestamp_row = frame_rows[0]
    if not np.allclose(timestamp_row, timestamp_row[0]):
        raise ValueError(f"Unexpected timestamp row: {timestamp_row}")

    joint_points = frame_rows[1:]
    return poses, joint_points


def center_on_pelvis(joint_points: np.ndarray) -> np.ndarray:
    pelvis_coord = joint_points[JOINT_NAME["pelvis"]]
    centered_array = joint_points - pelvis_coord
    return centered_array


def pelvis_height(joint_points: np.ndarray) -> float:
    pelvis_z = joint_points[0, 2]
    right_ankle_z = joint_points[3, 2]
    left_ankle_z = joint_points[6, 2]
    ankle_midpoint_z = 0.5 * (right_ankle_z + left_ankle_z)
    return float(pelvis_z - ankle_midpoint_z)


def print_pose_summary(poses: np.ndarray, points: np.ndarray) -> None:
    """Print evidence we will use to infer axes, scale, and units."""
    print(f"Sequence shape: {poses.shape}")
    print(f"Sequence dtype: {poses.dtype}")
    print(f"Selected frame: {FRAME_INDEX}")
    print(f"Joint point matrix shape: {points.shape}")

    for coordinate_index, values in enumerate(points.T):
        print(
            f"Coordinate {coordinate_index}: "
            f"min={values.min():.4f}, max={values.max():.4f}, "
            f"span={np.ptp(values):.4f}"
        )

def create_point_cloud(joint_positions: np.ndarray):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(joint_positions) # convert to vector
    point_cloud.paint_uniform_color([0.1, 0.6, 0.9])
    return point_cloud


def create_skeleton(joint_positions, bone_pairs):
    skeleton = o3d.geometry.LineSet()

    # Conver to open3D expected format
    skeleton.points = o3d.utility.Vector3dVector(joint_positions)
    skeleton.lines = o3d.utility.Vector2iVector(bone_pairs)
    bone_colors = np.tile(
        [0.9, 0.2, 0.2],
        (len(bone_pairs), 1),
    )

    skeleton.colors = o3d.utility.Vector3dVector(bone_colors)

    return skeleton


def visualize(points: np.ndarray) -> None:
    """Display the joints as points beside an Open3D coordinate frame."""
    point_cloud = create_point_cloud(points)
    skeleton = create_skeleton(points, BONE_PAIRS)

    # Adjust this size after forming a hypothesis about the dataset's units.
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=250.0)

    o3d.visualization.draw_geometries(
        [point_cloud, skeleton, coordinate_frame],
        window_name=f"MM-Fit w00, frame {FRAME_INDEX}",
    )


def main() -> None:
    poses, points = load_pose_frame(POSE_PATH, 4500)
    # print_pose_summary(poses, points)

    # Learning checkpoints before adding skeleton connections:
    # TODO: Identify which coordinate appears vertical in the viewer.
    # TODO: Form and test a hypothesis about the coordinate units.
    # TODO: Compare this frame with frames 4040 and 4500.
    # TODO: Explain why points alone do not encode human anatomy.
    centered_points = center_on_pelvis(points)
    print(pelvis_height(centered_points))
    visualize(points)

    # poses, points = load_pose_frame(POSE_PATH, 4270)
    # print_pose_summary(poses, points)
    # visualize(points)


if __name__ == "__main__":
    main()

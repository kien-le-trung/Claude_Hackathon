import json
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d

from config import OUTPUT_DIR
from extract_mediapipe import load_extracted_landmarks

LEFT_ANKLE = 27
RIGHT_ANKLE = 28
LEFT_HEEL = 29
RIGHT_HEEL = 30
LEFT_FOOT_INDEX = 31
RIGHT_FOOT_INDEX = 32
LEFT_HIP = 23
RIGHT_HIP = 24

FOOT_INDICES = [
    LEFT_ANKLE,
    RIGHT_ANKLE,
    LEFT_HEEL,
    RIGHT_HEEL,
    LEFT_FOOT_INDEX,
    RIGHT_FOOT_INDEX,
]

BONE_PAIRS = [
    # Torso
    (11, 12),  # shoulders
    (11, 23),  # left shoulder to left hip
    (12, 24),  # right shoulder to right hip
    (23, 24),  # hips

    # Left arm
    (11, 13),
    (13, 15),

    # Right arm
    (12, 14),
    (14, 16),

    # Left leg
    (23, 25),
    (25, 27),
    (27, 29),
    (29, 31),

    # Right leg
    (24, 26),
    (26, 28),
    (28, 30),
    (30, 32),

    # Optional ankle-to-toe connections
    (27, 31),
    (28, 32),
]


def landmarks_to_array(record: dict) -> np.ndarray:
    return np.asarray(
        [
            [
                landmark["x"],
                landmark["y"],
                landmark["z"],
            ]
            for landmark in record["world_landmarks"]
        ],
        dtype=np.float64,
    )

# Map MediaPipe coordinates to display coordinates (X, Y, Z) -> (X, Z, -Y)
def convert_axes(points: np.ndarray) -> np.ndarray:
    converted = points[:, [0, 2, 1]].copy()
    converted[:, 2] *= -1.0
    return converted


def split_contiguous_sequences(records: list[dict]) -> list[list[dict]]:
    if not records:
        return []

    sequences = [[records[0]]]

    for previous, current in zip(records, records[1:]):
        if current["frame_id"] != previous["frame_id"] + 1:
            sequences.append([])
        sequences[-1].append(current)

    return sequences


def create_joint_cloud(points):
    cloud = o3d.geometry.PointCloud()

    cloud.points = o3d.utility.Vector3dVector(
        points
    )
    cloud.paint_uniform_color([
        0.49,
        0.23,
        0.93,
    ])
    return cloud


def create_bone_lines(points, bone_pairs):
    skeleton = o3d.geometry.LineSet()
    skeleton.points = o3d.utility.Vector3dVector(points)
    skeleton.lines = o3d.utility.Vector2iVector(bone_pairs)
    skeleton.paint_uniform_color([0.14, 0.07, 0.25])
    return skeleton


def reconstruct_video() -> None:
    width = 960
    height = 720
    fps = 30.0

    records = [
        record
        for record in load_extracted_landmarks()
        if record.get("detected") and record.get("world_landmarks")
    ]
    sequences = split_contiguous_sequences(records)
    if not sequences:
        raise ValueError("No detected MediaPipe skeletons are available to render")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for sequence_index, sequence in enumerate(sequences, start=1):
        frames = []
        for record in sequence:
            joints = convert_axes(landmarks_to_array(record))

            # MediaPipe world coordinates are hip-centered. Reposition each
            # frame so the lowest foot landmark sits on a common floor while
            # retaining the visible vertical motion of the pelvis.
            pelvis = (joints[LEFT_HIP] + joints[RIGHT_HIP]) / 2.0
            foot_center = joints[FOOT_INDICES].mean(axis=0)
            floor_height = joints[FOOT_INDICES, 2].min()
            joints[:, 0] -= pelvis[0]
            joints[:, 1] -= foot_center[1]
            joints[:, 2] -= floor_height
            frames.append((int(record["frame_id"]), joints))

        first_frame_id, first_joints = frames[0]
        joint_cloud = create_joint_cloud(first_joints)
        skeleton = create_bone_lines(first_joints, BONE_PAIRS)
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.25,
            origin=[0.0, 0.0, 0.0],
        )
        floor = o3d.geometry.TriangleMesh.create_box(width=2.0, height=2.0, depth=0.01)
        floor.translate([-1.0, -1.0, -0.01])
        floor.paint_uniform_color([0.88, 0.86, 0.92])
        floor.compute_vertex_normals()

        visualizer = o3d.visualization.Visualizer()
        created = visualizer.create_window(
            window_name=f"Squat reconstruction {sequence_index}",
            width=960,
            height=720,
            visible=True,
        )
        if not created:
            raise RuntimeError("Open3D could not create a rendering window")

        visualizer.add_geometry(joint_cloud)
        visualizer.add_geometry(skeleton)
        visualizer.add_geometry(floor)
        visualizer.add_geometry(coordinate_frame)

        render_options = visualizer.get_render_option()
        render_options.background_color = np.array([0.97, 0.96, 1.0])
        render_options.point_size = 8.0
        render_options.line_width = 4.0

        view = visualizer.get_view_control()
        view.set_front([0.0, -1.0, 0.15])
        view.set_lookat([0.0, 0.0, 0.8])
        view.set_up([0.0, 0.0, 1.0])
        view.set_zoom(0.65)

        output_path = OUTPUT_DIR / f"w00_squat_{sequence_index}.mp4"
        writer = None
        encoded_size = None

        try:
            for frame_id, joints in frames:
                joint_cloud.points = o3d.utility.Vector3dVector(joints)
                skeleton.points = o3d.utility.Vector3dVector(joints)
                visualizer.update_geometry(joint_cloud)
                visualizer.update_geometry(skeleton)

                if not visualizer.poll_events():
                    raise RuntimeError("Open3D rendering window was closed")
                visualizer.update_renderer()

                captured = np.asarray(
                    visualizer.capture_screen_float_buffer(do_render=True)
                )
                rgb = np.clip(captured * 255.0, 0, 255).astype(np.uint8)

                # Windows display scaling can make Open3D return a physical
                # framebuffer larger than the requested logical window size.
                captured_height, captured_width = rgb.shape[:2]
                current_size = (
                    captured_width - captured_width % 2,
                    captured_height - captured_height % 2,
                )
                if writer is None:
                    encoded_size = current_size
                    writer = cv2.VideoWriter(
                        str(output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        encoded_size,
                    )
                    if not writer.isOpened():
                        raise RuntimeError(
                            f"Could not create output video: {output_path}"
                        )
                    print(
                        f"Open3D framebuffer: {captured_width}x{captured_height}; "
                        f"encoding at {encoded_size[0]}x{encoded_size[1]}"
                    )
                elif current_size != encoded_size:
                    raise RuntimeError(
                        "Open3D framebuffer size changed while rendering: "
                        f"expected {encoded_size}, received {current_size} "
                        f"at source frame {frame_id}. Keep the window size fixed."
                    )

                encoded_width, encoded_height = encoded_size
                rgb = rgb[:encoded_height, :encoded_width]
                writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        finally:
            if writer is not None:
                writer.release()
            visualizer.destroy_window()

        print(
            f"Wrote {output_path} "
            f"({len(frames)} frames, source frames "
            f"{first_frame_id}-{frames[-1][0]})"
        )


if __name__ == "__main__":
    reconstruct_video()

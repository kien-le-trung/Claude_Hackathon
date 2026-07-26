from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

from .artifacts import _safe_child


POSE_CONNECTIONS = (
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
)
LEFT_HIP, RIGHT_HIP = 23, 24
FOOT_INDICES = (27, 28, 29, 30, 31, 32)
RENDER_WIDTH = 960
RENDER_HEIGHT = 720


def media_destination(artifact_root: Path, video_id: str, name: str) -> tuple[Path, str]:
    relative = Path(video_id) / name
    destination = _safe_child(artifact_root, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination, relative.as_posix()


def normalized_skeleton_points(pose: dict) -> np.ndarray:
    """Convert MediaPipe world coordinates to a stable floor-anchored display pose."""
    landmarks = pose.get("world_landmarks") or []
    if len(landmarks) != 33:
        raise ValueError(f"Expected 33 world landmarks, received {len(landmarks)}")
    source = np.asarray(
        [[item["x"], item["y"], item["z"]] for item in landmarks],
        dtype=np.float64,
    )
    if not np.isfinite(source).all():
        raise ValueError("World landmarks contain non-finite coordinates")

    points = source[:, [0, 2, 1]].copy()
    points[:, 2] *= -1.0
    pelvis = (points[LEFT_HIP] + points[RIGHT_HIP]) / 2.0
    foot_center = points[list(FOOT_INDICES)].mean(axis=0)
    floor_height = points[list(FOOT_INDICES), 2].min()
    points[:, 0] -= pelvis[0]
    points[:, 1] -= foot_center[1]
    points[:, 2] -= floor_height
    return points


def reconstruction_frames(extraction_frames: Iterable[dict]) -> list[np.ndarray | None]:
    """Preserve the sampling timeline; missing poses become blank rendered frames."""
    output: list[np.ndarray | None] = []
    for frame in extraction_frames:
        poses = frame.get("poses") or []
        if not poses:
            output.append(None)
            continue
        try:
            output.append(normalized_skeleton_points(poses[0]))
        except ValueError:
            output.append(None)
    return output


def normalize_source_video(source: Path, destination: Path) -> None:
    """Transcode to silent H.264/yuv420p MP4 and replace the destination atomically."""
    import imageio_ffmpeg

    temporary = destination.with_name(f"{destination.stem}.tmp{destination.suffix}")
    temporary.unlink(missing_ok=True)
    command = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-i", str(source),
        "-map", "0:v:0",
        "-an",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(temporary),
    ]
    try:
        subprocess.run(
            command,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        temporary.replace(destination)
    except subprocess.CalledProcessError as exc:
        temporary.unlink(missing_ok=True)
        detail = (exc.stderr or "FFmpeg failed").strip().splitlines()[-1]
        raise RuntimeError(f"Source video normalization failed: {detail}") from exc


class Open3DFrameRenderer:
    """Windows OpenGL/GLFW renderer for an unsmoothed pose skeleton."""

    def __init__(self, width: int = RENDER_WIDTH, height: int = RENDER_HEIGHT):
        import cv2
        import open3d as o3d

        self.cv2 = cv2
        self.o3d = o3d
        self.width = width
        self.height = height
        self.visualizer = o3d.visualization.Visualizer()
        created = self.visualizer.create_window(
            window_name="SquatSpot reconstruction",
            width=width,
            height=height,
            visible=False,
        )
        if not created:
            raise RuntimeError("Open3D could not create its Windows rendering context")

        initial = np.zeros((33, 3), dtype=np.float64)
        self.cloud = o3d.geometry.PointCloud()
        self.cloud.points = o3d.utility.Vector3dVector(initial)
        self.cloud.paint_uniform_color([0.49, 0.23, 0.93])
        self.lines = o3d.geometry.LineSet()
        self.lines.points = o3d.utility.Vector3dVector(initial)
        self.lines.lines = o3d.utility.Vector2iVector(POSE_CONNECTIONS)
        self.lines.paint_uniform_color([0.14, 0.07, 0.25])
        floor = o3d.geometry.TriangleMesh.create_box(2.0, 2.0, 0.01)
        floor.translate([-1.0, -1.0, -0.01])
        floor.paint_uniform_color([0.88, 0.86, 0.92])
        floor.compute_vertex_normals()

        self.visualizer.add_geometry(self.cloud)
        self.visualizer.add_geometry(self.lines)
        self.visualizer.add_geometry(floor)
        options = self.visualizer.get_render_option()
        options.background_color = np.array([0.97, 0.96, 1.0])
        options.point_size = 8.0
        options.line_width = 4.0
        view = self.visualizer.get_view_control()
        view.set_front([0.0, -1.0, 0.15])
        view.set_lookat([0.0, 0.0, 0.75])
        view.set_up([0.0, 0.0, 1.0])
        view.set_zoom(0.65)

    def render(self, points: np.ndarray | None) -> np.ndarray:
        displayed = (
            points
            if points is not None
            else np.full((33, 3), 1_000_000.0, dtype=np.float64)
        )
        self.cloud.points = self.o3d.utility.Vector3dVector(displayed)
        self.lines.points = self.o3d.utility.Vector3dVector(displayed)
        self.visualizer.update_geometry(self.cloud)
        self.visualizer.update_geometry(self.lines)
        if not self.visualizer.poll_events():
            raise RuntimeError("Open3D Windows rendering context closed unexpectedly")
        self.visualizer.update_renderer()
        captured = np.asarray(
            self.visualizer.capture_screen_float_buffer(do_render=True)
        )
        rgb = np.clip(captured * 255.0, 0, 255).astype(np.uint8)
        if rgb.shape[:2] != (self.height, self.width):
            rgb = self.cv2.resize(
                rgb,
                (self.width, self.height),
                interpolation=self.cv2.INTER_AREA,
            )
        return rgb

    def close(self) -> None:
        self.visualizer.destroy_window()


def render_reconstruction_video(
    extraction_frames: Iterable[dict],
    destination: Path,
    fps: float,
    *,
    renderer_factory: Callable[[], object] | None = None,
) -> int:
    """Render the complete sampled timeline and atomically write an H.264 MP4."""
    import imageio_ffmpeg

    frames = reconstruction_frames(extraction_frames)
    if not frames:
        raise ValueError("No sampled frames are available for reconstruction")
    if fps <= 0:
        raise ValueError("Reconstruction FPS must be positive")

    temporary = destination.with_name(f"{destination.stem}.tmp{destination.suffix}")
    temporary.unlink(missing_ok=True)
    renderer = (renderer_factory or Open3DFrameRenderer)()
    writer = imageio_ffmpeg.write_frames(
        str(temporary),
        (RENDER_WIDTH, RENDER_HEIGHT),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-movflags", "+faststart", "-an"],
    )
    writer.send(None)
    try:
        for pose in frames:
            image = np.asarray(renderer.render(pose), dtype=np.uint8)
            if image.shape != (RENDER_HEIGHT, RENDER_WIDTH, 3):
                raise ValueError(f"Renderer returned unexpected image shape {image.shape}")
            writer.send(np.ascontiguousarray(image))
        writer.close()
        temporary.replace(destination)
    except Exception:
        try:
            writer.close()
        finally:
            temporary.unlink(missing_ok=True)
        raise
    finally:
        close = getattr(renderer, "close", None)
        if close is not None:
            close()
    return len(frames)

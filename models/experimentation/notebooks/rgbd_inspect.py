r"""Lesson 1: audit the MM-Fit RGB and depth videos before reconstruction.

Run from the SquatSpot repository root:

    .\venv\Scripts\python.exe models\experimentation\notebooks\rgbd_inspect.py

This script intentionally stops before creating an Open3D point cloud. Complete
the marked TODOs and understand the depth encoding and calibration first.
"""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import open3d as o3d


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "assets" / "data" / "mm-fit"
RGB_PATH = DATA_DIR / "w00_rgb.mp4"
DEPTH_PATH = DATA_DIR / "w00_depth.mp4"
FRAME_INDEX = 4040
VOXEL_SIZES = [0.1, 0.25, 0.5, 1.0]


@dataclass(frozen=True)
class VideoMetadata:
    path: Path
    width: int
    height: int
    fps: float
    frame_count: int
    fourcc: str

    @property
    def duration_seconds(self) -> float:
        if self.fps <= 0:
            raise ValueError(f"Invalid FPS for {self.path}: {self.fps}")
        return self.frame_count / self.fps


def decode_fourcc(value: float) -> str:
    """Convert OpenCV's numeric FourCC value into a readable four-character code."""
    integer_value = int(value)
    return "".join(chr((integer_value >> (8 * index)) & 0xFF) for index in range(4))


def inspect_video(path: Path) -> VideoMetadata:
    """Open a video, validate it, and return its basic stream metadata."""
    if not path.is_file():
        raise FileNotFoundError(f"Video not found: {path}")

    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"OpenCV could not open: {path}")

        return VideoMetadata(
            path=path,
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            fourcc=decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC)),
        )
    finally:
        capture.release()


def read_frame(path: Path, frame_index: int) -> np.ndarray:
    """Read one zero-based video frame and fail clearly if seeking or decoding fails."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"OpenCV could not open: {path}")

        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        if not success or frame is None:
            raise ValueError(f"Could not decode frame {frame_index} from {path}")

        return frame
    finally:
        capture.release()


def print_metadata(label: str, metadata: VideoMetadata) -> None:
    print(f"\n{label}")
    print(f"  Path: {metadata.path}")
    print(f"  Resolution: {metadata.width} x {metadata.height}")
    print(f"  FPS: {metadata.fps:.3f}")
    print(f"  Frame count: {metadata.frame_count}")
    print(f"  Duration: {metadata.duration_seconds:.3f} seconds")
    print(f"  Codec/FourCC: {metadata.fourcc!r}")


def print_frame_summary(label: str, frame: np.ndarray) -> None:
    """Print statistics shared by RGB and depth frames."""
    print(f"\n{label} frame {FRAME_INDEX}")
    print(f"  Shape: {frame.shape}")
    print(f"  Data type: {frame.dtype}")
    print(f"  Minimum: {frame.min()}")
    print(f"  Maximum: {frame.max()}")
    print(f"  Mean: {frame.mean():.3f}")


def extract_depth_channel(depth_frame: np.ndarray) -> np.ndarray:
    """Validate a decoded grayscale depth frame and return one channel."""
    if depth_frame.ndim == 2:
        return depth_frame

    if depth_frame.ndim != 3 or depth_frame.shape[2] != 3:
        raise ValueError(f"Unexpected depth-frame shape: {depth_frame.shape}")

    channels_equal = (
        np.array_equal(depth_frame[:, :, 0], depth_frame[:, :, 1])
        and np.array_equal(depth_frame[:, :, 1], depth_frame[:, :, 2])
    )
    print(f"  Three decoded channels are equal: {channels_equal}")

    if not channels_equal:
        raise ValueError(
            "Depth channels differ. Do not select one channel until the encoding "
            "has been investigated."
        )

    return depth_frame[:, :, 0]


def analyze_depth(depth: np.ndarray) -> None:
    """Print the depth statistics required for the initial data audit."""
    # TODO: Calculate and print:
    #   - the number of unique values
    #   - the percentage of pixels equal to zero
    #   - percentiles 1, 25, 50, 75, and 99
    #
    # Useful NumPy concepts to look up:
    #   np.unique
    #   np.count_nonzero
    #   ndarray.size
    #   np.percentile
    unique_values = np.unique(depth)
    zero_count = depth.size - np.count_nonzero(depth)
    zero_percentage = 100.0 * zero_count / depth.size
    percentile_levels = [1, 25, 50, 75, 99]
    percentile_values = np.percentile(depth, percentile_levels)

    print(f"  Number of unique values: {len(unique_values)}")
    print(f"  Zero-valued pixels: {zero_count:,} ({zero_percentage:.3f}%)")
    print("  Percentiles:")
    for level, value in zip(percentile_levels, percentile_values):
        print(f"    {level:>2}%: {value:.3f}")


def create_depth_visualization(depth: np.ndarray) -> np.ndarray:
    """Create an eight-bit color visualization without changing metric meaning."""
    # TODO:
    #   1. Normalize or contrast-stretch a COPY of `depth` for display.
    #   2. Convert that display image to uint8 if necessary.
    #   3. Apply an OpenCV color map.
    #
    # Do not return a resized image and do not overwrite the original depth data.
    minimum = float(depth.min())
    maximum = float(depth.max())
    value_range = maximum - minimum

    if value_range == 0:
        display_depth = np.zeros_like(depth, dtype=np.uint8)
    else:
        # Work in floating point so unsigned integer arithmetic cannot wrap.
        # This inversion makes smaller decoded values appear brighter.
        normalized_depth = (maximum - depth.astype(np.float32)) / value_range
        display_depth = np.round(normalized_depth * 255).astype(np.uint8)

    return cv2.applyColorMap(display_depth, cv2.COLORMAP_TURBO)


def display_frames(
    rgb_frame: np.ndarray,
    depth: np.ndarray,
    depth_visualization: np.ndarray,
) -> None:
    """Display the observations independently without implying spatial alignment."""
    cv2.imshow("MM-Fit RGB frame", rgb_frame)
    cv2.imshow("MM-Fit raw decoded depth", depth)
    cv2.imshow("MM-Fit depth visualization", depth_visualization)
    print("\nPress any key in an image window to close the displays.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def back_project(
        u: float,
        v: float,
        raw_depth: float,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        depth_scale: float,
) -> np.ndarray:
    z = raw_depth / depth_scale
    x = (u - cx)*z/fx
    y = (v - cy)*z/fy
    return np.array([x, y, z])


def backproject_frame(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float,
    stride: int = 1,
) -> np.ndarray:
    points = []

    for v in range(0, depth.shape[0], stride):
        for u in range(0, depth.shape[1], stride):
            raw_depth = float(depth[v, u])

            if raw_depth == 0:
                continue

            point = back_project(
                u,
                v,
                raw_depth,
                fx,
                fy,
                cx,
                cy,
                depth_scale,
            )
            points.append(point)

    return np.asarray(points, dtype=np.float64)


def main() -> None:
    rgb_metadata = inspect_video(RGB_PATH)
    depth_metadata = inspect_video(DEPTH_PATH)

    # print_metadata("RGB VIDEO", rgb_metadata)
    # print_metadata("DEPTH VIDEO", depth_metadata)
    # print(
    #     "\nDuration difference: "
    #     f"{abs(rgb_metadata.duration_seconds - depth_metadata.duration_seconds):.3f} "
    #     "seconds"
    # )

    rgb_frame = read_frame(RGB_PATH, FRAME_INDEX)
    depth_frame = read_frame(DEPTH_PATH, FRAME_INDEX)

    # print_frame_summary("RGB", rgb_frame)
    # print_frame_summary("Depth", depth_frame)

    depth = extract_depth_channel(depth_frame)
    analyze_depth(depth)

    height, width = depth.shape

    # Hypothetical values
    fx = 525.0
    fy = 525.0
    cx = (width - 1) / 2
    cy = (height - 1) / 2
    depth_scale = 1.0

    points = backproject_frame(
        depth,
        fx,
        fy,
        cx,
        cy,
        depth_scale,
        stride=1,
    )

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.paint_uniform_color([0.2,0.7,0.9])
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0)

    print(f"Before Voxel downsampling: {len(point_cloud.points)} points")
    for voxel_size in VOXEL_SIZES:
        downsampled_cloud = point_cloud.voxel_down_sample(voxel_size)
        print(f"After Voxel downsampling size {voxel_size}: {len(downsampled_cloud.points)} points")
        o3d.visualization.draw_geometries(
            [downsampled_cloud, coordinate_frame],
            window_name=f"downsample size {voxel_size}"
        )

    # o3d.visualization.draw_geometries(
    #     [point_cloud, coordinate_frame],
    #     window_name="MM-fit provisional depth cloud"
    # )

    # depth_visualization = create_depth_visualization(depth)
    # display_frames(rgb_frame, depth, depth_visualization)

    # maximum_point_count = depth.shape[0] * depth.shape[1]
    # print(f"\nMaximum possible raw point count: {maximum_point_count:,}")


if __name__ == "__main__":
    main()

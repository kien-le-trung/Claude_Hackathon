import sys
from types import SimpleNamespace

import numpy as np

from app.media import (
    normalized_skeleton_points,
    reconstruction_frames,
    render_reconstruction_video,
    skeleton_artifact,
    smooth_joint_sequence,
)


def pose():
    landmarks = [
        {"x": index / 100, "y": index / 200, "z": -index / 300}
        for index in range(33)
    ]
    return {"world_landmarks": landmarks}


def test_skeleton_points_map_axes_and_anchor_lowest_foot():
    points = normalized_skeleton_points(pose())

    assert points.shape == (33, 3)
    assert np.isclose(points[[27, 28, 29, 30, 31, 32], 2].min(), 0.0)
    assert np.isclose((points[23, 0] + points[24, 0]) / 2, 0.0)


def test_reconstruction_timeline_preserves_missing_pose_frames():
    frames = reconstruction_frames([
        {"poses": [pose()]},
        {"poses": []},
        {"poses": [pose()]},
    ])

    assert len(frames) == 3
    assert frames[0].shape == (33, 3)
    assert frames[1] is None
    assert frames[2].shape == (33, 3)


def test_smoothing_preserves_quadratic_motion_and_reduces_noise():
    rng = np.random.default_rng(7)
    time = np.linspace(-1.0, 1.0, 31)
    clean = np.broadcast_to(time[:, None, None] ** 2, (31, 33, 3))
    noisy = clean + rng.normal(0.0, 0.03, clean.shape)

    exact = smooth_joint_sequence(clean)
    smoothed = smooth_joint_sequence(noisy)

    np.testing.assert_allclose(exact, clean, atol=1e-12)
    assert np.mean((smoothed - clean) ** 2) < np.mean((noisy - clean) ** 2)


def test_smoothing_preserves_constant_motion_and_short_runs():
    constant = np.full((12, 33, 3), 0.25)
    short = np.arange(2 * 33 * 3, dtype=np.float64).reshape(2, 33, 3)

    np.testing.assert_allclose(smooth_joint_sequence(constant), constant)
    np.testing.assert_array_equal(smooth_joint_sequence(short), short)


def test_smoothing_does_not_cross_missing_pose_gap():
    first = pose()
    second = pose()
    second["world_landmarks"][0]["x"] = 100.0
    frames = [
        {"timestamp_ms": 0, "poses": [first]},
        {"timestamp_ms": 100, "poses": []},
        {"timestamp_ms": 200, "poses": [second]},
    ]

    reconstructed = reconstruction_frames(frames)

    assert reconstructed[0] is not None
    assert reconstructed[1] is None
    assert reconstructed[2] is not None
    assert reconstructed[2][0, 0] > 90.0


def test_skeleton_artifact_preserves_timestamps_and_metadata():
    frames = [
        {"timestamp_ms": 0, "poses": [pose()]},
        {"timestamp_ms": 100, "poses": []},
    ]

    artifact = skeleton_artifact(frames, 10.0)

    assert artifact["schema_version"] == 1
    assert artifact["smoothing"] == {
        "method": "savitzky_golay",
        "window_length": 7,
        "polynomial_order": 2,
    }
    assert [frame["timestamp_ms"] for frame in artifact["frames"]] == [0, 100]
    assert artifact["frames"][0]["points"] is not None
    assert artifact["frames"][1]["points"] is None


def test_renderer_writes_one_video_frame_per_sample(monkeypatch, tmp_path):
    sent = []

    class Writer:
        def __init__(self, path):
            self.path = path

        def send(self, value):
            if value is not None:
                sent.append(value)

        def close(self):
            from pathlib import Path
            Path(self.path).write_bytes(b"mp4")

    fake_ffmpeg = SimpleNamespace(
        write_frames=lambda path, *_args, **_kwargs: Writer(path)
    )
    monkeypatch.setitem(sys.modules, "imageio_ffmpeg", fake_ffmpeg)

    class Renderer:
        def render(self, _points):
            return np.zeros((720, 960, 3), dtype=np.uint8)

    destination = tmp_path / "reconstruction.mp4"
    count = render_reconstruction_video(
        [{"poses": [pose()]}, {"poses": []}, {"poses": [pose()]}],
        destination,
        10.0,
        renderer_factory=Renderer,
    )

    assert count == 3
    assert len(sent) == 3
    assert destination.read_bytes() == b"mp4"

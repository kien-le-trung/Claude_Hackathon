import sys
from types import SimpleNamespace

import numpy as np

from app.media import (
    normalized_skeleton_points,
    reconstruction_frames,
    render_reconstruction_video,
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

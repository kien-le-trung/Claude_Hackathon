from pathlib import Path

import cv2
import numpy as np
import pytest

from app.artifacts import (
    artifact_path,
    create_event_artifacts,
    delete_video_artifacts,
)


def _video(path: Path):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (1280, 720)
    )
    assert writer.isOpened()
    for index in range(4):
        frame = np.full((720, 1280, 3), 30 + index * 20, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _event():
    landmarks = [
        {
            "x": 0.25 + (index % 5) * 0.1,
            "y": 0.2 + (index // 5) * 0.08,
            "visibility": 0.9,
            "presence": 0.9,
        }
        for index in range(33)
    ]
    return {
        "id": "e3d88af1-8f6f-47f2-90ce-5c14df605fc7",
        "type": "bad_back",
        "confidence": 0.91,
        "representative_frame_number": 2,
        "_representative_pose": {"landmarks": landmarks},
        "frame_image_path": None,
        "artifact_warning": None,
    }


def test_representative_frame_is_annotated_resized_and_deleted(tmp_path):
    source = tmp_path / "source.avi"
    _video(source)
    root = tmp_path / "artifacts"
    event = _event()
    create_event_artifacts(source, [event], root, "video-id")

    stored = artifact_path(root, event["frame_image_path"])
    image = cv2.imread(str(stored))
    assert stored.is_file()
    assert image.shape[1] == 960
    assert event["artifact_warning"] is None

    delete_video_artifacts(root, "video-id")
    assert not stored.exists()


def test_artifact_path_rejects_escape(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        artifact_path(tmp_path, "../outside.webp")

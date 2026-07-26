import sys
from pathlib import Path

import numpy as np
import pytest


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EXPERIMENT_DIR))

from alignment import (  # noqa: E402
    load_pose_sequence,
    pose_row_for_rgb_frame,
    squat_frame_ids,
)


def test_pose_rows_are_joined_to_rgb_by_embedded_frame_id():
    frame_ids, joints, _ = load_pose_sequence()

    assert joints.shape == (63_918, 17, 3)
    assert frame_ids[0] == 4050
    assert frame_ids[-1] == 67967
    np.testing.assert_array_equal(np.diff(frame_ids), np.ones(len(frame_ids) - 1))
    assert pose_row_for_rgb_frame(4050, frame_ids) == 0
    assert pose_row_for_rgb_frame(4500, frame_ids) == 450


def test_missing_pre_pose_video_frame_is_not_silently_aligned():
    frame_ids, _, _ = load_pose_sequence()

    with pytest.raises(KeyError):
        pose_row_for_rgb_frame(4040, frame_ids)


def test_squat_frames_are_clipped_to_available_ground_truth():
    frame_ids = squat_frame_ids()

    assert frame_ids[0] == 4050
    assert frame_ids[-1] == 7685
    assert 4500 in frame_ids
    assert 5435 in frame_ids


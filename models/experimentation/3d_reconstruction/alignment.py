"""Inspect and load MM-Fit-to-RGB frame alignment for session w00."""

from __future__ import annotations

import csv
from dataclasses import dataclass

import cv2
import numpy as np

from config import LABELS_PATH, MMFIT_JOINT_NAMES, POSE_3D_PATH, RGB_PATH


@dataclass(frozen=True)
class VideoMetadata:
    frame_count: int
    fps: float

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps


@dataclass(frozen=True)
class ActivityRange:
    start_frame: int
    end_frame: int
    repetitions: int
    activity: str


def load_video_metadata() -> VideoMetadata:
    capture = cv2.VideoCapture(str(RGB_PATH))
    try:
        if not capture.isOpened():
            raise FileNotFoundError(f"Could not open {RGB_PATH}")
        return VideoMetadata(
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps=float(capture.get(cv2.CAP_PROP_FPS)),
        )
    finally:
        capture.release()


def load_pose_sequence(*, mmap: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return embedded RGB frame IDs, joints `(frame,joint,xyz)`, and raw data."""
    raw = np.load(POSE_3D_PATH, mmap_mode="r" if mmap else None)
    if raw.ndim != 3 or raw.shape[0] != 3 or raw.shape[2] != len(MMFIT_JOINT_NAMES) + 1:
        raise ValueError(f"Unexpected MM-Fit 3D pose shape: {raw.shape}")

    timestamp_rows = raw[:, :, 0]
    if not np.allclose(timestamp_rows, timestamp_rows[0:1, :]):
        raise ValueError("The timestamp/frame-ID value is not repeated across coordinates")

    frame_values = timestamp_rows[0]
    if not np.allclose(frame_values, np.round(frame_values)):
        raise ValueError("Embedded MM-Fit frame IDs are not integers")
    frame_ids = np.round(frame_values).astype(np.int64)
    if np.any(np.diff(frame_ids) <= 0):
        raise ValueError("Embedded MM-Fit frame IDs are not strictly increasing")

    joints = np.moveaxis(raw[:, :, 1:], 0, -1)
    return frame_ids, joints, raw


def load_activity_ranges() -> list[ActivityRange]:
    with LABELS_PATH.open(newline="", encoding="utf-8") as handle:
        return [
            ActivityRange(int(start), int(end), int(repetitions), activity)
            for start, end, repetitions, activity in csv.reader(handle)
        ]


def squat_frame_ids() -> np.ndarray:
    """Return available ground-truth frame IDs inside labeled squat ranges."""
    frame_ids, _, _ = load_pose_sequence()
    ranges = [item for item in load_activity_ranges() if item.activity == "squats"]
    mask = np.zeros(frame_ids.shape, dtype=bool)
    for item in ranges:
        mask |= (frame_ids >= item.start_frame) & (frame_ids <= item.end_frame)
    return frame_ids[mask]


def pose_row_for_rgb_frame(frame_id: int, frame_ids: np.ndarray) -> int:
    """Resolve by embedded ID; fail rather than silently choosing a nearby pose."""
    index = int(np.searchsorted(frame_ids, frame_id))
    if index >= len(frame_ids) or int(frame_ids[index]) != frame_id:
        raise KeyError(f"RGB frame {frame_id} has no MM-Fit 3D pose")
    return index


def main() -> None:
    video = load_video_metadata()
    frame_ids, joints, _ = load_pose_sequence()
    squats = squat_frame_ids()
    first_squat = next(item for item in load_activity_ranges() if item.activity == "squats")

    print(f"RGB frames: {video.frame_count}")
    print(f"RGB FPS: {video.fps:g}")
    print(f"RGB duration: {video.duration_seconds:.3f} seconds")
    print(f"Pose joint array: {joints.shape}")
    print(f"Pose frame IDs: {frame_ids[0]}..{frame_ids[-1]}")
    print(f"Pose frame-ID median step: {np.median(np.diff(frame_ids)):g}")
    print(f"First squat label: {first_squat.start_frame}..{first_squat.end_frame}")
    print(f"Comparable squat frames: {squats[0]}..{squats[-1]} ({len(squats)} total)")
    print(f"Pose row 0 maps to RGB frame {frame_ids[0]}")


if __name__ == "__main__":
    main()


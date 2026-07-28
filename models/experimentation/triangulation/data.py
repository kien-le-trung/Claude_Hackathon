"""Read-only helpers for navigating the trusted EC3D pickle."""

from __future__ import annotations

import pickle
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from config import EC3D_PICKLE


def load_ec3d(path: Path = EC3D_PICKLE) -> dict[str, Any]:
    """Load the trusted local EC3D artifact.

    Pickle is not safe for untrusted input. This function intentionally accepts
    only an explicit local path and performs no downloading.
    """
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"EC3D pickle not found: {resolved}")
    with resolved.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict) or not {"frames", "params"} <= data.keys():
        raise ValueError("Unexpected EC3D root schema; expected frames and params")
    return data


def iter_frames(data: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield `(path, frame)` across action/subject/trial/frame nesting."""
    for action, subjects in data["frames"].items():
        for subject, trials in subjects.items():
            for trial, frames in trials.items():
                for frame_id, frame in frames.items():
                    yield (
                        f"{action}/{subject}/{trial}/{frame_id}",
                        frame,
                    )


def first_usable_frame(
    data: dict[str, Any],
    minimum_views: int = 2,
    require_3d: bool = True,
) -> tuple[str, dict[str, Any]]:
    """Find an early frame suitable for a first triangulation exercise."""
    for path, frame in iter_frames(data):
        observations = frame.get("2D_op") or {}
        valid_views = sum(bool(points) for points in observations.values())
        if valid_views < minimum_views:
            continue
        if require_3d and not frame.get("3D_gt"):
            continue
        return path, frame
    raise ValueError("No EC3D frame satisfies the requested criteria")


def common_joint_names(
    frame: dict[str, Any],
    camera_ids: tuple[str, ...],
) -> list[str]:
    """Return joint names observed by every requested camera."""
    observations = frame.get("2D_op") or {}
    joint_sets = [
        set(observations.get(camera_id) or {})
        for camera_id in camera_ids
    ]
    if not joint_sets:
        return []
    common = set.intersection(*joint_sets)
    return sorted(common)


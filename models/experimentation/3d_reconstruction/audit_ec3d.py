"""Audit EC3D pickle contents for stereo-triangulation prerequisites.

Pickle files can execute code while loading. Run this only on a trusted local
artifact. The audit is read-only and does not extract images or write outputs.
"""

from __future__ import annotations

import argparse
import pickle
from collections import Counter, deque
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PATH = PROJECT_ROOT / "assets" / "data" / "ec3d" / "data.pickle"
MAX_VISITED_OBJECTS = 100_000
EXAMPLE_LIMIT = 8

CALIBRATION_TERMS = {
    "calibration", "camera_matrix", "intrinsic", "intrinsics", "extrinsic",
    "extrinsics", "distortion", "projection_matrix", "rotation", "translation",
}
TIME_TERMS = {
    "timestamp", "timestamps", "time", "frame", "frame_id", "frame_index",
    "frame_number",
}
VIEW_TERMS = {"camera", "camera_id", "view", "view_id", "front", "side"}
POINT_2D_TERMS = {
    "2d", "keypoints_2d", "joints_2d", "pose_2d", "landmarks_2d", "points_2d",
}
POINT_3D_TERMS = {
    "3d", "keypoints_3d", "joints_3d", "pose_3d", "landmarks_3d", "points_3d",
}


def describe(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return f"ndarray(shape={value.shape}, dtype={value.dtype})"
    if isinstance(value, dict):
        return f"dict(len={len(value)})"
    if isinstance(value, (list, tuple)):
        return f"{type(value).__name__}(len={len(value)})"
    return type(value).__name__


def normalized_key(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def matches_term(key: str, terms: set[str]) -> bool:
    return any(term == key or term in key for term in terms)


def audit_structure(root: Any) -> dict[str, Any]:
    queue = deque([("root", root)])
    visited_ids: set[int] = set()
    key_counts: Counter[str] = Counter()
    type_counts: Counter[str] = Counter()
    array_shapes: Counter[str] = Counter()
    examples: dict[str, list[str]] = {
        "calibration": [],
        "time": [],
        "view": [],
        "points_2d": [],
        "points_3d": [],
    }
    observation_view_ids: set[str] = set()
    calibrated_camera_ids: set[str] = set()
    truncated = False

    if isinstance(root, dict):
        params = root.get("params")
        if isinstance(params, dict):
            calibrated_camera_ids.update(str(key) for key in params)

    while queue:
        if len(visited_ids) >= MAX_VISITED_OBJECTS:
            truncated = True
            break
        path, value = queue.popleft()
        identity = id(value)
        if identity in visited_ids:
            continue
        visited_ids.add(identity)
        type_counts[type(value).__name__] += 1

        if isinstance(value, np.ndarray):
            array_shapes[f"{value.shape} {value.dtype}"] += 1
            continue

        if isinstance(value, dict):
            for key, child in value.items():
                key_name = normalized_key(key)
                key_counts[key_name] += 1
                child_path = f"{path}.{key}"
                categories = (
                    ("calibration", CALIBRATION_TERMS),
                    ("time", TIME_TERMS),
                    ("view", VIEW_TERMS),
                    ("points_2d", POINT_2D_TERMS),
                    ("points_3d", POINT_3D_TERMS),
                )
                for category, terms in categories:
                    if (
                        matches_term(key_name, terms)
                        and len(examples[category]) < EXAMPLE_LIMIT
                    ):
                        examples[category].append(
                            f"{child_path}: {describe(child)}"
                        )
                if matches_term(key_name, POINT_2D_TERMS) and isinstance(child, dict):
                    observation_view_ids.update(str(view_id) for view_id in child)
                queue.append((child_path, child))
        elif isinstance(value, (list, tuple)):
            # Inspect every structural element but avoid walking scalar payloads.
            for index, child in enumerate(value):
                if isinstance(child, (dict, list, tuple, np.ndarray)):
                    queue.append((f"{path}[{index}]", child))

    return {
        "visited_objects": len(visited_ids),
        "truncated": truncated,
        "key_counts": key_counts,
        "type_counts": type_counts,
        "array_shapes": array_shapes,
        "examples": examples,
        "observation_view_ids": observation_view_ids,
        "calibrated_camera_ids": calibrated_camera_ids,
    }


def print_counter(title: str, values: Counter[str], limit: int = 15) -> None:
    print(title)
    if not values:
        print("  (none)")
        return
    for name, count in values.most_common(limit):
        print(f"  {count:>6}  {name}")


def print_examples(title: str, values: list[str]) -> None:
    print(title)
    if not values:
        print("  (none found)")
        return
    for value in values:
        print(f"  {value}")


def triangulation_assessment(audit: dict[str, Any]) -> bool:
    keys = set(audit["key_counts"])
    has_calibration = any(matches_term(key, CALIBRATION_TERMS) for key in keys)
    has_time = any(matches_term(key, TIME_TERMS) for key in keys)
    view_ids = audit["observation_view_ids"]
    camera_ids = audit["calibrated_camera_ids"]
    has_views = len(view_ids) >= 2
    views_are_calibrated = has_views and view_ids.issubset(camera_ids)
    has_2d = any(matches_term(key, POINT_2D_TERMS) for key in keys)

    print("\nStereo-triangulation prerequisite screen")
    checks = {
        "Two or more identifiable camera views": has_views,
        "Shared frame/timestamp correspondence": has_time,
        "2D observations in each view": has_2d,
        "Camera calibration/projection parameters": has_calibration,
        "Observed view IDs have camera parameters": views_are_calibrated,
    }
    for label, passed in checks.items():
        print(f"  {'PASS' if passed else 'MISSING':<7} {label}")

    ready = all(checks.values())
    print("\nPreliminary verdict:")
    if ready:
        print(
            "  Candidate for stereo triangulation. The named fields exist, but "
            "their shapes, units, camera pairing, and reprojection consistency "
            "still require semantic validation."
        )
    else:
        missing = [label for label, passed in checks.items() if not passed]
        print("  Not ready for stereo triangulation from this pickle alone.")
        print(f"  Missing or undiscovered: {', '.join(missing)}.")
    return ready


def representative_frame(root: Any) -> tuple[str, dict] | None:
    """Return the first frame record from frames/action/subject/trial/frame."""
    frames = root.get("frames") if isinstance(root, dict) else None
    if not isinstance(frames, dict):
        return None
    path = "frames"
    value: Any = frames
    for _ in range(4):
        if not isinstance(value, dict) or not value:
            return None
        key = next(iter(value))
        path = f"{path}.{key}"
        value = value[key]
    return (path, value) if isinstance(value, dict) else None


def ec3d_frame_stats(root: Any) -> dict[str, Any]:
    """Count EC3D frame/view coverage without copying observation payloads."""
    frames = root.get("frames") if isinstance(root, dict) else None
    if not isinstance(frames, dict):
        return {}
    total = multi_view = four_view = with_3d = empty_2d = 0
    view_counts: Counter[str] = Counter()
    point_counts: Counter[str] = Counter()
    for subjects in frames.values():
        if not isinstance(subjects, dict):
            continue
        for trials in subjects.values():
            if not isinstance(trials, dict):
                continue
            for trial_frames in trials.values():
                if not isinstance(trial_frames, dict):
                    continue
                for record in trial_frames.values():
                    if not isinstance(record, dict):
                        continue
                    total += 1
                    observations = record.get("2D_op")
                    if not isinstance(observations, dict) or not observations:
                        empty_2d += 1
                        observations = {}
                    valid_views = 0
                    for view_id, points in observations.items():
                        point_count = len(points) if hasattr(points, "__len__") else 0
                        if point_count:
                            valid_views += 1
                            view_counts[str(view_id)] += 1
                            point_counts[f"{view_id}: {point_count} points"] += 1
                    multi_view += valid_views >= 2
                    four_view += valid_views >= 4
                    with_3d += bool(record.get("3D_gt"))
    return {
        "total": total,
        "multi_view": multi_view,
        "four_view": four_view,
        "with_3d": with_3d,
        "empty_2d": empty_2d,
        "view_counts": view_counts,
        "point_counts": point_counts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load and audit EC3D data.pickle for stereo prerequisites."
    )
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_PATH,
        help=f"trusted pickle to inspect (default: {DEFAULT_PATH})",
    )
    args = parser.parse_args()
    path = args.path.resolve()
    if not path.is_file():
        parser.error(f"pickle does not exist: {path}")

    print(f"Loading trusted pickle: {path}")
    print(f"File size: {path.stat().st_size / (1024 ** 2):.1f} MiB")
    with path.open("rb") as handle:
        data = pickle.load(handle)

    print(f"Root object: {describe(data)}")
    if isinstance(data, dict):
        print(f"Top-level keys: {list(data)[:20]}")
        params = data.get("params")
        if isinstance(params, dict):
            print(f"Calibrated camera IDs: {list(params)}")
            for camera_id, camera in list(params.items())[:4]:
                if isinstance(camera, dict):
                    print(f"  Camera {camera_id} fields: {list(camera)}")
                    for group, values in camera.items():
                        if isinstance(values, dict):
                            field_types = {
                                str(key): describe(value)
                                for key, value in values.items()
                            }
                            print(f"    {group}: {field_types}")
        sample = representative_frame(data)
        if sample:
            sample_path, sample_frame = sample
            print(f"Representative record: {sample_path}")
            print(f"Representative fields: {list(sample_frame)}")
            observations = sample_frame.get("2D_op")
            if isinstance(observations, dict):
                print(f"Representative 2D view IDs: {list(observations)}")
                print(
                    "Representative 2D observation sizes: "
                    f"{ {str(key): len(value) for key, value in observations.items()} }"
                )

    frame_stats = ec3d_frame_stats(data)
    if frame_stats:
        total = frame_stats["total"]
        print("\nEC3D frame coverage")
        print(f"  Total frame records: {total}")
        print(
            f"  At least two valid views: {frame_stats['multi_view']} "
            f"({frame_stats['multi_view'] / total * 100:.1f}%)"
        )
        print(
            f"  All four valid views: {frame_stats['four_view']} "
            f"({frame_stats['four_view'] / total * 100:.1f}%)"
        )
        print(f"  Empty 2D observations: {frame_stats['empty_2d']}")
        print(f"  Frames with 3D ground truth: {frame_stats['with_3d']}")
        print_counter("  Per-view valid-frame counts", frame_stats["view_counts"])
        print_counter("  Per-view observation sizes", frame_stats["point_counts"])

    audit = audit_structure(data)
    print(f"Objects inspected: {audit['visited_objects']}")
    print(f"Traversal truncated: {audit['truncated']}")
    print_counter("\nCommon object types", audit["type_counts"])
    print_counter("\nCommon ndarray shapes", audit["array_shapes"])
    for category, title in (
        ("view", "\nView/camera field examples"),
        ("time", "\nFrame/time field examples"),
        ("points_2d", "\n2D observation field examples"),
        ("points_3d", "\n3D observation field examples"),
        ("calibration", "\nCalibration field examples"),
    ):
        print_examples(title, audit["examples"][category])

    print(f"\nObserved 2D view IDs: {sorted(audit['observation_view_ids'])}")
    print(f"Calibrated camera IDs: {sorted(audit['calibrated_camera_ids'])}")
    triangulation_assessment(audit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

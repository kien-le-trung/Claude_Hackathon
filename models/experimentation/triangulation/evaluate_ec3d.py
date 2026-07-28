"""Evaluate calibrated DLT triangulation across one EC3D trial."""

from __future__ import annotations

import argparse
from collections import defaultdict
from typing import Any

import numpy as np

from data import common_joint_names, iter_frames, load_ec3d
from triangulate_ec3d import triangulate_joint


DEFAULT_TRIAL = "SQUAT/Hugues/1"
CAMERA_CONFIGURATIONS = {
    "6_1 + 6_2": ("6_1", "6_2"),
    "6_1 + 6_3": ("6_1", "6_3"),
    "6_1 + 6_4": ("6_1", "6_4"),
    "6_2 + 6_3": ("6_2", "6_3"),
    "6_2 + 6_4": ("6_2", "6_4"),
    "6_3 + 6_4": ("6_3", "6_4"),
    "all four": ("6_1", "6_2", "6_3", "6_4"),
}


def frame_number(path: str) -> int:
    """Extract the integer from a final path component such as frame_000095."""
    frame_name = path.rsplit("/", 1)[-1]
    if not frame_name.startswith("frame_"):
        raise ValueError(f"Unexpected EC3D frame name: {frame_name}")
    return int(frame_name.removeprefix("frame_"))


def select_trial_frames(
    data: dict[str, Any],
    trial_prefix: str,
    max_frames: int | None = None,
) -> list[tuple[str, dict]]:
    """Select and numerically order all frame records in one trial."""
    normalized = trial_prefix.strip("/")
    frames = [
        (path, frame)
        for path, frame in iter_frames(data)
        if path.startswith(f"{normalized}/")
    ]
    frames.sort(key=lambda item: frame_number(item[0]))
    if max_frames is not None:
        if max_frames <= 0:
            raise ValueError("max_frames must be positive")
        frames = frames[:max_frames]
    if not frames:
        raise ValueError(f"No EC3D frames found for trial {normalized!r}")
    return frames


def distribution(values: np.ndarray, scale: float = 1.0) -> dict[str, float]:
    """Summarize a non-empty one-dimensional measurement array."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("Expected a non-empty one-dimensional measurement array")
    return {
        "mean": float(values.mean() * scale),
        "median": float(np.median(values) * scale),
        "p95": float(np.percentile(values, 95) * scale),
    }


def summarize_per_joint(records: list[dict]) -> dict[Any, dict[str, float | int]]:
    errors_by_joint: dict[Any, list[float]] = defaultdict(list)
    for record in records:
        errors_by_joint[record["joint_name"]].append(record["error_3d"])

    summary = {}
    for joint_name, errors in sorted(
        errors_by_joint.items(),
        key=lambda item: str(item[0]),
    ):
        metrics = distribution(np.asarray(errors), scale=1000.0)
        summary[joint_name] = {
            "sample_count": len(errors),
            "mean_error_mm": metrics["mean"],
            "median_error_mm": metrics["median"],
            "p95_error_mm": metrics["p95"],
        }
    return summary


def evaluate_configuration(
    frames: list[tuple[str, dict]],
    camera_parameters: dict,
    camera_ids: tuple[str, ...],
) -> dict[str, Any]:
    """Evaluate one fixed set of cameras over all eligible joint-frame pairs."""
    records: list[dict] = []
    eligible_count = 0
    failure_count = 0
    frames_with_eligible_joints = 0
    frames_with_results: set[str] = set()

    for path, frame in frames:
        ground_truth = frame.get("3D_gt") or {}
        joint_names = [
            joint_name
            for joint_name in common_joint_names(frame, camera_ids)
            if joint_name in ground_truth
        ]
        if joint_names:
            frames_with_eligible_joints += 1
        eligible_count += len(joint_names)

        for joint_name in joint_names:
            try:
                result = triangulate_joint(
                    frame,
                    camera_parameters,
                    joint_name,
                    camera_ids,
                )
            except (ValueError, np.linalg.LinAlgError):
                failure_count += 1
                continue
            records.append({"frame_path": path, **result})
            frames_with_results.add(path)

    successful_count = len(records)
    if not successful_count:
        return {
            "summary": {
                "camera_ids": camera_ids,
                "trial_frame_count": len(frames),
                "frames_with_eligible_joints": frames_with_eligible_joints,
                "frames_with_results": 0,
                "eligible_joint_frames": eligible_count,
                "successful_joint_frames": 0,
                "failure_count": failure_count,
                "coverage_percent": 0.0,
                "average_joints_per_result_frame": 0.0,
            },
            "per_joint": {},
            "records": [],
        }

    errors_3d = np.asarray([record["error_3d"] for record in records])
    pixel_errors = np.concatenate([
        np.asarray(record["reprojection_errors"], dtype=np.float64)
        for record in records
    ])
    all_depths_positive = np.asarray([
        np.all(np.asarray(record["camera_depths"]) > 0)
        for record in records
    ])
    errors_3d_mm = distribution(errors_3d, scale=1000.0)
    reprojection_px = distribution(pixel_errors)

    summary = {
        "camera_ids": camera_ids,
        "trial_frame_count": len(frames),
        "frames_with_eligible_joints": frames_with_eligible_joints,
        "frames_with_results": len(frames_with_results),
        "eligible_joint_frames": eligible_count,
        "successful_joint_frames": successful_count,
        "failure_count": failure_count,
        "coverage_percent": (
            float(successful_count / eligible_count * 100.0)
            if eligible_count else 0.0
        ),
        "frame_coverage_percent": float(
            len(frames_with_results) / len(frames) * 100.0
        ),
        "average_joints_per_result_frame": float(
            successful_count / len(frames_with_results)
        ),
        "mpjpe_mm": errors_3d_mm["mean"],
        "median_3d_error_mm": errors_3d_mm["median"],
        "p95_3d_error_mm": errors_3d_mm["p95"],
        "mean_reprojection_error_px": reprojection_px["mean"],
        "median_reprojection_error_px": reprojection_px["median"],
        "p95_reprojection_error_px": reprojection_px["p95"],
        "positive_depth_percent": float(all_depths_positive.mean() * 100.0),
    }
    return {
        "summary": summary,
        "per_joint": summarize_per_joint(records),
        "records": records,
    }


def print_comparison(results: dict[str, dict[str, Any]]) -> None:
    print("\nCamera configuration comparison")
    print(
        f"{'Cameras':<16} {'Joint cov.':>10} {'Frame cov.':>10} "
        f"{'MPJPE':>10} {'P95 3D':>10} {'Median repr.':>13} {'Depth +':>9}"
    )
    print("-" * 84)
    for label, result in results.items():
        summary = result["summary"]
        if "mpjpe_mm" not in summary:
            print(
                f"{label:<16} {summary['coverage_percent']:9.1f}% "
                f"{0.0:9.1f}% {'n/a':>10} {'n/a':>10} {'n/a':>13} {'n/a':>9}"
            )
            continue
        print(
            f"{label:<16} "
            f"{summary['coverage_percent']:9.1f}% "
            f"{summary['frame_coverage_percent']:9.1f}% "
            f"{summary['mpjpe_mm']:9.1f} "
            f"{summary['p95_3d_error_mm']:9.1f} "
            f"{summary['median_reprojection_error_px']:12.2f} "
            f"{summary['positive_depth_percent']:8.1f}%"
        )


def print_per_joint(result: dict[str, Any], label: str) -> None:
    print(f"\nPer-joint error for {label}")
    print(f"{'Joint':>8} {'Samples':>9} {'Mean mm':>10} {'Median mm':>11} {'P95 mm':>10}")
    print("-" * 53)
    for joint_name, metrics in result["per_joint"].items():
        print(
            f"{str(joint_name):>8} "
            f"{metrics['sample_count']:9d} "
            f"{metrics['mean_error_mm']:10.1f} "
            f"{metrics['median_error_mm']:11.1f} "
            f"{metrics['p95_error_mm']:10.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DLT triangulation over one EC3D trial."
    )
    parser.add_argument("--trial", default=DEFAULT_TRIAL)
    parser.add_argument(
        "--max-frames",
        type=int,
        help="evaluate only the first N frames for a quick check",
    )
    args = parser.parse_args()

    print("Loading trusted EC3D pickle...")
    data = load_ec3d()
    frames = select_trial_frames(data, args.trial, args.max_frames)
    print(f"Trial: {args.trial}")
    print(
        f"Frames: {len(frames)} "
        f"({frame_number(frames[0][0])}-{frame_number(frames[-1][0])})"
    )

    results = {}
    for label, camera_ids in CAMERA_CONFIGURATIONS.items():
        results[label] = evaluate_configuration(
            frames,
            data["params"],
            camera_ids,
        )
    print_comparison(results)
    print_per_joint(results["all four"], "all four cameras")


if __name__ == "__main__":
    main()

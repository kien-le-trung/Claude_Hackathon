from __future__ import annotations

import csv
import json
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

JOINTS = 25


@dataclass
class Settings:
    root: Path
    values: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "Settings":
        path = Path(path).resolve()
        with path.open(encoding="utf-8") as handle:
            return cls(path.parent, yaml.safe_load(handle))

    def path(self, value: str) -> Path:
        return (self.root / value).resolve()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_pickle(path: Path) -> dict[str, Any]:
    """Load only the trusted local EC3D pickle (pickle is executable data)."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as handle:
        data = pickle.load(handle)
    if not isinstance(data, dict) or not {"frames", "params"} <= data.keys():
        raise ValueError("Expected EC3D root keys: frames and params")
    return data


def _points(mapping: dict[Any, Any] | None, dims: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.zeros((JOINTS, dims), np.float32)
    valid = np.zeros(JOINTS, np.float32)
    for key, point in (mapping or {}).items():
        index = int(key)
        array = np.asarray(point, dtype=np.float32).reshape(-1)
        if 0 <= index < JOINTS and len(array) >= dims and np.isfinite(array[:dims]).all():
            values[index] = array[:dims]
            valid[index] = 1
    return values, valid


def _normalize(sequence: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Pelvis-center and scale each sequence without converting missing zeros to poses."""
    out = sequence.copy()
    valid_points = out[mask.astype(bool)]
    if not len(valid_points):
        return out
    center = np.median(valid_points, axis=0)
    out -= center
    scale = np.median(np.linalg.norm(valid_points - center, axis=-1))
    out /= max(float(scale), 1e-6)
    out *= mask[..., None]
    return out


def _resample(array: np.ndarray, length: int) -> np.ndarray:
    if len(array) == length:
        return array
    indices = np.linspace(0, max(len(array) - 1, 0), length)
    nearest = np.rint(indices).astype(int)
    return array[nearest]


def inspect(config: Settings) -> dict[str, Any]:
    data = load_pickle(config.path(config.values["data"]["raw_pickle"]))
    summary = {
        "cameras": sorted(data["params"]),
        "actions": {
            action: {
                "subjects": len(subjects),
                "repetitions": sum(len(trials) for trials in subjects.values()),
                "frames": sum(len(frames) for trials in subjects.values() for frames in trials.values()),
            }
            for action, subjects in data["frames"].items()
        },
    }
    print(json.dumps(summary, indent=2))
    return summary


def preprocess(config: Settings) -> Path:
    cfg = config.values["data"]
    data = load_pickle(config.path(cfg["raw_pickle"]))
    unknown_cameras = set(cfg["cameras"]) - set(data["params"])
    if unknown_cameras:
        raise ValueError(
            f"Unknown camera IDs {sorted(unknown_cameras, key=str)}; "
            "quote IDs such as \"6_1\" in YAML"
        )
    output = config.path(cfg["processed_dir"])
    samples_dir = output / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    classes = sorted(data["frames"])
    split_by_subject = {
        subject: split
        for split in ("train", "val", "test")
        for subject in cfg[f"{split}_subjects"]
    }
    rows: list[dict[str, Any]] = []
    for action, subjects in data["frames"].items():
        for subject, trials in subjects.items():
            if subject not in split_by_subject:
                continue
            for trial, frames in trials.items():
                ordered = [frames[key] for key in sorted(frames)]
                three_d, three_mask = [], []
                views = {camera: [] for camera in cfg["cameras"]}
                view_masks = {camera: [] for camera in cfg["cameras"]}
                for frame in ordered:
                    xyz, xyz_mask = _points(frame.get("3D_gt"), 3)
                    three_d.append(xyz)
                    three_mask.append(xyz_mask)
                    for camera in cfg["cameras"]:
                        xy, xy_mask = _points((frame.get("2D_op") or {}).get(camera), 2)
                        views[camera].append(xy)
                        view_masks[camera].append(xy_mask)
                length = int(cfg["sequence_length"])
                xyz_mask = _resample(np.asarray(three_mask), length)
                xyz = _normalize(_resample(np.asarray(three_d), length), xyz_mask)
                if xyz_mask.mean() < float(cfg["min_valid_fraction"]):
                    continue
                for camera in cfg["cameras"]:
                    xy_mask = _resample(np.asarray(view_masks[camera]), length)
                    if xy_mask.mean() < float(cfg["min_valid_fraction"]):
                        continue
                    xy = _normalize(_resample(np.asarray(views[camera]), length), xy_mask)
                    sample_id = f"{action}_{subject}_{trial}_{camera}".replace(" ", "_")
                    relative = Path("samples") / f"{sample_id}.npz"
                    np.savez_compressed(
                        output / relative,
                        teacher=np.concatenate([xyz, xyz_mask[..., None]], axis=-1),
                        student=np.concatenate([xy, xy_mask[..., None]], axis=-1),
                        label=np.int64(classes.index(action)),
                    )
                    rows.append({
                        "sample_id": sample_id, "path": relative.as_posix(), "action": action,
                        "label": classes.index(action), "subject": subject, "trial": trial,
                        "camera": camera, "split": split_by_subject[subject],
                        "source_frames": len(ordered), "valid_3d": float(xyz_mask.mean()),
                        "valid_2d": float(xy_mask.mean()),
                    })
    manifest = output / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    (output / "classes.json").write_text(json.dumps(classes, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} view-specific samples to {output}")
    return manifest


def read_manifest(config: Settings, split: str) -> list[dict[str, str]]:
    path = config.path(config.values["data"]["processed_dir"]) / "manifest.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        return [row for row in csv.DictReader(handle) if row["split"] == split]

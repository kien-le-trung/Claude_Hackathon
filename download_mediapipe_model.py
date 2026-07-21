#!/usr/bin/env python3
"""Download the MediaPipe Pose Landmarker full model used by SquatSpot."""

from pathlib import Path
import urllib.request


MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/1/pose_landmarker_full.task"
)
MODEL_PATH = Path(__file__).resolve().parent / "models" / "pose_landmarker_full.task"


def main() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = MODEL_PATH.with_suffix(".task.part")
    print(f"Downloading MediaPipe Pose Landmarker full model to {MODEL_PATH}")
    try:
        urllib.request.urlretrieve(MODEL_URL, temporary_path)
        temporary_path.replace(MODEL_PATH)
    finally:
        temporary_path.unlink(missing_ok=True)
    print("Download complete.")


if __name__ == "__main__":
    main()

"""Manual implementation target: extract MediaPipe poses for aligned w00 frames."""

from __future__ import annotations

import numpy as np

import json
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from alignment import squat_frame_ids
from config import MEDIAPIPE_OUTPUT_PATH, MODEL_PATH, RGB_PATH, OUTPUT_DIR

OUTPUT_PATH = OUTPUT_DIR / "w00_mediapipe.jsonl"
LANDMARK_FIELD_ORDER = ("x", "y", "z", "visibility", "presence")


def extract() -> None:
    """Extract aligned MediaPipe normalized and world landmarks.

    Implementation checklist:
    1. Validate RGB_PATH and MODEL_PATH.
    2. Obtain the target IDs from squat_frame_ids().
    3. Decode w00 sequentially; do not seek once per frame.
    4. Call PoseLandmarker in VIDEO mode with frame_id / 30 converted to ms.
    5. Preallocate NaN arrays shaped (N, 33, 5) for normalized and world poses.
    6. Preserve one row per target frame, including missing detections.
    7. Save the proposed NPZ schema documented in README.md.

    Reuse the options and serialization semantics in apps/api/app/analysis.py,
    but keep this experiment independent from API persistence.
    """
    target_frame_ids = squat_frame_ids()
    target_set = set(int(frame_id) for frame_id in target_frame_ids)

    capture = cv2.VideoCapture(RGB_PATH)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    first_target_frame = target_frame_ids[0]
    last_target_frame = target_frame_ids[-1]

    # Count for book-keeping
    detected = 0
    written = 0
    frame_id = 0

    try:
        with create_landmarker() as landmarker:
            with OUTPUT_PATH.open("w", encoding="utf-8") as output_file:
                while capture.isOpened():
                    decoded, bgr_frame = capture.read()
                    if frame_id in target_set:
                        rgb_frame = cv2.cvtColor(
                            bgr_frame,
                            cv2.COLOR_BGR2RGB,
                        )
                        # convert to media pipe expected image
                        image = mp.Image(
                            image_format=mp.ImageFormat.SRGB,
                            data=rgb_frame,
                        )
                        timestamp_ms = int(
                            round(frame_id * 1000.0 / fps)
                        )
                        result = landmarker.detect_for_video(
                            image,
                            timestamp_ms
                        )
                        pose_detected = bool(
                            result.pose_world_landmarks
                        )
                        if pose_detected:
                            detected += 1
                            world_landmarks = serialize_landmarks(
                                result.pose_world_landmarks[0]
                            )

                            normalized_landmarks = serialize_landmarks(
                                result.pose_landmarks[0]
                            )
                        else:
                            world_landmarks = None
                            normalized_landmarks = None
                        record = {
                            "frame_id": frame_id,
                            "timestamp_ms": timestamp_ms,
                            "detected": pose_detected,
                            "world_landmarks": world_landmarks,
                            "normalized_landmarks": normalized_landmarks,
                        }
                        output_file.write(
                            json.dumps(record) + "\n"
                        )
                        written += 1
                    frame_id += 1
    finally:
        capture.release()

    print(f"Video frames: {frame_count}")
    print(f"Target frames: {len(target_frame_ids)}")
    print(f"Detected poses: {detected}")
    print(
        f"Detection rate: "
        f"{detected / len(target_frame_ids) * 100:.1f}%"
    )
    print(f"Output: {OUTPUT_PATH}")

    raise NotImplementedError(
        f"Implement MediaPipe extraction for {len(target_frame_ids)} aligned frames "
        f"from {RGB_PATH}; write {MEDIAPIPE_OUTPUT_PATH}"
    )


def create_landmarker():
    # pose landmarker config
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_PATH.resolve())
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_segmentation_masks=False,
    )
    return vision.PoseLandmarker.create_from_options(options)


def serialize_landmarks(landmarks):
    return [
        {
            "index" : index,
            "x" : float(landmark.x),
            "y" : float(landmark.y),
            "z" : float(landmark.z),
            "visibility" : (
                float(landmark.visibility)
                if landmark.visibility is not None
                else None
            ),
            "presence" : (
                float(landmark.presence)
                if landmark.presence is not None
                else None
            ),
        }
        for index, landmark in enumerate(landmarks)
    ]

def load_extracted_landmarks():
    records = []
    with OUTPUT_PATH.open(encoding="utf-8") as input_file:
        for line in input_file:
            records.append(json.loads(line))
    return records


def validate_output() -> None:
    """Validate the output contract after `extract` is implemented."""
    data = np.load(MEDIAPIPE_OUTPUT_PATH)
    required = {"frame_ids", "normalized_landmarks", "world_landmarks", "detected"}
    missing = required.difference(data.files)
    if missing:
        raise ValueError(f"Extraction artifact is missing: {sorted(missing)}")
    count = len(data["frame_ids"])
    if data["normalized_landmarks"].shape != (count, 33, 5):
        raise ValueError("Unexpected normalized landmark shape")
    if data["world_landmarks"].shape != (count, 33, 5):
        raise ValueError("Unexpected world landmark shape")
    if data["detected"].shape != (count,):
        raise ValueError("Unexpected detection mask shape")
    if not np.array_equal(data["frame_ids"], squat_frame_ids()):
        raise ValueError("Output frame IDs do not match the aligned squat frames")


if __name__ == "__main__":
    extract()


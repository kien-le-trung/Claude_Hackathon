import os
from pathlib import Path
from typing import Callable

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .config import Settings
from .validation import VideoMetadata


SCHEMA_VERSION = 1
LANDMARK_NAMES = (
    "NOSE", "LEFT_EYE_INNER", "LEFT_EYE", "LEFT_EYE_OUTER",
    "RIGHT_EYE_INNER", "RIGHT_EYE", "RIGHT_EYE_OUTER", "LEFT_EAR",
    "RIGHT_EAR", "MOUTH_LEFT", "MOUTH_RIGHT", "LEFT_SHOULDER",
    "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST",
    "RIGHT_WRIST", "LEFT_PINKY", "RIGHT_PINKY", "LEFT_INDEX",
    "RIGHT_INDEX", "LEFT_THUMB", "RIGHT_THUMB", "LEFT_HIP", "RIGHT_HIP",
    "LEFT_KNEE", "RIGHT_KNEE", "LEFT_ANKLE", "RIGHT_ANKLE", "LEFT_HEEL",
    "RIGHT_HEEL", "LEFT_FOOT_INDEX", "RIGHT_FOOT_INDEX",
)


def _optional_float(value) -> float | None:
    return None if value is None else float(value)


def _serialize_landmarks(landmarks: list) -> list[dict]:
    return [
        {
            "index": index,
            "name": LANDMARK_NAMES[index],
            "x": float(landmark.x),
            "y": float(landmark.y),
            "z": float(landmark.z),
            "visibility": _optional_float(getattr(landmark, "visibility", None)),
            "presence": _optional_float(getattr(landmark, "presence", None)),
        }
        for index, landmark in enumerate(landmarks)
    ]


class LandmarkExtractionService:
    def __init__(
        self,
        settings: Settings,
        landmarker_factory: Callable[[], object] | None = None,
    ):
        self.settings = settings
        self._landmarker_factory = landmarker_factory or self._create_landmarker

    def _create_landmarker(self):
        model_path = self.settings.mediapipe_model_path.resolve()
        if not model_path.is_file():
            raise FileNotFoundError(
                f"MediaPipe model not found at {model_path}. Run download_mediapipe_model.py."
            )
        options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self.settings.pose_detection_confidence,
            min_pose_presence_confidence=self.settings.pose_presence_confidence,
            min_tracking_confidence=self.settings.tracking_confidence,
            output_segmentation_masks=False,
        )
        return vision.PoseLandmarker.create_from_options(options)

    def extract(
        self,
        video_path: Path,
        metadata: VideoMetadata,
        progress_callback: Callable[[dict], None] | None = None,
    ) -> dict:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise ValueError("Validated video could not be reopened")

        frames: list[dict] = []
        detected_frame_count = 0
        frame_number = 0
        sample_interval_ms = 1000.0 / self.settings.target_sampling_fps
        next_sample_ms = 0.0
        effective_fps = min(metadata.fps, self.settings.target_sampling_fps)
        previous_timestamp_ms = -1.0

        try:
            with self._landmarker_factory() as landmarker:
                while capture.isOpened():
                    decoded, frame = capture.read()
                    if not decoded:
                        break
                    timestamp_ms = frame_number * 1000.0 / metadata.fps
                    if hasattr(capture, "get"):
                        reported_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                        if reported_ms >= 0 and (frame_number == 0 or reported_ms > previous_timestamp_ms):
                            timestamp_ms = reported_ms
                    previous_timestamp_ms = timestamp_ms
                    if timestamp_ms + 1e-6 >= next_sample_ms:
                        integer_timestamp_ms = int(round(timestamp_ms))
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                        result = landmarker.detect_for_video(image, integer_timestamp_ms)
                        poses = []
                        if result.pose_landmarks:
                            detected_frame_count += 1
                            poses.append({
                                "landmarks": _serialize_landmarks(result.pose_landmarks[0]),
                                "world_landmarks": _serialize_landmarks(
                                    result.pose_world_landmarks[0]
                                    if result.pose_world_landmarks else []
                                ),
                            })
                        frames.append({
                            "frame_number": frame_number,
                            "timestamp_ms": integer_timestamp_ms,
                            "poses": poses,
                        })
                        next_sample_ms += sample_interval_ms
                    if progress_callback is not None:
                        progress_callback({
                            "decoded_frame_count": frame_number + 1,
                            "sampled_frame_count": len(frames),
                            "detected_frame_count": detected_frame_count,
                            "total_frame_count": metadata.total_frames,
                            "percent": round(
                                min((frame_number + 1) / metadata.total_frames * 100, 100.0),
                                1,
                            ),
                        })
                    frame_number += 1
        finally:
            capture.release()

        sampled_frame_count = len(frames)
        summary = {
            "schema_version": SCHEMA_VERSION,
            "model": "pose_landmarker_full",
            "video": metadata.to_dict(),
            "sampling_fps": self.settings.target_sampling_fps,
            "effective_sampling_fps": effective_fps,
            "decoded_frame_count": frame_number,
            "sampled_frame_count": sampled_frame_count,
            "detected_frame_count": detected_frame_count,
            "detection_rate": round(
                detected_frame_count / sampled_frame_count * 100, 1
            ) if sampled_frame_count else 0.0,
        }
        return {
            "schema_version": SCHEMA_VERSION,
            "extractor": {
                "name": "mediapipe_pose_landmarker",
                "model": "pose_landmarker_full",
                "num_poses": 1,
                "min_pose_detection_confidence": self.settings.pose_detection_confidence,
                "min_pose_presence_confidence": self.settings.pose_presence_confidence,
                "min_tracking_confidence": self.settings.tracking_confidence,
            },
            "video": metadata.to_dict(),
            "sampling": {
                "target_fps": self.settings.target_sampling_fps,
                "effective_fps": effective_fps,
            },
            "summary": summary,
            "frames": frames,
        }

    @staticmethod
    def delete_temporary_file(path: Path) -> None:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

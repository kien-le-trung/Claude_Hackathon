import json
import os
from pathlib import Path

import cv2

from openpose_model import OpenPoseModel

from .comparator import compare_squats
from .config import Settings


class AnalysisService:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: OpenPoseModel | None = None
        self._reference: dict | None = None

    def _load_model(self) -> OpenPoseModel:
        if self._model is None:
            model = OpenPoseModel()
            model.load_model(
                str(self.settings.model_weights_path.resolve()),
                str(self.settings.model_config_path.resolve()),
            )
            self._model = model
        return self._model

    def _load_reference(self) -> dict:
        if self._reference is None:
            with self.settings.reference_json_path.resolve().open("r", encoding="utf-8") as handle:
                self._reference = json.load(handle)
        return self._reference

    def extract_keypoints(self, video_path: Path, frame_skip: int = 5) -> dict:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError("Could not open uploaded video")

        fps = capture.get(cv2.CAP_PROP_FPS)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = []
        frame_number = 0
        model = self._load_model()

        try:
            while capture.isOpened():
                success, frame = capture.read()
                if not success:
                    break
                frame_number += 1
                if frame_number % frame_skip:
                    continue
                keypoints, _ = model.detect(frame)
                detected = []
                for index, keypoint in enumerate(keypoints):
                    if keypoint is not None:
                        detected.append({
                            "body_part": model.get_body_part_name(index),
                            "index": index,
                            "x": float(keypoint[0]),
                            "y": float(keypoint[1]),
                        })
                frames.append({
                    "frame_number": frame_number,
                    "timestamp": frame_number / fps if fps else 0,
                    "keypoints": detected,
                })
        finally:
            capture.release()

        return {
            "fps": fps,
            "total_frames": total_frames,
            "duration": total_frames / fps if fps else 0,
            "frame_skip": frame_skip,
            "frames": frames,
        }

    def analyze(self, video_path: Path) -> dict:
        extracted = self.extract_keypoints(video_path)
        comparison = compare_squats(extracted, self._load_reference())
        return {"user_data": extracted, "results": comparison}

    @staticmethod
    def delete_temporary_file(path: Path) -> None:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

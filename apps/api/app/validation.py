from dataclasses import asdict, dataclass
import math
from pathlib import Path

import cv2

from .config import Settings


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    total_frames: int
    duration_seconds: float
    width: int
    height: int
    content_type: str

    def to_dict(self) -> dict:
        return asdict(self)


class VideoValidationError(ValueError):
    def __init__(self, detail: str, status_code: int = 400):
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


class VideoValidator:
    def __init__(self, settings: Settings):
        self.settings = settings

    def validate_upload(self, filename: str | None, content_type: str | None) -> str:
        suffix = Path(filename or "").suffix.lower().lstrip(".")
        if suffix not in self.settings.allowed_extensions:
            raise VideoValidationError("Unsupported video extension")

        normalized_content_type = (content_type or "").lower().split(";", 1)[0].strip()
        allowed = self.settings.allowed_mime_types.get(suffix, set())
        if normalized_content_type not in allowed:
            raise VideoValidationError("Video MIME type does not match its extension")
        return suffix

    def validate_file(self, video_path: Path, content_type: str) -> VideoMetadata:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise VideoValidationError("Could not open uploaded video")

        try:
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            total_frames_value = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            width_value = float(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height_value = float(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            decoded, _ = capture.read()
        finally:
            capture.release()

        numeric_values = (fps, total_frames_value, width_value, height_value)
        if not all(math.isfinite(value) and value > 0 for value in numeric_values):
            raise VideoValidationError("Video has invalid stream metadata")
        if not decoded:
            raise VideoValidationError("Could not decode the first video frame")

        total_frames = int(total_frames_value)
        width = int(width_value)
        height = int(height_value)
        duration = total_frames / fps
        if duration > self.settings.max_video_duration_seconds:
            raise VideoValidationError("Video exceeds the 120 second duration limit", 413)
        if width * height > self.settings.max_video_pixels:
            raise VideoValidationError("Video exceeds the 4K resolution limit", 413)

        return VideoMetadata(
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration,
            width=width,
            height=height,
            content_type=content_type.lower().split(";", 1)[0].strip(),
        )

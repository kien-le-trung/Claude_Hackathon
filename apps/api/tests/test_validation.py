from types import SimpleNamespace

import pytest

from app.validation import VideoValidationError, VideoValidator


class FakeCapture:
    def __init__(self, values: dict[int, float], opened: bool = True, decoded: bool = True):
        self.values = values
        self.opened = opened
        self.decoded = decoded
        self.released = False

    def isOpened(self) -> bool:
        return self.opened

    def get(self, key: int) -> float:
        return self.values.get(key, 0)

    def read(self):
        return self.decoded, object() if self.decoded else None

    def release(self) -> None:
        self.released = True


@pytest.fixture
def settings():
    return SimpleNamespace(
        allowed_extensions={"mp4", "avi", "mov", "mkv"},
        allowed_mime_types={
            "mp4": {"video/mp4"},
            "avi": {"video/x-msvideo"},
            "mov": {"video/quicktime"},
            "mkv": {"video/x-matroska"},
        },
        max_video_duration_seconds=120,
        max_video_pixels=3840 * 2160,
    )


def valid_values(cv2_module) -> dict[int, float]:
    return {
        cv2_module.CAP_PROP_FPS: 30,
        cv2_module.CAP_PROP_FRAME_COUNT: 3600,
        cv2_module.CAP_PROP_FRAME_WIDTH: 1080,
        cv2_module.CAP_PROP_FRAME_HEIGHT: 1920,
    }


def test_upload_requires_matching_extension_and_mime(settings):
    validator = VideoValidator(settings)
    assert validator.validate_upload("squat.mp4", "video/mp4; charset=binary") == "mp4"
    with pytest.raises(VideoValidationError, match="extension"):
        validator.validate_upload("squat.txt", "video/mp4")
    with pytest.raises(VideoValidationError, match="MIME"):
        validator.validate_upload("squat.mp4", "video/quicktime")


@pytest.mark.parametrize("width,height", [(3840, 2160), (2160, 3840)])
def test_valid_landscape_and_portrait_video_at_limits(
    monkeypatch, settings, width, height
):
    from app import validation

    values = valid_values(validation.cv2)
    values[validation.cv2.CAP_PROP_FRAME_WIDTH] = width
    values[validation.cv2.CAP_PROP_FRAME_HEIGHT] = height
    capture = FakeCapture(values)
    monkeypatch.setattr(validation.cv2, "VideoCapture", lambda _: capture)
    metadata = VideoValidator(settings).validate_file("video.mp4", "video/mp4")
    assert metadata.duration_seconds == 120
    assert (metadata.width, metadata.height) == (width, height)
    assert capture.released


@pytest.mark.parametrize(
    ("change", "message", "status_code"),
    [
        ({"opened": False}, "open", 400),
        ({"decoded": False}, "decode", 400),
        ({"fps": 0}, "metadata", 400),
        ({"frames": 3601}, "duration", 413),
        ({"width": 4096, "height": 2160}, "resolution", 413),
    ],
)
def test_invalid_video_is_rejected(monkeypatch, settings, change, message, status_code):
    from app import validation

    values = valid_values(validation.cv2)
    values[validation.cv2.CAP_PROP_FPS] = change.get("fps", values[validation.cv2.CAP_PROP_FPS])
    values[validation.cv2.CAP_PROP_FRAME_COUNT] = change.get(
        "frames", values[validation.cv2.CAP_PROP_FRAME_COUNT]
    )
    values[validation.cv2.CAP_PROP_FRAME_WIDTH] = change.get(
        "width", values[validation.cv2.CAP_PROP_FRAME_WIDTH]
    )
    values[validation.cv2.CAP_PROP_FRAME_HEIGHT] = change.get(
        "height", values[validation.cv2.CAP_PROP_FRAME_HEIGHT]
    )
    capture = FakeCapture(
        values,
        opened=change.get("opened", True),
        decoded=change.get("decoded", True),
    )
    monkeypatch.setattr(validation.cv2, "VideoCapture", lambda _: capture)
    with pytest.raises(VideoValidationError, match=message) as error:
        VideoValidator(settings).validate_file("video.mp4", "video/mp4")
    assert error.value.status_code == status_code

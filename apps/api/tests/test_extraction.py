from types import SimpleNamespace

from app import analysis
from app.analysis import LANDMARK_NAMES, LandmarkExtractionService
from app.validation import VideoMetadata


class FakeCapture:
    def __init__(self, frame_count: int):
        self.frames = list(range(frame_count))
        self.index = 0

    def isOpened(self):
        return True

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self):
        pass


class FakeLandmarker:
    def __init__(self):
        self.timestamps = []

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def detect_for_video(self, image, timestamp_ms):
        self.timestamps.append(timestamp_ms)
        landmarks = [
            SimpleNamespace(x=i / 100, y=i / 200, z=-i / 300, visibility=0.9, presence=0.8)
            for i in range(33)
        ]
        return SimpleNamespace(pose_landmarks=[landmarks], pose_world_landmarks=[landmarks])


def test_extraction_samples_by_timestamp_and_serializes_landmarks(monkeypatch):
    fake_landmarker = FakeLandmarker()
    converted_frames = []
    monkeypatch.setattr(analysis.cv2, "VideoCapture", lambda _: FakeCapture(31))
    monkeypatch.setattr(
        analysis.cv2,
        "cvtColor",
        lambda frame, conversion: converted_frames.append((frame, conversion)) or frame,
    )
    monkeypatch.setattr(analysis.mp, "Image", lambda **kwargs: kwargs)

    settings = SimpleNamespace(
        target_sampling_fps=10.0,
        pose_detection_confidence=0.5,
        pose_presence_confidence=0.5,
        tracking_confidence=0.5,
    )
    metadata = VideoMetadata(30.0, 31, 31 / 30, 640, 480, "video/mp4")
    payload = LandmarkExtractionService(
        settings, landmarker_factory=lambda: fake_landmarker
    ).extract("video.mp4", metadata)

    assert fake_landmarker.timestamps == list(range(0, 1001, 100))
    assert all(
        later > earlier
        for earlier, later in zip(fake_landmarker.timestamps, fake_landmarker.timestamps[1:])
    )
    assert len(converted_frames) == 11
    assert all(item[1] == analysis.cv2.COLOR_BGR2RGB for item in converted_frames)
    assert payload["schema_version"] == 1
    assert payload["summary"]["sampled_frame_count"] == 11
    assert payload["summary"]["detected_frame_count"] == 11
    pose = payload["frames"][0]["poses"][0]
    assert len(pose["landmarks"]) == len(LANDMARK_NAMES) == 33
    assert pose["landmarks"][23]["name"] == "LEFT_HIP"
    assert pose["world_landmarks"][0]["presence"] == 0.8


def test_extraction_preserves_frames_without_a_detected_pose(monkeypatch):
    landmarker = FakeLandmarker()
    landmarker.detect_for_video = lambda *_: SimpleNamespace(
        pose_landmarks=[], pose_world_landmarks=[]
    )
    monkeypatch.setattr(analysis.cv2, "VideoCapture", lambda _: FakeCapture(1))
    monkeypatch.setattr(analysis.cv2, "cvtColor", lambda frame, _: frame)
    monkeypatch.setattr(analysis.mp, "Image", lambda **kwargs: kwargs)
    settings = SimpleNamespace(
        target_sampling_fps=10.0,
        pose_detection_confidence=0.5,
        pose_presence_confidence=0.5,
        tracking_confidence=0.5,
    )
    metadata = VideoMetadata(30.0, 1, 1 / 30, 640, 480, "video/mp4")
    payload = LandmarkExtractionService(settings, lambda: landmarker).extract(
        "video.mp4", metadata
    )
    assert payload["frames"][0]["poses"] == []
    assert payload["summary"]["detection_rate"] == 0.0

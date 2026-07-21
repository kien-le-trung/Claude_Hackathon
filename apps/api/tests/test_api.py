from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient

from app import main
from app.database import get_db
from app.validation import VideoMetadata


class FakeDatabase:
    def __init__(self):
        self.records = []

    def add(self, record):
        record.id = uuid4()
        record.date_created = datetime.now(timezone.utc)
        self.records.append(record)

    def commit(self):
        pass

    def refresh(self, _record):
        pass

    def get(self, _model, record_id):
        return next((record for record in self.records if record.id == record_id), None)


def test_valid_upload_returns_queued_summary_and_schedules_extraction(monkeypatch, tmp_path):
    database = FakeDatabase()
    metadata = VideoMetadata(30.0, 300, 10.0, 1920, 1080, "video/mp4")
    scheduled = []
    monkeypatch.setattr(main.settings, "temp_dir", tmp_path)
    monkeypatch.setattr(main.video_validator, "validate_file", lambda *_: metadata)

    def fake_process(video_id, temporary_path, received_metadata):
        scheduled.append((video_id, temporary_path, received_metadata))
        temporary_path.unlink(missing_ok=True)

    monkeypatch.setattr(main, "process_video", fake_process)
    main.app.dependency_overrides[get_db] = lambda: database
    try:
        response = TestClient(main.app).post(
            "/api/videos",
            files={"video": ("squat.mp4", b"video bytes", "video/mp4")},
        )
    finally:
        main.app.dependency_overrides.clear()

    assert response.status_code == 202
    assert response.json()["status"] == "queued"
    assert response.json()["extraction"] is None
    assert len(database.records) == 1
    assert scheduled[0][2] == metadata


def test_oversized_upload_is_rejected_before_persistence(monkeypatch, tmp_path):
    database = FakeDatabase()
    monkeypatch.setattr(main.settings, "temp_dir", tmp_path)
    monkeypatch.setattr(main.settings, "max_upload_bytes", 3)
    main.app.dependency_overrides[get_db] = lambda: database
    try:
        response = TestClient(main.app).post(
            "/api/videos",
            files={"video": ("squat.mp4", b"four", "video/mp4")},
        )
    finally:
        main.app.dependency_overrides.clear()

    assert response.status_code == 413
    assert database.records == []
    assert list(tmp_path.iterdir()) == []


def test_status_response_exposes_summary_but_not_raw_landmarks():
    summary = {"schema_version": 1, "sampled_frame_count": 10}
    video = SimpleNamespace(
        id=uuid4(),
        date_created=datetime.now(timezone.utc),
        status="completed",
        original_filename="squat.mp4",
        content_type="video/mp4",
        extracted_json={"summary": summary, "frames": [{"poses": ["raw"]}]},
        error_message=None,
    )
    payload = main.serialize_video(video)
    assert payload["extraction"] == summary
    assert "frames" not in payload
    assert "result" not in payload


def test_background_processing_persists_payload_and_deletes_file(monkeypatch, tmp_path):
    record = SimpleNamespace(status="queued", extracted_json=None, error_message=None)
    commits = []

    class ProcessingDatabase:
        def get(self, *_):
            return record

        def commit(self):
            commits.append(record.status)

        def rollback(self):
            pass

        def close(self):
            pass

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    metadata = VideoMetadata(30.0, 30, 1.0, 640, 480, "video/mp4")
    payload = {"schema_version": 1, "summary": {"sampled_frame_count": 10}}
    monkeypatch.setattr(main, "SessionLocal", lambda: ProcessingDatabase())
    monkeypatch.setattr(main.extraction_service, "extract", lambda *_: payload)

    main.process_video(uuid4(), video_path, metadata)

    assert commits == ["processing", "completed"]
    assert record.extracted_json == payload
    assert record.error_message is None
    assert not video_path.exists()


def test_background_processing_records_failure_and_deletes_file(monkeypatch, tmp_path):
    record = SimpleNamespace(status="queued", extracted_json=None, error_message=None)

    class ProcessingDatabase:
        def get(self, *_):
            return record

        def commit(self):
            pass

        def rollback(self):
            pass

        def close(self):
            pass

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    metadata = VideoMetadata(30.0, 30, 1.0, 640, 480, "video/mp4")
    monkeypatch.setattr(main, "SessionLocal", lambda: ProcessingDatabase())
    monkeypatch.setattr(
        main.extraction_service,
        "extract",
        lambda *_: (_ for _ in ()).throw(RuntimeError("model failed")),
    )

    main.process_video(uuid4(), video_path, metadata)

    assert record.status == "failed"
    assert record.error_message == "model failed"
    assert not video_path.exists()

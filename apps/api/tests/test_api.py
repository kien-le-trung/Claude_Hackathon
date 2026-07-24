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

    def delete(self, record):
        self.records.remove(record)


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
    assert payload["classification"] is None
    assert "frames" not in payload
    assert "result" not in payload


def test_status_response_exposes_classification_summary_only():
    classification = {"model_version": "v1", "frames": [{"probabilities": {}}],
                      "summary": {"result": "good", "events": []}}
    video = SimpleNamespace(
        id=uuid4(), date_created=datetime.now(timezone.utc), status="completed",
        original_filename="squat.mp4", content_type="video/mp4",
        extracted_json={"summary": {}, "classification": classification}, error_message=None,
    )
    payload = main.serialize_video(video)
    assert payload["classification"] == classification["summary"]
    assert "frames" not in payload["classification"]


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
    payload = {
        "schema_version": 1,
        "summary": {
            "decoded_frame_count": 30,
            "sampled_frame_count": 1,
            "detected_frame_count": 1,
        },
        "sampling": {"effective_fps": 10.0},
        "frames": [{"frame_number": 0, "timestamp_ms": 0, "poses": [{}]}],
    }

    class Classifier:
        def classify(self, _payload):
            return {
                "frames": [{
                    "frame_number": 0,
                    "timestamp_ms": 0,
                    "predicted_class": "good",
                    "probabilities": {"good": 0.9},
                    "measurements": {},
                }],
                "events": [],
                "summary": {
                    "result": "good",
                    "events": [],
                    "classified_frame_count": 1,
                },
            }

    monkeypatch.setattr(main, "SessionLocal", lambda: ProcessingDatabase())
    monkeypatch.setattr(main.extraction_service, "extract", lambda *_, **__: payload)
    monkeypatch.setattr(main, "get_classifier", lambda: Classifier())
    monkeypatch.setattr(main, "create_event_artifacts", lambda *_: None)

    main.process_video(uuid4(), video_path, metadata)

    assert commits == ["processing", "completed"]
    assert record.extracted_json["classification"]["summary"]["result"] == "good"
    assert record.extracted_json["evidence"]["frames"][0]["predicted_class"] == "good"
    assert "frames" not in record.extracted_json
    assert record.error_message is None
    assert not video_path.exists()


def test_background_processing_persists_frame_classification(monkeypatch, tmp_path):
    record = SimpleNamespace(status="queued", extracted_json=None, error_message=None)

    class ProcessingDatabase:
        def get(self, *_): return record
        def commit(self): pass
        def rollback(self): pass
        def close(self): pass

    class Classifier:
        def classify(self, _payload):
            return {
                "frames": [],
                "events": [],
                "summary": {
                    "result": "good",
                    "events": [],
                    "classified_frame_count": 0,
                },
            }

    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    metadata = VideoMetadata(30.0, 30, 1.0, 640, 480, "video/mp4")
    monkeypatch.setattr(main, "SessionLocal", lambda: ProcessingDatabase())
    monkeypatch.setattr(
        main.extraction_service,
        "extract",
        lambda *_, **__: {
            "summary": {
                "decoded_frame_count": 30,
                "sampled_frame_count": 0,
                "detected_frame_count": 0,
            },
            "sampling": {"effective_fps": 10.0},
            "frames": [],
        },
    )
    monkeypatch.setattr(main, "get_classifier", lambda: Classifier())
    monkeypatch.setattr(main, "create_event_artifacts", lambda *_: None)
    main.process_video(uuid4(), video_path, metadata)
    assert record.extracted_json["classification"]["summary"]["result"] == "good"
    assert record.extracted_json["evidence"]["frames"] == []
    assert record.status == "completed"


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
        lambda *_, **__: (_ for _ in ()).throw(RuntimeError("model failed")),
    )

    main.process_video(uuid4(), video_path, metadata)

    assert record.status == "failed"
    assert record.error_message == "model failed"
    assert not video_path.exists()


def test_event_frame_is_served_and_delete_removes_record_and_artifacts(
    monkeypatch, tmp_path
):
    database = FakeDatabase()
    video_id = uuid4()
    event_id = uuid4()
    relative = f"{video_id}/{event_id}.webp"
    stored = tmp_path / relative
    stored.parent.mkdir(parents=True)
    stored.write_bytes(b"webp")
    record = SimpleNamespace(
        id=video_id,
        date_created=datetime.now(timezone.utc),
        status="completed",
        original_filename="squat.mp4",
        content_type="video/mp4",
        extracted_json={
            "summary": {},
            "classification": {
                "summary": {
                    "result": "errors_detected",
                    "events": [{
                        "id": str(event_id),
                        "frame_image_path": relative,
                    }],
                }
            },
        },
        error_message=None,
    )
    database.records.append(record)
    monkeypatch.setattr(main.settings, "artifact_root", tmp_path)
    main.app.dependency_overrides[get_db] = lambda: database
    try:
        client = TestClient(main.app)
        frame_response = client.get(
            f"/api/videos/{video_id}/events/{event_id}/frame"
        )
        delete_response = client.delete(f"/api/videos/{video_id}")
    finally:
        main.app.dependency_overrides.clear()

    assert frame_response.status_code == 200
    assert frame_response.content == b"webp"
    assert delete_response.status_code == 204
    assert database.records == []
    assert not stored.exists()

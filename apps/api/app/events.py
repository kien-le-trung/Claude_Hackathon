from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4


ERROR_TYPES = ("bad_back", "bad_heel")


@dataclass
class _Run:
    error_type: str
    first: dict
    last: dict
    representative: dict
    qualifying: list[dict] = field(default_factory=list)


def _qualifying_error(frame: dict, threshold: float) -> str | None:
    predicted = frame.get("predicted_class")
    probability = frame.get("probabilities", {}).get(predicted, 0.0)
    return predicted if predicted in ERROR_TYPES and probability >= threshold else None


def consolidate_error_events(
    frames: list[dict],
    *,
    threshold: float,
    sampling_fps: float,
    maximum_gap_ms: int = 200,
    minimum_duration_ms: int = 300,
    minimum_evidence_frames: int = 3,
) -> list[dict]:
    """Consolidate stable, same-type frame predictions into chronological events."""
    if sampling_fps <= 0:
        raise ValueError("sampling_fps must be positive")
    sample_interval_ms = 1000.0 / sampling_fps
    events: list[dict] = []
    current: _Run | None = None

    def finish(run: _Run | None) -> None:
        if run is None:
            return
        start_ms = int(run.first["timestamp_ms"])
        end_ms = int(run.last["timestamp_ms"])
        duration_ms = int(round(end_ms - start_ms + sample_interval_ms))
        if len(run.qualifying) < minimum_evidence_frames or duration_ms < minimum_duration_ms:
            return
        representative = run.representative
        events.append({
            "id": str(uuid4()),
            "type": run.error_type,
            "start_frame": int(run.first["frame_number"]),
            "end_frame": int(run.last["frame_number"]),
            "start_timestamp_ms": start_ms,
            "end_timestamp_ms": end_ms,
            "duration_ms": duration_ms,
            "evidence_frame_count": len(run.qualifying),
            "confidence": float(representative["probabilities"][run.error_type]),
            "representative_frame_number": int(representative["frame_number"]),
            "representative_timestamp_ms": int(representative["timestamp_ms"]),
            "measurements": representative.get("measurements", {}),
            "_representative_pose": representative.get("_pose"),
            "frame_image_path": None,
            "frame_image_url": None,
            "artifact_warning": None,
        })

    for frame in sorted(frames, key=lambda item: item["timestamp_ms"]):
        error_type = _qualifying_error(frame, threshold)
        if error_type is None:
            continue

        if current is not None:
            unqualified_gap_ms = (
                frame["timestamp_ms"]
                - current.last["timestamp_ms"]
                - sample_interval_ms
            )
            if error_type != current.error_type or unqualified_gap_ms > maximum_gap_ms + 1e-6:
                finish(current)
                current = None

        if current is None:
            current = _Run(error_type, frame, frame, frame, [frame])
            continue

        current.last = frame
        current.qualifying.append(frame)
        current_confidence = current.representative["probabilities"][error_type]
        frame_confidence = frame["probabilities"][error_type]
        if frame_confidence > current_confidence:
            current.representative = frame

    finish(current)
    return events


def public_event(event: dict, video_id: str) -> dict:
    output = {key: value for key, value in event.items() if not key.startswith("_")}
    if output.get("frame_image_path"):
        output["frame_image_url"] = (
            f"/api/videos/{video_id}/events/{event['id']}/frame"
        )
    output.pop("frame_image_path", None)
    return output

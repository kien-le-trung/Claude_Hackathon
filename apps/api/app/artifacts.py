from __future__ import annotations

import shutil
from pathlib import Path

import cv2


POSE_CONNECTIONS = (
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),
)


def _safe_child(root: Path, relative: Path) -> Path:
    resolved_root = root.resolve()
    candidate = (resolved_root / relative).resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise ValueError("Artifact path escapes configured root")
    return candidate


def artifact_path(root: Path, relative_path: str) -> Path:
    return _safe_child(root, Path(relative_path))


def _annotate(frame, event: dict) -> None:
    pose = event.get("_representative_pose") or {}
    landmarks = pose.get("landmarks", [])
    height, width = frame.shape[:2]

    def pixel(index: int):
        if index >= len(landmarks):
            return None
        item = landmarks[index]
        visibility = item.get("visibility")
        presence = item.get("presence")
        if visibility is not None and visibility < 0.5:
            return None
        if presence is not None and presence < 0.5:
            return None
        return int(round(item["x"] * width)), int(round(item["y"] * height))

    for first, second in POSE_CONNECTIONS:
        start, end = pixel(first), pixel(second)
        if start is not None and end is not None:
            cv2.line(frame, start, end, (60, 220, 255), 3, cv2.LINE_AA)
    for index in range(len(landmarks)):
        point = pixel(index)
        if point is not None:
            cv2.circle(frame, point, 4, (40, 80, 255), -1, cv2.LINE_AA)

    label = event["type"].replace("_", " ").title()
    text = f"{label}  {event['confidence'] * 100:.1f}%"
    cv2.rectangle(frame, (16, 16), (min(width - 16, 430), 64), (15, 23, 42), -1)
    cv2.putText(
        frame, text, (28, 50), cv2.FONT_HERSHEY_SIMPLEX,
        0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )


def create_event_artifacts(
    video_path: Path,
    events: list[dict],
    artifact_root: Path,
    video_id: str,
) -> None:
    """Create representative WebP images; individual failures remain nonfatal."""
    if not events:
        return
    video_directory = _safe_child(artifact_root, Path(video_id))
    video_directory.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            for event in events:
                event["artifact_warning"] = "Could not reopen source video"
            return
        for event in events:
            try:
                capture.set(cv2.CAP_PROP_POS_FRAMES, event["representative_frame_number"])
                decoded, frame = capture.read()
                if not decoded or frame is None:
                    raise ValueError("Representative source frame could not be decoded")
                _annotate(frame, event)
                height, width = frame.shape[:2]
                if width > 960:
                    scale = 960 / width
                    frame = cv2.resize(
                        frame,
                        (960, int(round(height * scale))),
                        interpolation=cv2.INTER_AREA,
                    )
                relative = Path(video_id) / f"{event['id']}.webp"
                destination = _safe_child(artifact_root, relative)
                success, encoded = cv2.imencode(
                    ".webp", frame, [cv2.IMWRITE_WEBP_QUALITY, 85]
                )
                if not success:
                    raise ValueError("OpenCV could not encode representative frame")
                temporary = destination.with_suffix(".tmp")
                temporary.write_bytes(encoded.tobytes())
                temporary.replace(destination)
                event["frame_image_path"] = relative.as_posix()
            except Exception as exc:
                event["artifact_warning"] = str(exc)[:300]
    finally:
        capture.release()


def delete_video_artifacts(artifact_root: Path, video_id: str) -> None:
    directory = _safe_child(artifact_root, Path(video_id))
    if directory.is_dir():
        shutil.rmtree(directory)

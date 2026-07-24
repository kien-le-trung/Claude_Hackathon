from app.events import consolidate_error_events


def frame(index, predicted="good", confidence=0.9):
    probabilities = {"good": 0.05, "bad_back": 0.025, "bad_heel": 0.025}
    probabilities[predicted] = confidence
    return {
        "frame_number": index * 3,
        "timestamp_ms": index * 100,
        "predicted_class": predicted,
        "probabilities": probabilities,
        "measurements": {"torso_lean_world": float(index)},
        "_pose": {"landmarks": []},
    }


def group(frames):
    return consolidate_error_events(
        frames, threshold=0.8, sampling_fps=10.0
    )


def test_isolated_spike_is_suppressed():
    assert group([frame(0), frame(1, "bad_back"), frame(2)]) == []


def test_three_stable_samples_form_one_event_and_choose_highest_confidence():
    events = group([
        frame(0, "bad_back", 0.81),
        frame(1, "bad_back", 0.95),
        frame(2, "bad_back", 0.90),
    ])
    assert len(events) == 1
    assert events[0]["duration_ms"] == 300
    assert events[0]["representative_timestamp_ms"] == 100
    assert events[0]["confidence"] == 0.95


def test_short_good_gap_is_bridged_but_other_error_is_hard_boundary():
    bridged = group([
        frame(0, "bad_back"), frame(1), frame(2), frame(3, "bad_back"),
        frame(4, "bad_back"), frame(5, "bad_back"),
    ])
    assert len(bridged) == 1
    split = group([
        frame(0, "bad_back"), frame(1, "bad_back"), frame(2, "bad_back"),
        frame(3, "bad_heel"), frame(4, "bad_heel"), frame(5, "bad_heel"),
    ])
    assert [event["type"] for event in split] == ["bad_back", "bad_heel"]


def test_threshold_is_inclusive_and_tie_uses_earliest_frame():
    events = group([
        frame(0, "bad_heel", 0.8),
        frame(1, "bad_heel", 0.8),
        frame(2, "bad_heel", 0.8),
    ])
    assert len(events) == 1
    assert events[0]["representative_timestamp_ms"] == 0

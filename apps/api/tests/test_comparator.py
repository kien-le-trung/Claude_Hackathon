from app.comparator import compare_squats


def frame(x_offset: float = 0.0) -> dict:
    return {
        "keypoints": [
            {"index": index, "x": float(index) + x_offset, "y": float(index * 2)}
            for index in range(18)
        ]
    }


def test_identical_sequences_score_highly() -> None:
    sequence = {"frames": [frame() for _ in range(4)]}
    result = compare_squats(sequence, sequence)
    assert result["score"] == 100.0
    assert result["details"]["detection_rate"] == 100.0


def test_empty_detection_returns_zero() -> None:
    user = {"frames": [{"keypoints": []}]}
    reference = {"frames": [frame()]}
    result = compare_squats(user, reference)
    assert result["score"] == 0
    assert result["details"]["frames_analyzed"] == 0

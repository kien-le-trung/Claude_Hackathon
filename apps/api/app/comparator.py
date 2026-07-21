"""Legacy cosine-similarity comparator preserved for Phase 1.

This module intentionally mirrors the existing Flask implementation so the
engineering refactor does not silently change scoring behavior.
"""

import numpy as np


def normalize_keypoints_coco(keypoints):
    if isinstance(keypoints, list):
        points = []
        for point in keypoints:
            if point is None:
                points.append([0, 0, 0])
            elif len(point) == 2:
                points.append([point[0], point[1], 1.0])
            else:
                points.append([point[0], point[1], point[2]])
        keypoints_array = np.asarray(points, dtype=np.float32)
    else:
        keypoints_array = keypoints.copy()

    if keypoints_array.shape[0] < 18:
        padding = np.zeros((18 - keypoints_array.shape[0], 3), dtype=np.float32)
        keypoints_array = np.vstack([keypoints_array, padding])

    left_hip, right_hip = 11, 8
    left_shoulder, right_shoulder = 5, 2
    confidence = (
        keypoints_array[:, 2]
        if keypoints_array.shape[1] > 2
        else np.ones(18, dtype=np.float32)
    )
    valid = confidence > 0.1

    if confidence[left_hip] > 0.1 and confidence[right_hip] > 0.1:
        root = (keypoints_array[left_hip, :2] + keypoints_array[right_hip, :2]) / 2.0
    elif np.any(valid):
        root = np.mean(keypoints_array[valid, :2], axis=0)
    else:
        root = np.array([0.0, 0.0])

    keypoints_array[:, :2] -= root

    if confidence[left_shoulder] > 0.1 and confidence[right_shoulder] > 0.1:
        shoulder_midpoint = (
            keypoints_array[left_shoulder, :2]
            + keypoints_array[right_shoulder, :2]
        ) / 2.0
    elif np.any(valid):
        shoulder_midpoint = np.mean(keypoints_array[valid, :2], axis=0)
    else:
        shoulder_midpoint = np.array([0.0, 0.0])

    scale = np.linalg.norm(shoulder_midpoint) + 1e-6
    keypoints_array[:, :2] /= scale
    keypoints_array[~valid, :2] = 0
    return keypoints_array[:, :2]


def pose_similarity_coco(model_keypoints, user_keypoints) -> float:
    model_vector = normalize_keypoints_coco(model_keypoints).flatten()
    user_vector = normalize_keypoints_coco(user_keypoints).flatten()
    denominator = np.linalg.norm(model_vector) * np.linalg.norm(user_vector)
    if denominator < 1e-8:
        return 0.0
    cosine_similarity = np.dot(model_vector, user_vector) / denominator
    return float((cosine_similarity + 1) / 2.0)


def _frame_to_array(frame: dict) -> list:
    indexed = {
        keypoint["index"]: (keypoint["x"], keypoint["y"])
        for keypoint in frame["keypoints"]
    }
    return [indexed.get(index) for index in range(18)]


def compare_squats(user_data: dict, model_data: dict) -> dict:
    user_frames = user_data["frames"]
    model_frames = model_data["frames"]
    valid_user_frames = [frame for frame in user_frames if frame["keypoints"]]
    valid_model_frames = [frame for frame in model_frames if frame["keypoints"]]

    if not valid_user_frames or not valid_model_frames:
        return {
            "score": 0,
            "feedback": "Could not detect pose in video. Please ensure you are fully visible in the frame.",
            "details": {
                "detection_rate": 0,
                "frames_analyzed": 0,
                "total_frames": len(user_frames),
                "consistency": 0,
            },
        }

    sample_count = min(20, len(valid_user_frames), len(valid_model_frames))
    similarities = []
    for index in range(sample_count):
        user_index = int(index * len(valid_user_frames) / sample_count)
        model_index = int(index * len(valid_model_frames) / sample_count)
        similarities.append(
            pose_similarity_coco(
                _frame_to_array(valid_model_frames[model_index]),
                _frame_to_array(valid_user_frames[user_index]),
            )
        )

    average = float(np.mean(similarities))
    standard_deviation = float(np.std(similarities))
    detection_rate = len(valid_user_frames) / len(user_frames) * 100
    score = average * 100

    if score >= 85:
        feedback = ["Excellent form! Very similar to the model squat."]
    elif score >= 75:
        feedback = ["Good form! Minor differences from the model."]
    elif score >= 60:
        feedback = ["Fair form. Several areas could be improved."]
    else:
        feedback = ["Form needs work. Significant differences from the model."]

    if detection_rate < 80:
        feedback.append("Make sure you stay fully visible in the frame throughout the movement.")
    if standard_deviation > 0.15:
        feedback.append("Try to maintain consistent form throughout the movement.")
    else:
        feedback.append("Great consistency throughout the movement!")

    return {
        "score": round(score, 1),
        "feedback": " ".join(feedback),
        "details": {
            "detection_rate": round(detection_rate, 1),
            "frames_analyzed": len(valid_user_frames),
            "total_frames": len(user_frames),
            "consistency": round((1 - standard_deviation) * 100, 1),
        },
    }

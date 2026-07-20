"""
pose_similarity.py

This module provides a function to:
1. Normalize OpenPose keypoint data (translation + scale invariance)
2. Compute full-pose similarity using cosine similarity (Algorithm 3)

---------------------------------------------------------------
EXPECTED INPUT FORMAT
---------------------------------------------------------------
The function expects keypoints in the following form:

    keypoints: numpy array of shape (N, 3)

Where:
    - N is the number of body joints (e.g., 25 for BODY_25)
    - Each keypoint is [x, y, confidence]
    - Missing points may have confidence = 0 → they will be ignored

Example shape for BODY_25:
    (25, 3)

---------------------------------------------------------------
DATA ASSUMPTIONS
---------------------------------------------------------------
- Coordinates are 2D screen coordinates from OpenPose.
- Root joint used for centering = mid-hip.
  For BODY_25, left hip = 9, right hip = 12.
- Scale normalization is done using hip-to-shoulder distance.
"""

import numpy as np


def normalize_keypoints(keypoints):
    """
    Normalize OpenPose keypoints:
    1. Remove low-confidence points
    2. Translate so the root (mid-hip) is at the origin
    3. Scale by shoulder-to-hip distance for size invariance

    Parameters
    ----------
    keypoints : np.ndarray of shape (N, 3)
        Array of [x, y, confidence] values.

    Returns
    -------
    np.ndarray of shape (N, 2)
        Normalized (x, y) coordinates. Low-confidence points become [0, 0].
    """

    kp = keypoints.copy()

    # Confidence mask
    conf = kp[:, 2]
    valid = conf > 0.1

    # Body_25 joint indices (modify if using COCO)
    L_HIP = 9
    R_HIP = 12
    L_SHOULDER = 5
    R_SHOULDER = 2

    # Compute root joint = mid-hip
    if conf[L_HIP] > 0.1 and conf[R_HIP] > 0.1:
        root = (kp[L_HIP, :2] + kp[R_HIP, :2]) / 2.0
    else:
        # fallback: mean of all confident points
        root = np.mean(kp[valid, :2], axis=0)

    # Translation: subtract root from all joints
    kp[:, :2] -= root

    # Compute scale (distance between mid-hip and mid-shoulder)
    if conf[L_SHOULDER] > 0.1 and conf[R_SHOULDER] > 0.1:
        shoulder_mid = (kp[L_SHOULDER, :2] + kp[R_SHOULDER, :2]) / 2.0
    else:
        # fallback: mean of confident joints
        shoulder_mid = np.mean(kp[valid, :2], axis=0)

    scale = np.linalg.norm(shoulder_mid) + 1e-6  # avoid divide-by-zero

    kp[:, :2] /= scale

    # Zero-out invalid joints
    kp[~valid, :2] = 0

    return kp[:, :2]  # return only x,y normalized


def pose_similarity_full(model_kp, user_kp):
    """
    Compute full-pose similarity using cosine similarity.

    Parameters
    ----------
    model_kp : np.ndarray (N, 3)
        Model keypoints (x,y,confidence)
    user_kp : np.ndarray (N, 3)
        User keypoints (x,y,confidence)

    Returns
    -------
    float
        Cosine similarity between 0 and 1 (higher = more similar).
    """

    # Normalize both poses
    model_norm = normalize_keypoints(model_kp)   # shape (N, 2)
    user_norm = normalize_keypoints(user_kp)

    # Flatten into pose vectors
    model_vec = model_norm.flatten()
    user_vec = user_norm.flatten()

    # Cosine similarity
    dot = np.dot(model_vec, user_vec)
    norm = np.linalg.norm(model_vec) * np.linalg.norm(user_vec)

    if norm < 1e-8:
        return 0.0

    cosine_sim = dot / norm

    # Map [-1,1] → [0,1] to ensure positivity
    return (cosine_sim + 1) / 2.0

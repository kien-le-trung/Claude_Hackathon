#!/usr/bin/env python3
"""
Compare user's squat video with the model squat using pose similarity.
Uses cosine similarity to compare normalized pose keypoints.
"""

import cv2
import numpy as np
import json
import os
from openpose_model import OpenPoseModel


def normalize_keypoints_coco(keypoints):
    """
    Normalize OpenPose COCO keypoints (18 points):
    1. Remove low-confidence points
    2. Translate so the root (mid-hip) is at the origin
    3. Scale by shoulder-to-hip distance for size invariance

    COCO Format indices:
    - Left Hip: 11
    - Right Hip: 8
    - Left Shoulder: 5
    - Right Shoulder: 2

    Parameters
    ----------
    keypoints : list of tuples or np.ndarray
        List of (x, y) or (x, y, confidence) for 18 COCO keypoints

    Returns
    -------
    np.ndarray of shape (18, 2)
        Normalized (x, y) coordinates. Missing points become [0, 0].
    """
    # Convert to numpy array with confidence
    if isinstance(keypoints, list):
        # Handle list of tuples (x, y) - add confidence of 1.0
        kp_list = []
        for pt in keypoints:
            if pt is not None:
                if len(pt) == 2:
                    kp_list.append([pt[0], pt[1], 1.0])
                else:
                    kp_list.append([pt[0], pt[1], pt[2]])
            else:
                kp_list.append([0, 0, 0])
        kp = np.array(kp_list, dtype=np.float32)
    else:
        kp = keypoints.copy()

    # Ensure we have 18 keypoints
    if kp.shape[0] < 18:
        # Pad with zeros if needed
        padding = np.zeros((18 - kp.shape[0], 3), dtype=np.float32)
        kp = np.vstack([kp, padding])

    # COCO joint indices
    L_HIP = 11
    R_HIP = 8
    L_SHOULDER = 5
    R_SHOULDER = 2

    # Confidence mask
    conf = kp[:, 2] if kp.shape[1] > 2 else np.ones(18)
    valid = conf > 0.1

    # Compute root joint = mid-hip
    if conf[L_HIP] > 0.1 and conf[R_HIP] > 0.1:
        root = (kp[L_HIP, :2] + kp[R_HIP, :2]) / 2.0
    elif np.any(valid):
        # fallback: mean of all confident points
        root = np.mean(kp[valid, :2], axis=0)
    else:
        root = np.array([0.0, 0.0])

    # Translation: subtract root from all joints
    kp[:, :2] -= root

    # Compute scale (distance between mid-hip and mid-shoulder)
    if conf[L_SHOULDER] > 0.1 and conf[R_SHOULDER] > 0.1:
        shoulder_mid = (kp[L_SHOULDER, :2] + kp[R_SHOULDER, :2]) / 2.0
    elif np.any(valid):
        # fallback: mean of confident joints
        shoulder_mid = np.mean(kp[valid, :2], axis=0)
    else:
        shoulder_mid = np.array([0.0, 0.0])

    scale = np.linalg.norm(shoulder_mid) + 1e-6  # avoid divide-by-zero

    kp[:, :2] /= scale

    # Zero-out invalid joints
    kp[~valid, :2] = 0

    return kp[:, :2]  # return only x,y normalized


def pose_similarity_coco(model_kp, user_kp):
    """
    Compute full-pose similarity using cosine similarity for COCO keypoints.

    Parameters
    ----------
    model_kp : list or np.ndarray
        Model keypoints (18 COCO points)
    user_kp : list or np.ndarray
        User keypoints (18 COCO points)

    Returns
    -------
    float
        Cosine similarity between 0 and 1 (higher = more similar).
    """
    # Normalize both poses
    model_norm = normalize_keypoints_coco(model_kp)
    user_norm = normalize_keypoints_coco(user_kp)

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


def extract_user_keypoints(video_path, model):
    """
    Extract pose keypoints from user video.

    Args:
        video_path: Path to user video
        model: OpenPoseModel instance

    Returns:
        Dictionary with fps, total_frames, and frames data
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing user video: {os.path.basename(video_path)}")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {fps}")

    all_keypoints = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        try:
            keypoints, _ = model.detect(frame)

            frame_data = {
                'frame_number': frame_count,
                'timestamp': frame_count / fps,
                'keypoints': []
            }

            # Convert keypoints to serializable format
            for i, keypoint in enumerate(keypoints):
                if keypoint is not None:
                    frame_data['keypoints'].append({
                        'body_part': model.get_body_part_name(i),
                        'index': i,
                        'x': float(keypoint[0]),
                        'y': float(keypoint[1])
                    })

            all_keypoints.append(frame_data)

            if frame_count % 30 == 0:
                print(f"  Processed {frame_count}/{total_frames} frames...")

        except Exception as e:
            all_keypoints.append({
                'frame_number': frame_count,
                'timestamp': frame_count / fps,
                'keypoints': [],
                'error': str(e)
            })

    cap.release()
    print(f"  Completed: {frame_count} frames processed\n")

    return {
        'fps': fps,
        'total_frames': total_frames,
        'duration': total_frames / fps,
        'frames': all_keypoints
    }


def convert_frame_keypoints_to_array(frame_data):
    """
    Convert frame keypoints from dictionary format to numpy array.

    Args:
        frame_data: Frame data with keypoints list

    Returns:
        numpy array of shape (18, 2) with keypoint positions
    """
    kp_array = []
    keypoints_dict = {kp['index']: (kp['x'], kp['y']) for kp in frame_data['keypoints']}

    for i in range(18):  # COCO has 18 keypoints
        if i in keypoints_dict:
            kp_array.append(keypoints_dict[i])
        else:
            kp_array.append(None)

    return kp_array


def compare_videos(user_data, model_data):
    """
    Compare user video with model video using pose similarity.

    Args:
        user_data: User's keypoints data
        model_data: Model's keypoints data

    Returns:
        Dictionary with comparison results
    """
    print("="*60)
    print("Comparing User Squat with Model Squat")
    print("="*60)

    user_frames = [f for f in user_data['frames'] if len(f['keypoints']) > 0]
    model_frames = [f for f in model_data['frames'] if len(f['keypoints']) > 0]

    print(f"User frames with pose detected: {len(user_frames)}/{len(user_data['frames'])}")
    print(f"Model frames with pose detected: {len(model_frames)}/{len(model_data['frames'])}")

    if not user_frames or not model_frames:
        return {
            'error': 'Insufficient pose detections',
            'user_detection_rate': len(user_frames) / len(user_data['frames']) * 100,
            'model_detection_rate': len(model_frames) / len(model_data['frames']) * 100
        }

    # Sample frames at similar time points
    num_samples = min(20, len(user_frames), len(model_frames))
    similarities = []

    print(f"\nComparing {num_samples} frame pairs...\n")

    for i in range(num_samples):
        user_idx = int(i * len(user_frames) / num_samples)
        model_idx = int(i * len(model_frames) / num_samples)

        user_frame = user_frames[user_idx]
        model_frame = model_frames[model_idx]

        # Convert to arrays
        user_kp = convert_frame_keypoints_to_array(user_frame)
        model_kp = convert_frame_keypoints_to_array(model_frame)

        # Calculate similarity
        similarity = pose_similarity_coco(model_kp, user_kp)
        similarities.append(similarity)

        if i % 5 == 0:
            print(f"  Sample {i+1}/{num_samples}: Similarity = {similarity:.3f} "
                  f"(User t={user_frame['timestamp']:.1f}s, Model t={model_frame['timestamp']:.1f}s)")

    # Calculate statistics
    avg_similarity = np.mean(similarities)
    min_similarity = np.min(similarities)
    max_similarity = np.max(similarities)
    std_similarity = np.std(similarities)

    # Convert similarity to score (0-100)
    score = avg_similarity * 100

    print(f"\n{'='*60}")
    print("Comparison Results")
    print(f"{'='*60}")
    print(f"Average Similarity: {avg_similarity:.3f}")
    print(f"Score: {score:.1f}/100")
    print(f"Min Similarity: {min_similarity:.3f}")
    print(f"Max Similarity: {max_similarity:.3f}")
    print(f"Std Dev: {std_similarity:.3f}")
    print(f"Consistency: {(1 - std_similarity):.3f}")
    print(f"{'='*60}\n")

    # Generate feedback
    feedback = []
    if score >= 85:
        feedback.append("Excellent form! Very similar to the model squat.")
    elif score >= 75:
        feedback.append("Good form! Minor differences from the model.")
    elif score >= 60:
        feedback.append("Fair form. Several areas could be improved.")
    else:
        feedback.append("Form needs work. Significant differences from the model.")

    if std_similarity > 0.15:
        feedback.append("Form consistency could be improved - try to maintain the same technique throughout.")
    else:
        feedback.append("Great consistency throughout the movement!")

    return {
        'average_similarity': float(avg_similarity),
        'score': float(score),
        'min_similarity': float(min_similarity),
        'max_similarity': float(max_similarity),
        'std_similarity': float(std_similarity),
        'consistency': float(1 - std_similarity),
        'num_samples': num_samples,
        'user_detection_rate': len(user_frames) / len(user_data['frames']) * 100,
        'model_detection_rate': len(model_frames) / len(model_data['frames']) * 100,
        'feedback': ' '.join(feedback),
        'frame_similarities': [float(s) for s in similarities]
    }


def main():
    """Main comparison function."""
    # Paths
    user_video_path = "assets/test_vid_2.mp4"
    model_analysis_path = "output/model_squat_analysis.json"
    output_path = "output/test_vid_comparison.json"

    print("="*60)
    print("Squat Comparison Analysis")
    print("="*60)
    print(f"User video: {user_video_path}")
    print(f"Model data: {model_analysis_path}")
    print(f"="*60 + "\n")

    # Check files exist
    if not os.path.exists(user_video_path):
        print(f"ERROR: User video not found: {user_video_path}")
        return

    if not os.path.exists(model_analysis_path):
        print(f"ERROR: Model analysis not found: {model_analysis_path}")
        print("Please run: python analyze_model_squat.py first")
        return

    # Initialize OpenPose model
    print("Initializing OpenPose model...")
    model = OpenPoseModel()

    model_path = os.path.join('models', 'pose_iter_440000.caffemodel')
    proto_path = os.path.join('models', 'pose_deploy_linevec.prototxt')

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found. Please run: python download_openpose_model.py")
        return

    model.load_model(model_path, proto_path)
    print("Model loaded successfully!\n")

    # Load model squat data
    print("Loading model squat analysis...")
    with open(model_analysis_path, 'r') as f:
        model_data = json.load(f)
    print(f"  Model video: {model_data['video_info']['total_frames']} frames, "
          f"{model_data['video_info']['duration_seconds']:.1f}s\n")

    # Extract user video keypoints
    print("Analyzing user video...")
    user_data = extract_user_keypoints(user_video_path, model)

    # Compare videos
    results = compare_videos(user_data, model_data)

    # Save results
    output_data = {
        'user_video': user_video_path,
        'model_analysis': model_analysis_path,
        'user_data': user_data,
        'comparison_results': results
    }

    os.makedirs('output', exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Results saved to: {output_path}\n")

    # Display final summary
    print("="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"User Score: {results['score']:.1f}/100")
    print(f"Feedback: {results['feedback']}")
    print(f"Detection Rate: {results['user_detection_rate']:.1f}%")
    print(f"Consistency: {results['consistency']:.3f}")
    print("="*60)


if __name__ == "__main__":
    main()

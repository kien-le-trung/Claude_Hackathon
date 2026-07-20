#!/usr/bin/env python3
"""
Analyze the model squat video using OpenPose.
This script extracts pose keypoints from the exemplary squat form video.
"""

import cv2
import numpy as np
from openpose_model import OpenPoseModel
import json
import os


def analyze_video(video_path, model, output_dir="output"):
    """
    Analyze a video and extract pose keypoints from each frame.

    Args:
        video_path: Path to the video file
        model: OpenPoseModel instance
        output_dir: Directory to save output files

    Returns:
        List of keypoints for each frame
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"\n{'='*60}")
    print(f"Video Analysis: {os.path.basename(video_path)}")
    print(f"{'='*60}")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")
    print(f"{'='*60}\n")

    all_keypoints = []
    frame_count = 0

    # Prepare video writer for output (use H.264 for web browser compatibility)
    output_video_path = os.path.join(output_dir, f"analyzed_{os.path.basename(video_path)}")
    # Try H.264 codec first (best browser support), fallback to mp4v if not available
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')  # H.264 codec
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
        if not out.isOpened():
            raise Exception("avc1 codec failed")
        print("  Using H.264 (avc1) codec for web compatibility")
    except:
        print("  Warning: H.264 codec not available, using mp4v (may not work in browsers)")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    print("Processing frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Detect pose
        try:
            keypoints, heatmap = model.detect(frame)

            # Draw skeleton on frame
            output_frame = model.draw_skeleton(frame.copy(), keypoints)

            # Write annotated frame
            out.write(output_frame)

            # Store keypoints
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
                        'y': float(keypoint[1]),
                        'confidence': float(keypoint[2]) if len(keypoint) > 2 else 1.0
                    })

            all_keypoints.append(frame_data)

            # Print progress
            if frame_count % 10 == 0 or frame_count == total_frames:
                print(f"  Processed {frame_count}/{total_frames} frames ({frame_count/total_frames*100:.1f}%)")

        except Exception as e:
            print(f"  Warning: Error processing frame {frame_count}: {e}")
            all_keypoints.append({
                'frame_number': frame_count,
                'timestamp': frame_count / fps,
                'keypoints': [],
                'error': str(e)
            })

    cap.release()
    out.release()

    print(f"\nVideo analysis complete!")
    print(f"  Annotated video saved to: {output_video_path}")

    return all_keypoints, fps, total_frames


def save_analysis_results(keypoints_data, fps, total_frames, output_path="output/model_squat_analysis.json"):
    """Save keypoints analysis to JSON file."""
    results = {
        'video_info': {
            'fps': fps,
            'total_frames': total_frames,
            'duration_seconds': total_frames / fps
        },
        'frames': keypoints_data
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"  Analysis data saved to: {output_path}")
    return output_path


def print_summary(keypoints_data):
    """Print a summary of the detected poses."""
    print(f"\n{'='*60}")
    print("Analysis Summary")
    print(f"{'='*60}")

    # Count frames with detected poses
    frames_with_poses = sum(1 for frame in keypoints_data if len(frame['keypoints']) > 0)
    total_frames = len(keypoints_data)

    print(f"  Total frames: {total_frames}")
    print(f"  Frames with detected poses: {frames_with_poses}")
    print(f"  Detection rate: {frames_with_poses/total_frames*100:.1f}%")

    # Show sample keypoints from first frame with detection
    for frame in keypoints_data:
        if len(frame['keypoints']) > 0:
            print(f"\n  Sample keypoints from frame {frame['frame_number']}:")
            for kp in frame['keypoints'][:5]:  # Show first 5 keypoints
                print(f"    {kp['body_part']:15s} -> ({kp['x']:6.1f}, {kp['y']:6.1f})")
            if len(frame['keypoints']) > 5:
                print(f"    ... and {len(frame['keypoints']) - 5} more keypoints")
            break

    print(f"{'='*60}\n")


def main():
    """Main function to analyze the model squat video."""
    video_path = "assets/test_vid_2.mp4"

    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}")
        return

    print("Initializing OpenPose model...")
    model = OpenPoseModel()

    if not model.model_loaded:
        print("\nModel weights not loaded. Downloading...")
        print("This may take a few minutes (model is ~200MB)...\n")
        try:
            proto_path, model_path = model.download_model()
            model.load_model(model_path, proto_path)
            print("Model loaded successfully!\n")
        except Exception as e:
            print(f"ERROR: Failed to download/load model: {e}")
            print("\nPlease ensure you have internet connection and try again.")
            return

    # Analyze the video
    try:
        keypoints_data, fps, total_frames = analyze_video(video_path, model)

        # Save results
        save_analysis_results(keypoints_data, fps, total_frames)

        # Print summary
        print_summary(keypoints_data)

        print("Next steps:")
        print("  1. Review the annotated video in output/analyzed_model_squat.mp4")
        print("  2. Check the keypoints data in output/model_squat_analysis.json")
        print("  3. Use this data as reference for comparing user squat videos")

    except Exception as e:
        print(f"\n✗ Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

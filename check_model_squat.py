#!/usr/bin/env python3
"""
Check the model squat video properties.
This script provides basic information about the video without requiring OpenPose.
"""

import cv2
import os


def analyze_video_info(video_path):
    """Get basic information about the video."""
    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found at {video_path}")
        return None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        return None

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = total_frames / fps if fps > 0 else 0

    print("="*60)
    print("Model Squat Video Information")
    print("="*60)
    print(f"  File: {os.path.abspath(video_path)}")
    print(f"  File size: {os.path.getsize(video_path) / 1024 / 1024:.2f} MB")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps:.2f}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {duration:.2f} seconds")
    print("="*60)

    # Extract some sample frames
    print("\nExtracting sample frames...")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # Extract frames at different time points
    sample_points = [0.0, 0.25, 0.5, 0.75, 1.0]  # Start, 25%, 50%, 75%, End
    saved_frames = []

    for point in sample_points:
        if point == 1.0:
            frame_num = max(0, total_frames - 1)
        else:
            frame_num = int(total_frames * point)

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()

        if ret:
            timestamp = frame_num / fps
            output_path = os.path.join(output_dir, f"model_squat_frame_{int(point*100):03d}pct_t{timestamp:.2f}s.jpg")
            cv2.imwrite(output_path, frame)
            saved_frames.append((point * 100, timestamp, output_path))
            print(f"  Frame at {point*100:5.1f}% (t={timestamp:5.2f}s) -> {os.path.basename(output_path)}")

    cap.release()

    print("\n" + "="*60)
    print("Sample frames saved to:", os.path.abspath(output_dir))
    print("="*60)
    print("\nNext steps:")
    print("  1. Review the extracted frames to understand the squat motion")
    print("  2. Download OpenPose model weights:")
    print("     python download_openpose_model.py")
    print("  3. Run full analysis:")
    print("     python analyze_model_squat.py")
    print("="*60)

    return {
        'fps': fps,
        'total_frames': total_frames,
        'width': width,
        'height': height,
        'duration': duration,
        'sample_frames': saved_frames
    }


def main():
    video_path = "assets/model_squat.mp4"
    info = analyze_video_info(video_path)

    if info:
        print(f"\nVideo has {info['total_frames']} frames over {info['duration']:.2f} seconds")
        print("This gives us a good dataset to establish baseline squat form.")


if __name__ == "__main__":
    main()

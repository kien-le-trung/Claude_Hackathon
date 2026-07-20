#!/usr/bin/env python3
"""
Convert videos to web-compatible format (H.264 codec).
This fixes the issue where videos don't play in browsers.
"""

import cv2
import os


def convert_video_to_h264(input_path, output_path=None):
    """
    Convert video to H.264 codec for web browser compatibility.

    Args:
        input_path: Path to input video
        output_path: Path for output video (optional, defaults to input_path with _web suffix)

    Returns:
        Path to converted video
    """
    if output_path is None:
        # Create output path with _web suffix
        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_web{ext}"

    print(f"\nConverting: {input_path}")
    print(f"Output: {output_path}")

    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")

    # Try different H.264 codec options
    # avc1 is the most compatible H.264 codec for web browsers
    codec_options = [
        ('avc1', 'H.264 (avc1)'),
        ('H264', 'H.264 (H264)'),
        ('X264', 'H.264 (X264)'),
        ('h264', 'H.264 (h264)'),
    ]

    out = None
    used_codec = None

    for codec, codec_name in codec_options:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if out.isOpened():
                print(f"  Using codec: {codec_name}")
                used_codec = codec
                break
            else:
                out.release()
                out = None
        except Exception as e:
            print(f"  Failed to use {codec_name}: {e}")
            continue

    if out is None or not out.isOpened():
        print("\n  ERROR: None of the H.264 codecs are available!")
        print("  Falling back to mp4v (may not work in all browsers)")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        used_codec = 'mp4v'

    # Process frames
    frame_count = 0
    print("  Processing frames...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        out.write(frame)

        if frame_count % 30 == 0 or frame_count == total_frames:
            print(f"    Progress: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")

    cap.release()
    out.release()

    print(f"  ✓ Conversion complete!")
    print(f"  Codec used: {used_codec}")

    return output_path


def main():
    """Convert all videos in output directory to web-compatible format."""
    output_dir = "output"

    videos_to_convert = [
        "analyzed_model_squat.mp4",
        "analyzed_test_vid_2.mp4"
    ]

    print("="*60)
    print("Converting Videos to Web-Compatible Format")
    print("="*60)

    for video_name in videos_to_convert:
        input_path = os.path.join(output_dir, video_name)

        if not os.path.exists(input_path):
            print(f"\nSkipping {video_name} (not found)")
            continue

        # Convert and replace original
        temp_output = os.path.join(output_dir, f"temp_{video_name}")

        try:
            convert_video_to_h264(input_path, temp_output)

            # Replace original with converted version
            if os.path.exists(temp_output):
                # Backup original
                backup_path = os.path.join(output_dir, f"backup_{video_name}")
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                os.rename(input_path, backup_path)

                # Move converted to original name
                os.rename(temp_output, input_path)
                print(f"  ✓ Replaced original (backup saved as backup_{video_name})")

        except Exception as e:
            print(f"  ✗ Error converting {video_name}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*60)
    print("Conversion Complete!")
    print("="*60)
    print("\nYou can now reload your browser to see the videos.")


if __name__ == "__main__":
    main()

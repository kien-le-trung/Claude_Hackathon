#!/usr/bin/env python3
"""
Download OpenPose model files.
This is a standalone script to download the required model files.
"""

import os
import urllib.request
import sys


def download_with_progress(url, output_path):
    """Download a file with progress bar."""
    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        sys.stdout.write(f"\r  Progress: {percent}% ({count * block_size / 1024 / 1024:.1f}MB / {total_size / 1024 / 1024:.1f}MB)")
        sys.stdout.flush()

    print(f"Downloading to: {output_path}")
    urllib.request.urlretrieve(url, output_path, reporthook)
    print("\n  Download complete!")


def main():
    """Download OpenPose model files."""
    output_dir = "models"
    os.makedirs(output_dir, exist_ok=True)

    # URLs for COCO model - using alternative mirror
    proto_url = "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt"
    # Alternative URL from SNU mirror (CMU server appears to be down)
    model_url = "http://vcl.snu.ac.kr/OpenPose/models/pose/coco/pose_iter_440000.caffemodel"

    proto_path = os.path.join(output_dir, "pose_deploy_linevec.prototxt")
    model_path = os.path.join(output_dir, "pose_iter_440000.caffemodel")

    print("="*60)
    print("OpenPose Model Downloader")
    print("="*60)
    print("\nThis will download:")
    print(f"  1. Config file (~30KB): {proto_path}")
    print(f"  2. Model weights (~200MB): {model_path}")
    print("\nNote: The model file is large and may take several minutes to download.")
    print("="*60)

    # Download prototxt
    if os.path.exists(proto_path):
        print(f"\nConfig file already exists: {proto_path}")
    else:
        print(f"\n[1/2] Downloading config file...")
        try:
            download_with_progress(proto_url, proto_path)
        except Exception as e:
            print(f"\nERROR downloading config file: {e}")
            return False

    # Download caffemodel
    if os.path.exists(model_path):
        print(f"\nModel weights already exist: {model_path}")
        file_size = os.path.getsize(model_path) / 1024 / 1024
        print(f"  File size: {file_size:.1f}MB")
    else:
        print(f"\n[2/2] Downloading model weights (~200MB)...")
        print("This may take several minutes depending on your connection...")
        try:
            download_with_progress(model_url, model_path)
        except Exception as e:
            print(f"\nERROR downloading model weights: {e}")
            print("\nIf download fails, you can manually download from:")
            print(f"  {model_url}")
            print(f"And save it to: {model_path}")
            return False

    print("\n" + "="*60)
    print("Download Complete!")
    print("="*60)
    print(f"\nModel files saved to: {os.path.abspath(output_dir)}/")
    print(f"  - {os.path.basename(proto_path)}")
    print(f"  - {os.path.basename(model_path)}")
    print("\nYou can now run the analysis script:")
    print("  python analyze_model_squat.py")
    print("="*60)

    return True


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user.")
        sys.exit(1)

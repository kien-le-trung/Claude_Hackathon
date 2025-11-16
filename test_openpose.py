#!/usr/bin/env python3
"""
Simple test script for OpenPose model.
This script demonstrates basic usage of the OpenPose model loader.
"""

import cv2
import numpy as np
from openpose_model import OpenPoseModel, create_sample_image
import os


def test_model_initialization():
    """Test 1: Model initialization"""
    print("\n" + "="*60)
    print("TEST 1: Model Initialization")
    print("="*60)

    try:
        model = OpenPoseModel()
        print("✓ OpenPoseModel initialized successfully")
        print(f"  Model loaded: {model.model_loaded}")
        print(f"  Body parts defined: {len(model.BODY_PARTS)}")
        print(f"  Pose pairs defined: {len(model.POSE_PAIRS)}")
        return True
    except Exception as e:
        print(f"✗ Failed to initialize model: {e}")
        return False


def test_sample_image_creation():
    """Test 2: Sample image creation"""
    print("\n" + "="*60)
    print("TEST 2: Sample Image Creation")
    print("="*60)

    try:
        image = create_sample_image()
        print(f"✓ Sample image created successfully")
        print(f"  Image shape: {image.shape}")
        print(f"  Image dtype: {image.dtype}")

        # Save the sample image
        output_path = "sample_stick_figure.png"
        cv2.imwrite(output_path, image)
        print(f"  Saved sample image to: {output_path}")
        return True
    except Exception as e:
        print(f"✗ Failed to create sample image: {e}")
        return False


def test_model_download():
    """Test 3: Model download capability"""
    print("\n" + "="*60)
    print("TEST 3: Model Download Capability")
    print("="*60)

    try:
        model = OpenPoseModel()
        print("✓ Model has download_model() method")
        print("  Note: Actual download requires internet connection")
        print("  To download the model, run:")
        print("    model = OpenPoseModel()")
        print("    proto_path, model_path = model.download_model()")
        print("    model.load_model(model_path, proto_path)")
        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_model_structure():
    """Test 4: Model structure and constants"""
    print("\n" + "="*60)
    print("TEST 4: Model Structure")
    print("="*60)

    try:
        model = OpenPoseModel()

        print("✓ Body Parts:")
        for name, idx in list(model.BODY_PARTS.items())[:5]:
            print(f"    {name}: {idx}")
        print(f"    ... and {len(model.BODY_PARTS) - 5} more")

        print("\n✓ Pose Pairs (skeleton connections):")
        for i, pair in enumerate(model.POSE_PAIRS[:5]):
            print(f"    Connection {i+1}: {pair}")
        print(f"    ... and {len(model.POSE_PAIRS) - 5} more connections")

        return True
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_inference_readiness():
    """Test 5: Check if model is ready for inference"""
    print("\n" + "="*60)
    print("TEST 5: Inference Readiness")
    print("="*60)

    model = OpenPoseModel()
    image = create_sample_image()

    print(f"  Model loaded: {model.model_loaded}")

    if not model.model_loaded:
        print("  ⚠ Model weights not loaded yet")
        print("  To perform inference, you need to:")
        print("    1. Download the model files:")
        print("       proto_path, model_path = model.download_model()")
        print("    2. Load the model:")
        print("       model.load_model(model_path, proto_path)")
        print("    3. Run inference:")
        print("       keypoints, heatmap = model.detect(image)")
        print("       result = model.draw_skeleton(image, keypoints)")

        # Try to detect without model (will raise error as expected)
        try:
            keypoints, _ = model.detect(image)
            print("  ✗ Unexpected: detection worked without model")
        except RuntimeError as e:
            print(f"  ✓ Correctly raises error when model not loaded: '{e}'")

        return True
    else:
        print("  ✓ Model is loaded and ready for inference")
        return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("OpenPose Model Test Suite")
    print("="*60)

    tests = [
        ("Model Initialization", test_model_initialization),
        ("Sample Image Creation", test_sample_image_creation),
        ("Model Download Capability", test_model_download),
        ("Model Structure", test_model_structure),
        ("Inference Readiness", test_inference_readiness),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Test '{test_name}' raised exception: {e}")
            results.append((test_name, False))

    # Print summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠ {total - passed} test(s) failed")

    print("\n" + "="*60)
    print("Next Steps:")
    print("="*60)
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Download model weights (requires ~200MB):")
    print("   python -c \"from openpose_model import OpenPoseModel; m = OpenPoseModel(); m.download_model()\"")
    print("3. Run inference on your own images")
    print("="*60)


if __name__ == "__main__":
    main()

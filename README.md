# OpenPose Model Implementation

A simple Python implementation for importing and using OpenPose models for human pose estimation.

## Overview

This project provides a clean interface for loading and using OpenPose models to detect human pose keypoints in images. It uses OpenCV's DNN module with pre-trained COCO models.

## Features

- Easy model loading and initialization
- Support for both Caffe and TensorFlow models
- Automatic model downloading from official sources
- GPU acceleration support (when available)
- Simple API for pose detection
- Skeleton visualization

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Claude_Hackathon
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Running the Test Script

```bash
python test_openpose.py
```

This will run a series of tests to verify the model implementation.

### Basic Usage

```python
from openpose_model import OpenPoseModel
import cv2

# Initialize model
model = OpenPoseModel()

# Download pre-trained model (first time only)
proto_path, model_path = model.download_model()

# Load the model
model.load_model(model_path, proto_path)

# Load an image
image = cv2.imread('your_image.jpg')

# Detect pose keypoints
keypoints, heatmap = model.detect(image)

# Draw skeleton on image
result = model.draw_skeleton(image, keypoints)

# Save or display result
cv2.imwrite('result.jpg', result)
```

## Model Details

### Body Parts Detected

The model detects 18 body keypoints:
- Nose, Neck
- Shoulders (Left/Right)
- Elbows (Left/Right)
- Wrists (Left/Right)
- Hips (Left/Right)
- Knees (Left/Right)
- Ankles (Left/Right)
- Eyes (Left/Right)
- Ears (Left/Right)

### Model Files

The implementation uses the COCO model from CMU's OpenPose:
- **Config**: `pose_deploy_linevec.prototxt`
- **Weights**: `pose_iter_440000.caffemodel` (~200MB)

These files are automatically downloaded when you call `download_model()`.

## Project Structure

```
Claude_Hackathon/
├── openpose_model.py      # Main OpenPose model implementation
├── test_openpose.py       # Test script
├── requirements.txt       # Python dependencies
├── models/               # Directory for model files (created automatically)
├── .gitignore           # Git ignore rules
└── README.md            # This file
```

## API Reference

### OpenPoseModel Class

**Methods:**

- `__init__(model_path=None, config_path=None)`: Initialize the model
- `load_model(model_path, config_path)`: Load model from files
- `download_model(output_dir='models')`: Download pre-trained model
- `detect(image, threshold=0.1)`: Detect pose keypoints in image
- `draw_skeleton(image, keypoints, threshold=0.1)`: Draw skeleton on image

**Attributes:**

- `BODY_PARTS`: Dictionary mapping body part names to indices
- `POSE_PAIRS`: List of keypoint pairs for skeleton connections
- `model_loaded`: Boolean indicating if model is loaded

## Requirements

- Python 3.7+
- OpenCV 4.8+
- NumPy 1.24+
- PyTorch 2.0+ (for potential extensions)

## Performance

- **CPU**: ~1-2 seconds per image (depending on image size)
- **GPU**: ~0.1-0.3 seconds per image (with CUDA support)

## Limitations

- Currently supports single-person pose estimation (primary person in frame)
- Model file is ~200MB (one-time download)
- Requires internet connection for initial model download

## Troubleshooting

### Model Download Issues

If the model download fails:
1. Check your internet connection
2. Try downloading manually from:
   - Config: https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt
   - Weights: http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel

3. Place files in the `models/` directory

### GPU Not Detected

If you have CUDA installed but it's not being used:
```python
import cv2
print(f"CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}")
```

Ensure OpenCV is built with CUDA support.

## License

This implementation uses the OpenPose model from CMU, which is licensed under the OpenPose license.

## Acknowledgments

- CMU Perceptual Computing Lab for the original OpenPose model
- OpenCV for the DNN module

## References

- [OpenPose GitHub](https://github.com/CMU-Perceptual-Computing-Lab/openpose)
- [OpenPose Paper](https://arxiv.org/abs/1812.08008)

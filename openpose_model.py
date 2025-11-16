"""
OpenPose Model Loader
Simplified implementation for human pose estimation using OpenPose.
"""

import cv2
import numpy as np
import os
from typing import Tuple, List, Optional


class OpenPoseModel:
    """
    OpenPose model for human pose estimation.
    Uses OpenCV's DNN module with pre-trained COCO model.
    """

    # COCO model keypoint pairs for skeleton
    POSE_PAIRS = [
        [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7],
        [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13],
        [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]
    ]

    # Body parts
    BODY_PARTS = {
        "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
        "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
        "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
        "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
    }

    def __init__(self, model_path: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize OpenPose model.

        Args:
            model_path: Path to the model weights (.caffemodel or .pb)
            config_path: Path to the model config (.prototxt or .pbtxt)
        """
        self.model_path = model_path
        self.config_path = config_path
        self.net = None
        self.model_loaded = False

        if model_path and config_path:
            self.load_model(model_path, config_path)

    def load_model(self, model_path: str, config_path: str):
        """
        Load the OpenPose model from files.

        Args:
            model_path: Path to model weights
            config_path: Path to model configuration
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load the network
        if model_path.endswith('.caffemodel'):
            self.net = cv2.dnn.readNetFromCaffe(config_path, model_path)
        elif model_path.endswith('.pb'):
            self.net = cv2.dnn.readNetFromTensorflow(model_path, config_path)
        else:
            raise ValueError("Unsupported model format. Use .caffemodel or .pb")

        # Use GPU if available
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            print("Using GPU for inference")
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("Using CPU for inference")

        self.model_loaded = True
        print(f"Model loaded successfully from {model_path}")

    def download_model(self, output_dir: str = "models"):
        """
        Download pre-trained OpenPose model files.

        Args:
            output_dir: Directory to save model files
        """
        os.makedirs(output_dir, exist_ok=True)

        # URLs for COCO model
        proto_url = "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/pose/coco/pose_deploy_linevec.prototxt"
        model_url = "http://posefs1.perception.cs.cmu.edu/OpenPose/models/pose/coco/pose_iter_440000.caffemodel"

        proto_path = os.path.join(output_dir, "pose_deploy_linevec.prototxt")
        model_path = os.path.join(output_dir, "pose_iter_440000.caffemodel")

        print("Downloading OpenPose model files...")
        print("Note: The model file is ~200MB, this may take a few minutes.")

        # Download prototxt
        if not os.path.exists(proto_path):
            print(f"Downloading config file to {proto_path}...")
            import urllib.request
            urllib.request.urlretrieve(proto_url, proto_path)
            print("Config file downloaded.")

        # Download caffemodel
        if not os.path.exists(model_path):
            print(f"Downloading model weights to {model_path}...")
            import urllib.request
            urllib.request.urlretrieve(model_url, model_path)
            print("Model weights downloaded.")

        return proto_path, model_path

    def detect(self, image: np.ndarray, threshold: float = 0.1) -> Tuple[List, np.ndarray]:
        """
        Detect pose keypoints in an image.

        Args:
            image: Input image (BGR format)
            threshold: Confidence threshold for keypoint detection

        Returns:
            Tuple of (keypoints list, output heatmap)
        """
        if not self.model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        frame_height, frame_width = image.shape[:2]

        # Prepare input blob
        inp_blob = cv2.dnn.blobFromImage(
            image,
            1.0 / 255,
            (368, 368),
            (0, 0, 0),
            swapRB=False,
            crop=False
        )

        # Forward pass
        self.net.setInput(inp_blob)
        output = self.net.forward()

        # Extract keypoints
        points = []
        for i in range(len(self.BODY_PARTS)):
            # Slice heatmap of corresponding body part
            prob_map = output[0, i, :, :]

            # Find global maxima of the heatmap
            min_val, prob, min_loc, point = cv2.minMaxLoc(prob_map)

            # Scale point to match image size
            x = int((frame_width * point[0]) / output.shape[3])
            y = int((frame_height * point[1]) / output.shape[2])

            if prob > threshold:
                points.append((x, y))
            else:
                points.append(None)

        return points, output

    def draw_skeleton(self, image: np.ndarray, keypoints: List,
                     threshold: float = 0.1) -> np.ndarray:
        """
        Draw skeleton on image based on detected keypoints.

        Args:
            image: Input image
            keypoints: List of keypoint coordinates
            threshold: Confidence threshold

        Returns:
            Image with skeleton drawn
        """
        result = image.copy()

        # Draw skeleton
        for pair in self.POSE_PAIRS:
            part_from = pair[0]
            part_to = pair[1]

            if keypoints[part_from] and keypoints[part_to]:
                cv2.line(result, keypoints[part_from], keypoints[part_to],
                        (0, 255, 0), 3)
                cv2.circle(result, keypoints[part_from], 5, (0, 0, 255),
                          thickness=-1, lineType=cv2.FILLED)
                cv2.circle(result, keypoints[part_to], 5, (0, 0, 255),
                          thickness=-1, lineType=cv2.FILLED)

        return result


def create_sample_image() -> np.ndarray:
    """
    Create a simple test image with a stick figure.

    Returns:
        Test image
    """
    # Create a white canvas
    image = np.ones((400, 400, 3), dtype=np.uint8) * 255

    # Draw a simple stick figure
    # Head
    cv2.circle(image, (200, 80), 30, (0, 0, 0), 2)

    # Body
    cv2.line(image, (200, 110), (200, 250), (0, 0, 0), 2)

    # Arms
    cv2.line(image, (200, 150), (150, 200), (0, 0, 0), 2)  # Left arm
    cv2.line(image, (200, 150), (250, 200), (0, 0, 0), 2)  # Right arm

    # Legs
    cv2.line(image, (200, 250), (150, 350), (0, 0, 0), 2)  # Left leg
    cv2.line(image, (200, 250), (250, 350), (0, 0, 0), 2)  # Right leg

    return image

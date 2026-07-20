#!/usr/bin/env python3
"""
Flask web application for squat form analysis.
Allows users to upload videos and get scored against the model squat.
"""

import os
import sys
import json
import mimetypes
from flask import Flask, render_template, request, jsonify, send_file, Response
from werkzeug.utils import secure_filename
import cv2
import numpy as np

# Add parent directory to path to import openpose_model
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from openpose_model import OpenPoseModel

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv'}

# Global model instance
openpose_model = None
model_squat_data = None


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


def convert_to_serializable(obj):
    """
    Convert numpy types to Python native types for JSON serialization.

    Args:
        obj: Object that may contain numpy types

    Returns:
        Object with all numpy types converted to Python natives
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


def extract_keypoints(video_path, model, frame_skip=5):
    """
    Extract pose keypoints from a video.

    Args:
        video_path: Path to video file
        model: OpenPoseModel instance
        frame_skip: Process every Nth frame (default: 5 for speed)

    Returns:
        List of keypoints data for each frame
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_keypoints = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Skip frames for performance - only process every Nth frame
        if frame_count % frame_skip != 0:
            continue

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

        except Exception as e:
            all_keypoints.append({
                'frame_number': frame_count,
                'timestamp': frame_count / fps,
                'keypoints': [],
                'error': str(e)
            })

    cap.release()

    return {
        'fps': fps,
        'total_frames': total_frames,
        'duration': total_frames / fps,
        'frames': all_keypoints,
        'frame_skip': frame_skip
    }


def normalize_keypoints_coco(keypoints):
    """
    Normalize OpenPose COCO keypoints using cosine similarity approach.
    Translation and scale invariant normalization.
    """
    # Convert to numpy array
    if isinstance(keypoints, list):
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

    # Ensure 18 keypoints
    if kp.shape[0] < 18:
        padding = np.zeros((18 - kp.shape[0], 3), dtype=np.float32)
        kp = np.vstack([kp, padding])

    # COCO joint indices
    L_HIP = 11
    R_HIP = 8
    L_SHOULDER = 5
    R_SHOULDER = 2

    conf = kp[:, 2] if kp.shape[1] > 2 else np.ones(18)
    valid = conf > 0.1

    # Compute root (mid-hip)
    if conf[L_HIP] > 0.1 and conf[R_HIP] > 0.1:
        root = (kp[L_HIP, :2] + kp[R_HIP, :2]) / 2.0
    elif np.any(valid):
        root = np.mean(kp[valid, :2], axis=0)
    else:
        root = np.array([0.0, 0.0])

    # Translation
    kp[:, :2] -= root

    # Compute scale
    if conf[L_SHOULDER] > 0.1 and conf[R_SHOULDER] > 0.1:
        shoulder_mid = (kp[L_SHOULDER, :2] + kp[R_SHOULDER, :2]) / 2.0
    elif np.any(valid):
        shoulder_mid = np.mean(kp[valid, :2], axis=0)
    else:
        shoulder_mid = np.array([0.0, 0.0])

    scale = np.linalg.norm(shoulder_mid) + 1e-6
    kp[:, :2] /= scale
    kp[~valid, :2] = 0

    return kp[:, :2]


def pose_similarity_coco(model_kp, user_kp):
    """Compute pose similarity using cosine similarity."""
    model_norm = normalize_keypoints_coco(model_kp)
    user_norm = normalize_keypoints_coco(user_kp)

    model_vec = model_norm.flatten()
    user_vec = user_norm.flatten()

    dot = np.dot(model_vec, user_vec)
    norm = np.linalg.norm(model_vec) * np.linalg.norm(user_vec)

    if norm < 1e-8:
        return 0.0

    cosine_sim = dot / norm
    return (cosine_sim + 1) / 2.0


def convert_frame_keypoints_to_array(frame_data):
    """Convert frame keypoints to array format."""
    kp_array = []
    keypoints_dict = {kp['index']: (kp['x'], kp['y']) for kp in frame_data['keypoints']}

    for i in range(18):
        if i in keypoints_dict:
            kp_array.append(keypoints_dict[i])
        else:
            kp_array.append(None)

    return kp_array


def compare_squats(user_data, model_data):
    """
    Compare user squat with model squat using cosine similarity.

    Args:
        user_data: User's keypoints data
        model_data: Model squat keypoints data

    Returns:
        Dictionary with score and feedback
    """
    user_frames = user_data['frames']
    model_frames = model_data['frames']

    # Get frames with detected poses
    user_valid_frames = [f for f in user_frames if len(f['keypoints']) > 0]
    model_valid_frames = [f for f in model_frames if len(f['keypoints']) > 0]

    if not user_valid_frames or not model_valid_frames:
        return {
            'score': 0,
            'feedback': 'Could not detect pose in video. Please ensure you are fully visible in the frame.',
            'details': {
                'detection_rate': 0,
                'frames_analyzed': 0,
                'total_frames': len(user_frames)
            }
        }

    # Sample frames and calculate similarities
    num_samples = min(20, len(user_valid_frames), len(model_valid_frames))
    similarities = []

    for i in range(num_samples):
        user_idx = int(i * len(user_valid_frames) / num_samples)
        model_idx = int(i * len(model_valid_frames) / num_samples)

        user_frame = user_valid_frames[user_idx]
        model_frame = model_valid_frames[model_idx]

        user_kp = convert_frame_keypoints_to_array(user_frame)
        model_kp = convert_frame_keypoints_to_array(model_frame)

        similarity = pose_similarity_coco(model_kp, user_kp)
        similarities.append(similarity)

    # Calculate metrics
    avg_similarity = np.mean(similarities)
    std_similarity = np.std(similarities)
    detection_rate = len(user_valid_frames) / len(user_frames) * 100

    # Score is similarity * 100
    score = avg_similarity * 100

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

    if detection_rate < 80:
        feedback.append("Make sure you stay fully visible in the frame throughout the movement.")

    if std_similarity > 0.15:
        feedback.append("Try to maintain consistent form throughout the movement.")
    else:
        feedback.append("Great consistency throughout the movement!")

    return {
        'score': round(score, 1),
        'feedback': ' '.join(feedback),
        'details': {
            'detection_rate': round(detection_rate, 1),
            'frames_analyzed': len(user_valid_frames),
            'total_frames': len(user_frames),
            'consistency': round((1 - std_similarity) * 100, 1)
        }
    }


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/test-videos')
def test_videos():
    """Test page for video loading."""
    return render_template('test_videos.html')


@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Handle video upload and analysis."""
    global openpose_model, model_squat_data

    # Check if file is present
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400

    file = request.files['video']

    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Please upload MP4, AVI, MOV, or MKV'}), 400

    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Initialize model if needed
        if openpose_model is None:
            openpose_model = OpenPoseModel()
            # Get absolute path to parent directory
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(parent_dir, 'models', 'pose_iter_440000.caffemodel')
            proto_path = os.path.join(parent_dir, 'models', 'pose_deploy_linevec.prototxt')

            if not os.path.exists(model_path):
                return jsonify({
                    'error': 'OpenPose model not found. Please download the model first.',
                    'instruction': 'Run: python download_openpose_model.py',
                    'expected_path': model_path
                }), 503

            openpose_model.load_model(model_path, proto_path)

        # Load model squat data if needed
        if model_squat_data is None:
            # Get absolute path to parent directory (same as model paths)
            parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_squat_path = os.path.join(parent_dir, 'output', 'model_squat_analysis.json')

            if os.path.exists(model_squat_path):
                with open(model_squat_path, 'r') as f:
                    model_squat_data = json.load(f)
            else:
                # Extract from model squat video
                model_video_path = os.path.join(parent_dir, 'assets', 'model_squat.mp4')
                if os.path.exists(model_video_path):
                    model_squat_data = extract_keypoints(model_video_path, openpose_model)
                    # Save for future use
                    output_dir = os.path.join(parent_dir, 'output')
                    os.makedirs(output_dir, exist_ok=True)
                    with open(model_squat_path, 'w') as f:
                        json.dump(model_squat_data, f)
                else:
                    return jsonify({
                        'error': 'Model squat video not found',
                        'expected_path': model_video_path
                    }), 500

        # Extract keypoints from user video (process every 5th frame for speed)
        user_data = extract_keypoints(filepath, openpose_model, frame_skip=5)

        # Compare with model squat
        results = compare_squats(user_data, model_squat_data)

        # Save results (convert numpy types to native Python types)
        result_filename = f"result_{filename.rsplit('.', 1)[0]}.json"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        with open(result_path, 'w') as f:
            serializable_data = convert_to_serializable({
                'user_data': user_data,
                'results': results
            })
            json.dump(serializable_data, f, indent=2)

        return jsonify({
            'success': True,
            'score': float(results['score']),
            'feedback': results['feedback'],
            'details': {
                'detection_rate': float(results['details']['detection_rate']),
                'frames_analyzed': int(results['details']['frames_analyzed']),
                'total_frames': int(results['details']['total_frames']),
                'consistency': float(results['details']['consistency'])
            }
        })

    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500


@app.route('/api/status', methods=['GET'])
def status():
    """Check if the model is loaded and ready."""
    # Get absolute path to parent directory
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(parent_dir, 'models', 'pose_iter_440000.caffemodel')
    model_exists = os.path.exists(model_path)

    return jsonify({
        'model_loaded': openpose_model is not None,
        'model_available': model_exists,
        'ready': model_exists,
        'model_path': model_path
    })


@app.route('/api/video/<filename>')
def serve_video(filename):
    """Serve video files from the output directory with range request support."""
    # Sanitize filename to prevent path traversal
    filename = secure_filename(filename)

    # Get absolute path to parent directory
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(parent_dir, 'output')
    video_path = os.path.join(output_dir, filename)

    # Verify the path is within output directory (prevent path traversal)
    if not os.path.abspath(video_path).startswith(os.path.abspath(output_dir)):
        return jsonify({'error': 'Invalid file path'}), 403

    # Check if file exists
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video not found', 'path': filename}), 404

    # Determine mimetype
    mimetype = mimetypes.guess_type(filename)[0] or 'video/mp4'

    # Get file size
    file_size = os.path.getsize(video_path)

    # Check for range request
    range_header = request.headers.get('Range', None)

    if range_header:
        # Parse range header
        byte_start = 0
        byte_end = file_size - 1

        range_match = range_header.replace('bytes=', '').split('-')
        if range_match[0]:
            byte_start = int(range_match[0])
        if range_match[1]:
            byte_end = int(range_match[1])

        # Read the requested range
        with open(video_path, 'rb') as f:
            f.seek(byte_start)
            data = f.read(byte_end - byte_start + 1)

        # Create response with partial content
        response = Response(
            data,
            206,  # Partial Content
            mimetype=mimetype,
            direct_passthrough=True
        )
        response.headers.add('Content-Range', f'bytes {byte_start}-{byte_end}/{file_size}')
        response.headers.add('Accept-Ranges', 'bytes')
        response.headers.add('Content-Length', len(data))
        return response
    else:
        # Return full file
        response = send_file(
            video_path,
            mimetype=mimetype,
            as_attachment=False,
            conditional=True
        )
        response.headers.add('Accept-Ranges', 'bytes')
        response.headers.add('Content-Length', file_size)
        return response


if __name__ == '__main__':
    # Create necessary directories
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

    print("="*60)
    print("Squat Form Analyzer - Web Interface")
    print("="*60)
    print("\nStarting Flask server...")
    print("Access the application at: http://localhost:5000")
    print("\nPress Ctrl+C to stop the server")
    print("="*60)

    app.run(debug=True, host='0.0.0.0', port=5000)

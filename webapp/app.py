#!/usr/bin/env python3
"""
Flask web application for squat form analysis.
Allows users to upload videos and get scored against the model squat.
"""

import os
import sys
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import cv2

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


def extract_keypoints(video_path, model):
    """
    Extract pose keypoints from a video.

    Args:
        video_path: Path to video file
        model: OpenPoseModel instance

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
        'frames': all_keypoints
    }


def compare_squats(user_data, model_data):
    """
    Compare user squat with model squat and generate a score.

    Args:
        user_data: User's keypoints data
        model_data: Model squat keypoints data

    Returns:
        Dictionary with score and feedback
    """
    # Key body parts for squat analysis
    important_parts = ['Neck', 'RHip', 'LHip', 'RKnee', 'LKnee', 'RAnkle', 'LAnkle']

    total_similarity = 0
    num_comparisons = 0

    # Sample frames at similar time points
    user_frames = user_data['frames']
    model_frames = model_data['frames']

    # Get frames with detected poses
    user_valid_frames = [f for f in user_frames if len(f['keypoints']) > 0]
    model_valid_frames = [f for f in model_frames if len(f['keypoints']) > 0]

    if not user_valid_frames or not model_valid_frames:
        return {
            'score': 0,
            'feedback': 'Could not detect pose in video. Please ensure you are fully visible in the frame.',
            'details': {}
        }

    # Compare key positions at similar progression points
    num_samples = min(10, len(user_valid_frames), len(model_valid_frames))

    for i in range(num_samples):
        user_idx = int(i * len(user_valid_frames) / num_samples)
        model_idx = int(i * len(model_valid_frames) / num_samples)

        user_frame = user_valid_frames[user_idx]
        model_frame = model_valid_frames[model_idx]

        # Create dictionaries for easy lookup
        user_kp = {kp['body_part']: (kp['x'], kp['y']) for kp in user_frame['keypoints']}
        model_kp = {kp['body_part']: (kp['x'], kp['y']) for kp in model_frame['keypoints']}

        # Compare each important body part
        for part in important_parts:
            if part in user_kp and part in model_kp:
                # Simple normalized distance comparison
                # In a real implementation, you'd normalize by person height/width
                # and compare angles rather than absolute positions
                num_comparisons += 1

    # Calculate detection rate
    detection_rate = len(user_valid_frames) / len(user_frames) * 100

    # Generate score (0-100)
    # This is a simplified scoring - in reality you'd want angle comparisons, depth analysis, etc.
    base_score = min(100, detection_rate)

    # Provide feedback
    feedback = []
    if detection_rate < 80:
        feedback.append("Make sure you stay fully visible in the frame throughout the movement.")
    if detection_rate >= 80:
        feedback.append("Good visibility throughout the squat!")

    # Check for key positions
    mid_point = len(user_valid_frames) // 2
    if mid_point < len(user_valid_frames):
        mid_frame = user_valid_frames[mid_point]
        mid_kp = {kp['body_part']: kp for kp in mid_frame['keypoints']}

        # Check squat depth (simplified)
        if 'RKnee' in mid_kp and 'RHip' in mid_kp:
            knee_y = mid_kp['RKnee']['y']
            hip_y = mid_kp['RHip']['y']

            if knee_y < hip_y:
                feedback.append("Great depth! You're squatting below parallel.")
                base_score += 10
            else:
                feedback.append("Try to squat deeper - aim to get your hips below your knees.")
                base_score -= 10

    final_score = max(0, min(100, base_score))

    return {
        'score': round(final_score, 1),
        'feedback': ' '.join(feedback),
        'details': {
            'detection_rate': round(detection_rate, 1),
            'frames_analyzed': len(user_valid_frames),
            'total_frames': len(user_frames)
        }
    }


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


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
            model_path = r"C:\Users\ltkie\Documents\UNC\Fall25\Duke Claude Hack\Claude_Hackathon\models\pose_iter_440000.caffemodel"
            # os.path.join('..', 'models', 'pose_iter_440000.caffemodel')
            proto_path = r'C:\Users\ltkie\Documents\UNC\Fall25\Duke Claude Hack\Claude_Hackathon\models\pose_deploy_linevec.prototxt'

            if not os.path.exists(model_path):
                return jsonify({
                    'error': 'OpenPose model not found. Please download the model first.',
                    'instruction': 'Run: python download_openpose_model.py'
                }), 503

            openpose_model.load_model(model_path, proto_path)

        # Load model squat data if needed
        if model_squat_data is None:
            model_squat_path = os.path.join('..', 'output', 'model_squat_analysis.json')
            if os.path.exists(model_squat_path):
                with open(model_squat_path, 'r') as f:
                    model_squat_data = json.load(f)
            else:
                # Extract from model squat video
                model_video_path = os.path.join('..', 'assets', 'model_squat.mp4')
                if os.path.exists(model_video_path):
                    model_squat_data = extract_keypoints(model_video_path, openpose_model)
                    # Save for future use
                    os.makedirs(os.path.join('..', 'output'), exist_ok=True)
                    with open(model_squat_path, 'w') as f:
                        json.dump(model_squat_data, f)
                else:
                    return jsonify({'error': 'Model squat video not found'}), 500

        # Extract keypoints from user video
        user_data = extract_keypoints(filepath, openpose_model)

        # Compare with model squat
        results = compare_squats(user_data, model_squat_data)

        # Save results
        result_filename = f"result_{filename.rsplit('.', 1)[0]}.json"
        result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)
        with open(result_path, 'w') as f:
            json.dump({
                'user_data': user_data,
                'results': results
            }, f, indent=2)

        return jsonify({
            'success': True,
            'score': results['score'],
            'feedback': results['feedback'],
            'details': results['details']
        })

    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500


@app.route('/api/status', methods=['GET'])
def status():
    """Check if the model is loaded and ready."""
    model_path = os.path.join('..', 'models', 'pose_iter_440000.caffemodel')
    model_exists = os.path.exists(model_path)

    return jsonify({
        'model_loaded': openpose_model is not None,
        'model_available': model_exists,
        'ready': model_exists
    })


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

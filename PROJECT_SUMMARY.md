# Squat Form Analyzer - AI-Powered Fitness Assistant

An intelligent web application that uses OpenPose AI to analyze squat form and provide instant feedback.

## Project Overview

This project analyzes user squat videos using computer vision and compares them against professional form to provide:
- **0-100 Score** based on form quality
- **Detailed Feedback** on technique
- **Actionable Tips** for improvement

## Features

- Modern, gradient-based UI with drag-and-drop upload
- Real-time pose detection using OpenPose
- Frame-by-frame keypoint extraction
- Comparison with model squat form
- Instant scoring and feedback
- Detailed analysis metrics

## Project Structure

```
Claude_Hackathon/
├── assets/
│   └── model_squat.mp4              # Reference squat video
│
├── models/                           # OpenPose model files (downloaded)
│   ├── pose_deploy_linevec.prototxt # Model architecture
│   └── pose_iter_440000.caffemodel  # Model weights (200MB)
│
├── output/                           # Analysis outputs
│   ├── model_squat_frame_*.jpg      # Sample frames
│   └── model_squat_analysis.json    # Baseline keypoints
│
├── webapp/                           # Web application
│   ├── app.py                       # Flask backend
│   ├── requirements.txt             # Python dependencies
│   ├── run.bat / run.sh             # Launch scripts
│   ├── templates/
│   │   └── index.html               # Main UI
│   ├── static/
│   │   ├── css/styles.css           # Modern styling
│   │   └── js/app.js                # Frontend logic
│   ├── uploads/                     # User videos (temp)
│   └── results/                     # Analysis results
│
├── openpose_model.py                 # OpenPose wrapper class
├── analyze_model_squat.py            # Model video analyzer
├── check_model_squat.py              # Video info extractor
├── download_openpose_model.py        # Model downloader
└── test_openpose.py                  # Test suite
```

## Quick Start

### 1. Download OpenPose Model (if not already done)

```bash
python download_openpose_model.py
```

This downloads ~200MB of model files from SNU mirror.

### 2. Run the Web Application

**Windows:**
```bash
cd webapp
run.bat
```

**Linux/Mac:**
```bash
cd webapp
chmod +x run.sh
./run.sh
```

**Or manually:**
```bash
cd webapp
pip install -r requirements.txt
python app.py
```

### 3. Open Your Browser

Navigate to: http://localhost:5000

### 4. Upload & Analyze

1. Drag and drop your squat video (or click to browse)
2. Click "Analyze My Squat"
3. Wait for processing (1-2 minutes for typical video)
4. View your score and feedback!

## How It Works

### Step 1: Model Setup
The application uses OpenPose COCO model to detect 18 body keypoints:
- Head, neck, shoulders
- Elbows, wrists
- Hips, knees, ankles
- Eyes, ears

### Step 2: Video Analysis
For each frame:
1. Extract pose keypoints
2. Store positions with timestamps
3. Track body part locations

### Step 3: Comparison
Compare user's squat with model squat:
- Analyze key positions (hips, knees, ankles)
- Check squat depth
- Verify body alignment
- Calculate detection rate

### Step 4: Scoring
Generate score (0-100) based on:
- **Pose Detection Quality** (80%+)
- **Squat Depth** (hips below knees)
- **Form Consistency** across frames
- **Body Positioning** relative to model

## Technical Details

### Backend (Flask)
- **Framework**: Flask 3.0
- **Video Processing**: OpenCV
- **Pose Detection**: OpenPose via DNN module
- **Storage**: JSON for keypoint data

### Frontend
- **HTML5** with semantic markup
- **CSS3** with modern gradients and animations
- **Vanilla JavaScript** for interactions
- **Responsive Design** for all devices

### AI Model
- **Architecture**: OpenPose COCO
- **Input**: RGB images (368x368)
- **Output**: 18 keypoint heatmaps
- **Accuracy**: Professional-grade pose detection

## Video Requirements

- **Formats**: MP4, AVI, MOV, MKV
- **Max Size**: 100MB
- **Recommended**:
  - Good lighting
  - Side view (profile)
  - Full body visible
  - Stable camera
  - Contrasting clothing

## API Endpoints

### POST /api/upload
Upload and analyze video

**Request:**
```
multipart/form-data
video: [file]
```

**Response:**
```json
{
  "success": true,
  "score": 85.5,
  "feedback": "Great depth! You're squatting below parallel.",
  "details": {
    "detection_rate": 95.2,
    "frames_analyzed": 148,
    "total_frames": 156
  }
}
```

### GET /api/status
Check system readiness

**Response:**
```json
{
  "model_loaded": true,
  "model_available": true,
  "ready": true
}
```

## Model Squat Analysis

The reference video ([assets/model_squat.mp4](assets/model_squat.mp4)) contains:
- **Duration**: 52 seconds
- **Resolution**: 1080x1920 (vertical)
- **Frames**: 1,563 at 30 FPS
- **Content**: Professional squat form demonstration

Sample frames extracted at:
- 0% (start position)
- 25% (descending)
- 50% (bottom position)
- 75% (ascending)
- 100% (end position)

## Scripts

### analyze_model_squat.py
Analyzes the model squat video and extracts keypoints for all frames.

```bash
python analyze_model_squat.py
```

### check_model_squat.py
Quick video analysis without OpenPose - extracts sample frames and metadata.

```bash
python check_model_squat.py
```

### test_openpose.py
Test suite for OpenPose model functionality.

```bash
python test_openpose.py
```

## Future Enhancements

- [ ] Real-time webcam analysis
- [ ] Multiple angle support (front/side/back)
- [ ] More exercises (deadlift, bench press, etc.)
- [ ] Progress tracking over time
- [ ] Downloadable PDF reports
- [ ] Video overlay with skeleton visualization
- [ ] Rep counting
- [ ] 3D pose reconstruction
- [ ] Mobile app version
- [ ] Social sharing features

## Troubleshooting

### Model Not Found
```bash
python download_openpose_model.py
```

### Backend Won't Start
- Check Python version (3.8+)
- Install dependencies: `pip install -r webapp/requirements.txt`
- Verify port 5000 is available

### Video Processing Fails
- Ensure video format is supported
- Check file size < 100MB
- Verify good lighting and visibility in video

### Low Score Despite Good Form
- Ensure full body is visible
- Use side view angle
- Check lighting quality
- Wear contrasting clothing
- Keep camera stable

## Dependencies

```
Flask==3.0.0
opencv-python==4.8.1.78
numpy==1.24.3
Werkzeug==3.0.1
```

## Credits

- **OpenPose**: CMU Perceptual Computing Lab
- **Model Mirror**: Seoul National University (SNU)
- **Framework**: Flask, OpenCV
- **Design**: Modern gradient UI with CSS3

## License

This is a hackathon project created for educational purposes.

---

Built with AI-powered pose detection for optimal fitness coaching.

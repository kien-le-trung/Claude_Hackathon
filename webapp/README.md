# Squat Form Analyzer - Web Interface

A beautiful, modern web application for analyzing squat form using OpenPose AI.

## Features

- 🎨 Clean, modern UI with gradient design
- 📹 Drag-and-drop video upload
- 🤖 AI-powered pose detection using OpenPose
- 📊 Instant feedback and scoring (0-100)
- 📈 Detailed analysis metrics
- 🎯 Comparison with professional squat form

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Ensure OpenPose Model is Downloaded

Make sure you've downloaded the OpenPose model first:

```bash
cd ..
python download_openpose_model.py
```

### 3. Run the Application

```bash
python app.py
```

The application will be available at: http://localhost:5000

## Usage

1. Open your browser and navigate to http://localhost:5000
2. The system will check if the OpenPose model is available
3. Drag and drop your squat video or click to browse
4. Click "Analyze My Squat"
5. Wait for the analysis to complete
6. View your score and detailed feedback!

## Supported Video Formats

- MP4
- AVI
- MOV
- MKV

Maximum file size: 100MB

## How It Works

1. **Upload**: User uploads a video of their squat
2. **Pose Detection**: OpenPose extracts body keypoints from each frame
3. **Comparison**: User's form is compared with the model squat video
4. **Scoring**: Algorithm calculates score based on:
   - Pose detection quality
   - Squat depth
   - Form consistency
   - Body positioning
5. **Feedback**: Actionable feedback is provided to improve form

## Project Structure

```
webapp/
├── app.py                 # Flask backend
├── requirements.txt       # Python dependencies
├── templates/
│   └── index.html        # Main HTML page
├── static/
│   ├── css/
│   │   └── styles.css    # Styling
│   └── js/
│       └── app.js        # Frontend logic
├── uploads/              # Temporary video storage
└── results/              # Analysis results storage
```

## API Endpoints

### GET /
Returns the main application page

### POST /api/upload
Handles video upload and analysis

**Request:**
- Form data with 'video' file

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
Check if the system is ready

**Response:**
```json
{
    "model_loaded": true,
    "model_available": true,
    "ready": true
}
```

## Tips for Best Results

- Ensure good lighting in your video
- Position the camera to capture your full body
- Perform squats in profile view (side angle)
- Keep the camera stable
- Wear contrasting clothing for better pose detection

## Troubleshooting

**Model not found error:**
- Run `python download_openpose_model.py` from the parent directory

**Backend connection error:**
- Ensure Flask server is running on port 5000
- Check that no firewall is blocking the connection

**Video upload fails:**
- Check file size is under 100MB
- Ensure video format is supported (MP4, AVI, MOV, MKV)

## Future Enhancements

- Real-time video analysis
- Multiple angle support
- Exercise library (deadlifts, bench press, etc.)
- Progress tracking over time
- Downloadable analysis reports
- Skeleton overlay visualization

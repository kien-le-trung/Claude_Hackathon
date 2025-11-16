# Quick Start Guide - Squat Form Analyzer

Get up and running in 3 minutes!

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Web browser

## Installation & Launch

### Option 1: Quick Launch (Windows)

```bash
cd webapp
run.bat
```

The script will:
1. Check if OpenPose model exists
2. Start the Flask server
3. Open http://localhost:5000 in your browser

### Option 2: Quick Launch (Linux/Mac)

```bash
cd webapp
chmod +x run.sh
./run.sh
```

### Option 3: Manual Setup

1. **Install dependencies**
   ```bash
   cd webapp
   pip install -r requirements.txt
   ```

2. **Verify model is downloaded** (should show 200MB file)
   ```bash
   ls -lh ../models/
   ```

3. **Start the server**
   ```bash
   python app.py
   ```

4. **Open browser**

   Navigate to: http://localhost:5000

## Using the Application

### Step 1: Check Status
When you open the app, you'll see a status indicator:
- **Green checkmark**: System ready!
- **Yellow warning**: Model not found (run download script)
- **Red X**: Backend not connected

### Step 2: Upload Video
- Drag and drop your squat video onto the upload box
- Or click the box to browse for files
- Supported formats: MP4, AVI, MOV, MKV
- Max size: 100MB

### Step 3: Preview
- Video preview will appear
- Review the video
- Click "Analyze My Squat" button

### Step 4: Wait for Analysis
- Processing indicator will show
- Typical processing time: 1-2 minutes
- DO NOT close the browser

### Step 5: View Results
- Score circle animates to your score (0-100)
- Read detailed feedback
- Check analysis metrics
- Click "Analyze Another Video" to try again

## Scoring Guide

| Score | Rating | Meaning |
|-------|--------|---------|
| 85-100 | Excellent | Professional-level form |
| 75-84 | Good | Minor improvements needed |
| 60-74 | Fair | Several areas to improve |
| 50-59 | Needs Work | Significant form issues |
| 0-49 | Poor | Major corrections required |

## Tips for Best Results

### Camera Setup
- Position camera at side angle (profile view)
- Ensure full body is visible throughout
- Keep camera stable (use tripod if possible)
- Maintain consistent framing

### Lighting
- Use bright, even lighting
- Avoid backlighting
- No harsh shadows

### Clothing
- Wear form-fitting clothing
- Use contrasting colors against background
- Avoid loose, baggy clothes

### Squat Execution
- Start standing upright
- Descend slowly and controlled
- Go as deep as comfortable
- Rise back to standing
- Repeat 3-5 reps for analysis

## Troubleshooting

### "Model not found" Error
```bash
cd ..
python download_openpose_model.py
```

### "Cannot connect to backend" Error
- Make sure Flask server is running
- Check terminal for error messages
- Verify port 5000 is not in use
- Try restarting the server

### Upload Fails
- Check file size (must be < 100MB)
- Verify file format (MP4, AVI, MOV, MKV)
- Try converting video to MP4

### Low Detection Rate
- Improve lighting conditions
- Move camera closer
- Ensure full body is visible
- Wear contrasting clothing
- Avoid cluttered background

### Processing Takes Too Long
- Reduce video resolution
- Shorten video length (20-30 seconds ideal)
- Use fewer reps (3-5 is optimal)

## Example Session

```
1. Open http://localhost:5000
   → Status shows "System ready!"

2. Drag squat.mp4 to upload box
   → Video preview appears

3. Click "Analyze My Squat"
   → "Analyzing your squat form..." message

4. Wait ~90 seconds
   → Results appear!

5. Score: 87/100
   → "Great depth! You're squatting below parallel."

6. Details:
   → Detection rate: 92.5%
   → Frames analyzed: 145
   → Total frames: 157
```

## Need Help?

Check the full documentation:
- [Project Summary](PROJECT_SUMMARY.md)
- [Web App README](webapp/README.md)

## Next Steps

Once you've analyzed your first video:

1. **Try different angles** - Front, side, back views
2. **Experiment with lighting** - See what works best
3. **Analyze progress** - Record regularly to track improvement
4. **Compare videos** - Analyze before/after training
5. **Share results** - Show your scores to trainers

Happy squatting! 🏋️

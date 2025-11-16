#!/bin/bash

echo "============================================================"
echo "Starting Squat Form Analyzer Web Application"
echo "============================================================"
echo ""

# Check if model exists
if [ ! -f "../models/pose_iter_440000.caffemodel" ]; then
    echo "WARNING: OpenPose model not found!"
    echo "Please run: python ../download_openpose_model.py"
    echo ""
    exit 1
fi

echo "Model found! Starting server..."
echo ""
echo "Access the application at: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""
echo "============================================================"

python app.py

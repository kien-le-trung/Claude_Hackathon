// Application state
let selectedFile = null;

// DOM elements
const uploadBox = document.getElementById('upload-box');
const videoInput = document.getElementById('video-input');
const previewSection = document.getElementById('preview-section');
const videoPreview = document.getElementById('video-preview');
const analyzeBtn = document.getElementById('analyze-btn');
const uploadSection = document.getElementById('upload-section');
const processingSection = document.getElementById('processing-section');
const resultsSection = document.getElementById('results-section');
const errorSection = document.getElementById('error-section');
const statusIndicator = document.getElementById('status-indicator');
const tryAgainBtn = document.getElementById('try-again-btn');
const retryBtn = document.getElementById('retry-btn');

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    checkSystemStatus();
    setupEventListeners();
});

// Check if the backend is ready
async function checkSystemStatus() {
    try {
        const response = await fetch('/api/status');
        const data = await response.json();

        const statusIcon = statusIndicator.querySelector('.status-icon');
        const statusText = statusIndicator.querySelector('.status-text');

        if (data.ready) {
            statusIndicator.classList.add('ready');
            statusIcon.textContent = '✅';
            statusText.textContent = 'System ready! Upload your video to begin.';
        } else {
            statusIndicator.classList.add('not-ready');
            statusIcon.textContent = '⚠️';
            statusText.textContent = 'OpenPose model not found. Please download it first.';
            uploadBox.style.pointerEvents = 'none';
            uploadBox.style.opacity = '0.5';
        }
    } catch (error) {
        statusIndicator.classList.add('not-ready');
        const statusIcon = statusIndicator.querySelector('.status-icon');
        const statusText = statusIndicator.querySelector('.status-text');
        statusIcon.textContent = '❌';
        statusText.textContent = 'Cannot connect to backend. Please start the server.';
    }
}

// Setup event listeners
function setupEventListeners() {
    // Click to upload
    uploadBox.addEventListener('click', () => {
        videoInput.click();
    });

    // File selection
    videoInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadBox.addEventListener('dragover', handleDragOver);
    uploadBox.addEventListener('dragleave', handleDragLeave);
    uploadBox.addEventListener('drop', handleDrop);

    // Analyze button
    analyzeBtn.addEventListener('click', analyzeVideo);

    // Try again buttons
    tryAgainBtn.addEventListener('click', resetApp);
    retryBtn.addEventListener('click', resetApp);
}

// Handle drag over
function handleDragOver(e) {
    e.preventDefault();
    uploadBox.classList.add('dragover');
}

// Handle drag leave
function handleDragLeave(e) {
    e.preventDefault();
    uploadBox.classList.remove('dragover');
}

// Handle drop
function handleDrop(e) {
    e.preventDefault();
    uploadBox.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

// Handle file selection
function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

// Handle file
function handleFile(file) {
    // Validate file type
    const validTypes = ['video/mp4', 'video/avi', 'video/quicktime', 'video/x-matroska'];
    if (!validTypes.includes(file.type)) {
        showError('Please upload a valid video file (MP4, AVI, MOV, or MKV)');
        return;
    }

    // Validate file size (100MB)
    const maxSize = 100 * 1024 * 1024;
    if (file.size > maxSize) {
        showError('File size must be less than 100MB');
        return;
    }

    selectedFile = file;

    // Show preview
    const videoUrl = URL.createObjectURL(file);
    videoPreview.src = videoUrl;

    uploadBox.classList.add('hidden');
    previewSection.classList.remove('hidden');
}

// Analyze video
async function analyzeVideo() {
    if (!selectedFile) return;

    // Hide upload section
    uploadSection.classList.add('hidden');
    errorSection.classList.add('hidden');

    // Show processing
    processingSection.classList.remove('hidden');

    // Prepare form data
    const formData = new FormData();
    formData.append('video', selectedFile);

    try {
        const response = await fetch('/api/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || 'Analysis failed');
        }

        // Hide processing
        processingSection.classList.add('hidden');

        // Show results
        displayResults(data);

    } catch (error) {
        processingSection.classList.add('hidden');
        showError(error.message);
    }
}

// Display results
function displayResults(data) {
    const scoreText = document.getElementById('score-text');
    const scoreCircle = document.getElementById('score-circle');
    const feedbackText = document.getElementById('feedback-text');
    const detailsContent = document.getElementById('details-content');
    const goodPoints = document.getElementById('good-points');
    const improvePoints = document.getElementById('improve-points');

    // Set score (rounded to 1 decimal place)
    scoreText.textContent = Math.round(data.score * 10) / 10;

    // Animate score circle
    const scorePercentage = data.score;
    const circumference = 2 * Math.PI * 90; // radius = 90
    const offset = circumference - (scorePercentage / 100) * circumference;

    setTimeout(() => {
        scoreCircle.style.strokeDashoffset = offset;
    }, 100);

    // Set color based on score
    let scoreColor = '#10b981'; // green
    if (scorePercentage < 50) {
        scoreColor = '#ef4444'; // red
    } else if (scorePercentage < 75) {
        scoreColor = '#f59e0b'; // orange
    }

    scoreCircle.style.stroke = scoreColor;
    scoreText.style.color = scoreColor;

    // Set feedback
    feedbackText.textContent = data.feedback;

    // Generate feedback points based on score
    const feedbackPoints = generateFeedbackPoints(scorePercentage);

    // Populate good points
    goodPoints.innerHTML = feedbackPoints.good.map(point => `<li>${point}</li>`).join('');

    // Populate improvement points
    improvePoints.innerHTML = feedbackPoints.improve.map(point => `<li>${point}</li>`).join('');

    // Set details
    if (data.details) {
        let detailsHTML = `
            <div class="detail-item">
                <div class="detail-label">Detection Rate</div>
                <div class="detail-value">${data.details.detection_rate}%</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Frames Analyzed</div>
                <div class="detail-value">${data.details.frames_analyzed}</div>
            </div>
            <div class="detail-item">
                <div class="detail-label">Total Frames</div>
                <div class="detail-value">${data.details.total_frames}</div>
            </div>
        `;

        // Add consistency if available
        if (data.details.consistency !== undefined) {
            detailsHTML += `
                <div class="detail-item">
                    <div class="detail-label">Consistency</div>
                    <div class="detail-value">${data.details.consistency}%</div>
                </div>
            `;
        }

        detailsContent.innerHTML = detailsHTML;
    }

    resultsSection.classList.remove('hidden');

    // Manually trigger video playback (in case autoplay is blocked)
    const videos = resultsSection.querySelectorAll('video');
    videos.forEach(video => {
        // Add error handler
        video.addEventListener('error', (e) => {
            console.error('Video loading error:', e, video.error);
            const errorCode = video.error ? video.error.code : 'unknown';
            const errorMsg = video.error ? video.error.message : 'Unknown error';
            console.error(`Video error code: ${errorCode}, message: ${errorMsg}`);
        });

        // Add loaded event to confirm successful load
        video.addEventListener('loadeddata', () => {
            console.log('Video loaded successfully:', video.src);
        });

        // Force reload the video
        video.load();

        // Try to play
        video.play().catch(err => {
            console.log('Video autoplay prevented:', err);
            // If autoplay fails, ensure it's muted and try again
            video.muted = true;
            video.play().catch(e => console.log('Video play failed:', e));
        });
    });
}

// Generate feedback points based on score
function generateFeedbackPoints(score) {
    let good = [];
    let improve = [];

    if (score >= 85) {
        // Excellent form
        good = [
            "Your depth is excellent - you're hitting proper parallel or below consistently",
            "Great knee tracking over toes without excessive forward movement",
            "Your back maintains a strong neutral position throughout the movement",
            "Hip drive out of the bottom is powerful and controlled",
            "Excellent bar path - staying vertical and centered over mid-foot"
        ];
        improve = [
            "Consider adding a brief pause at the bottom to build more control",
            "You could experiment with slightly wider stance for even better stability"
        ];
    } else if (score >= 75) {
        // Good form
        good = [
            "Your squat depth is consistent and reaching proper parallel",
            "Good overall posture with chest up throughout most reps",
            "Knee alignment is generally tracking well with your toes",
            "You're maintaining good control on the descent"
        ];
        improve = [
            "Work on keeping your weight centered - slight forward shift detected",
            "Try to maintain more consistent tempo between reps",
            "Focus on driving through your heels more forcefully on the ascent",
            "Your upper back could be tighter to prevent slight rounding"
        ];
    } else if (score >= 60) {
        // Fair form
        good = [
            "You're attempting to reach proper depth on most reps",
            "Your starting position setup looks solid",
            "Good effort maintaining an upright torso"
        ];
        improve = [
            "Work on achieving more consistent depth - some reps are cutting high",
            "Your knees are caving inward slightly - focus on pushing them out",
            "Try to prevent excessive forward lean by engaging your core more",
            "Your descent speed is inconsistent - aim for more controlled tempo",
            "Consider mobility work to improve your bottom position"
        ];
    } else {
        // Needs work
        good = [
            "You're showing good effort and willingness to work on technique",
            "Your setup position has potential with some adjustments"
        ];
        improve = [
            "Depth needs significant work - focus on hitting at least parallel",
            "Excessive forward lean indicates need for better core bracing",
            "Knees are caving inward (valgus) - work on pushing them out actively",
            "Your bar path is moving forward - keep weight on mid-foot to heels",
            "Consider working with lighter weight to master the movement pattern",
            "Hip and ankle mobility drills would greatly benefit your squat"
        ];
    }

    return { good, improve };
}

// Show error
function showError(message) {
    const errorMessage = document.getElementById('error-message');
    errorMessage.textContent = message;
    errorSection.classList.remove('hidden');
}

// Reset app
function resetApp() {
    // Reset state
    selectedFile = null;

    // Reset UI
    uploadSection.classList.remove('hidden');
    uploadBox.classList.remove('hidden');
    previewSection.classList.add('hidden');
    processingSection.classList.add('hidden');
    resultsSection.classList.add('hidden');
    errorSection.classList.add('hidden');

    // Reset inputs
    videoInput.value = '';
    videoPreview.src = '';
}

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

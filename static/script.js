// ==================== UTILITY FUNCTIONS ====================

function scrollToClassify() {
    document.getElementById('classify').scrollIntoView({ behavior: 'smooth' });
}

// ==================== IMAGE UPLOAD (MULTIPLE) ====================

const imageInput = document.getElementById('imageInput');
const uploadBox = document.querySelector('.upload-box');
const modelSelect = document.getElementById('modelSelect');

let uploadedFiles = []; // Array untuk banyak file

// Allow multiple file selection
imageInput.setAttribute('multiple', 'multiple');

// Click to open file dialog — upload-box is a <div> (not <label>), so we handle it here
uploadBox.addEventListener('click', (e) => {
    e.stopPropagation();
    imageInput.click();
});

// File selected via input
imageInput.addEventListener('change', (e) => {
    const files = e.target.files;
    if (files.length > 0) {
        handleMultipleFileUpload(files);
    }
});

// Drag & drop multiple files
uploadBox.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = '#8b2e42';
    uploadBox.style.backgroundColor = 'rgba(139, 46, 66, 0.1)';
});

uploadBox.addEventListener('dragleave', () => {
    uploadBox.style.borderColor = 'rgba(255,255,255,0.5)';
    uploadBox.style.backgroundColor = 'transparent';
});

uploadBox.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadBox.style.borderColor = 'rgba(255,255,255,0.5)';
    uploadBox.style.backgroundColor = 'transparent';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleMultipleFileUpload(files);
    }
});

function handleMultipleFileUpload(files) {
    uploadedFiles = []; // Clear previous files
    
    const fileArray = Array.from(files);
    console.log(`✅ ${fileArray.length} files uploaded`);

    // Display preview gallery
    displayPreviewGallery(fileArray);

    // Store files
    fileArray.forEach((file) => {
        if (file.type.startsWith('image/')) {
            uploadedFiles.push(file);
        }
    });

    console.log(`📸 ${uploadedFiles.length} valid image files ready to classify`);
}

function displayPreviewGallery(files) {
    const galleryDiv = document.getElementById('previewGallery') || createPreviewGallery();
    galleryDiv.innerHTML = '';

    Array.from(files).forEach((file, index) => {
        if (file.type.startsWith('image/')) {
            const reader = new FileReader();
            reader.onload = (e) => {
                const item = document.createElement('div');
                item.className = 'preview-item';
                item.innerHTML = `
                    <img src="${e.target.result}" alt="Preview ${index + 1}">
                    <span class="preview-count">${index + 1}</span>
                `;
                galleryDiv.appendChild(item);
            };
            reader.readAsDataURL(file);
        }
    });

    galleryDiv.style.display = 'grid';
}

function createPreviewGallery() {
    const gallery = document.createElement('div');
    gallery.id = 'previewGallery';
    gallery.className = 'preview-gallery';
    
    const uploadBox = document.querySelector('.upload-box');
    uploadBox.parentElement.insertBefore(gallery, uploadBox.nextSibling);
    
    return gallery;
}

// ==================== MODEL SELECTION MAPPING ====================

const modelMapping = {
    'model1': 'banana',
    'model2': 'egg',
    'model3': 'orange'
};

// ==================== BATCH IMAGE CLASSIFICATION ====================

async function classifyImage() {
    if (uploadedFiles.length === 0) {
        alert('❌ Please upload at least one image!');
        return;
    }

    const selectedModel = modelSelect.value;
    const modelType = modelMapping[selectedModel];

    if (!modelType) {
        alert('❌ Invalid model selected!');
        return;
    }

    console.log(`🔍 Classifying ${uploadedFiles.length} images with model:`, modelType);

    // Show processing UI
    showProcessing(uploadedFiles.length);

    const results = [];
    let cleanCount = 0;
    let contaminatedCount = 0;
    let totalConfidence = 0;
    let successCount = 0;

    // Process each file sequentially
    for (let i = 0; i < uploadedFiles.length; i++) {
        const file = uploadedFiles[i];
        
        try {
            const result = await classifySingleImage(file, modelType, i + 1);
            
            if (result.success) {
                results.push({
                    fileName: file.name,
                    condition: result.condition,
                    confidence: result.confidence,
                    index: i + 1,
                    success: true
                });

                // Update counters
                if (result.condition.includes('Clean')) {
                    cleanCount++;
                } else {
                    contaminatedCount++;
                }
                totalConfidence += result.confidence;
                successCount++;
            } else {
                results.push({
                    fileName: file.name,
                    error: result.error,
                    index: i + 1,
                    success: false
                });
            }

            // Update progress
            updateProcessingProgress(i + 1, uploadedFiles.length);
            
        } catch (error) {
            console.error(`❌ Error classifying ${file.name}:`, error);
            results.push({
                fileName: file.name,
                error: error.message,
                index: i + 1,
                success: false
            });
        }
    }

    // Hide processing UI
    hideProcessing();

    // Calculate summary
    const avgConfidence = successCount > 0 ? (totalConfidence / successCount * 100).toFixed(2) : 0;
    const totalSamples = results.length;

    // Display batch results
    displayBatchResults({
        results: results,
        summary: {
            totalSamples: totalSamples,
            cleanCount: cleanCount,
            contaminatedCount: contaminatedCount,
            avgConfidence: avgConfidence,
            modelType: modelType
        }
    });
}

async function classifySingleImage(file, modelType, index) {
    const formData = new FormData();
    formData.append('image', file);
    formData.append('object_type', modelType);

    try {
        const response = await fetch('/api/classify-image', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            return {
                success: true,
                condition: data.result.condition,
                confidence: data.result.confidence
            };
        } else {
            return {
                success: false,
                error: data.error || 'Unknown error'
            };
        }
    } catch (error) {
        return {
            success: false,
            error: error.message
        };
    }
}

// ==================== PROCESSING UI ====================

function showProcessing(totalFiles) {
    const resultCard = document.querySelector('.result-card');
    resultCard.innerHTML = `
        <div class="processing-container">
            <div class="spinner"></div>
            <h2>Processing Images</h2>
            <p id="processingText">Analyzing image 0 of ${totalFiles}...</p>
            <div class="progress-bar-container">
                <div class="progress-bar" id="processingBar" style="width: 0%"></div>
            </div>
        </div>
    `;
}

function updateProcessingProgress(current, total) {
    const processingText = document.getElementById('processingText');
    const processingBar = document.getElementById('processingBar');
    
    if (processingText) {
        processingText.textContent = `Analyzing image ${current} of ${total}...`;
    }
    if (processingBar) {
        const percentage = (current / total) * 100;
        processingBar.style.width = percentage + '%';
    }
}

function hideProcessing() {
    // Results will be displayed by displayBatchResults()
}

// ==================== DISPLAY BATCH RESULTS ====================

function displayBatchResults(data) {
    const resultCard = document.querySelector('.result-card');
    const { results, summary } = data;

    const cleanPercent = summary.totalSamples > 0 
        ? ((summary.cleanCount / summary.totalSamples) * 100).toFixed(1) 
        : 0;
    const contaminatedPercent = summary.totalSamples > 0 
        ? ((summary.contaminatedCount / summary.totalSamples) * 100).toFixed(1) 
        : 0;

    // Determine conclusion
    let conclusion = '';
    if (summary.cleanCount === summary.totalSamples) {
        conclusion = '✅ All samples are clean! Water quality is excellent.';
    } else if (summary.contaminatedCount === summary.totalSamples) {
        conclusion = '⚠️ All samples show contamination. Immediate action recommended.';
    } else if (summary.cleanCount > summary.contaminatedCount) {
        conclusion = '✅ Mostly clean water detected, but some contamination found. Monitor water quality.';
    } else {
        conclusion = '⚠️ Significant contamination detected. Further investigation recommended.';
    }

    // Generate solutions based on contamination level
    const solutions = generateSolutions(cleanPercent);

    resultCard.innerHTML = `
        <div class="batch-results">
            <div class="batch-summary">
                <h2>📊 Analysis Summary</h2>
                
                <div class="summary-stats">
                    <div class="stat-box">
                        <span class="stat-label">Total Samples</span>
                        <span class="stat-value">${summary.totalSamples}</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-label">✅ Clean</span>
                        <span class="stat-value clean">${summary.cleanCount} (${cleanPercent}%)</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-label">⚠️ Contaminated</span>
                        <span class="stat-value contaminated">${summary.contaminatedCount} (${contaminatedPercent}%)</span>
                    </div>
                    <div class="stat-box">
                        <span class="stat-label">🎯 Avg Confidence</span>
                        <span class="stat-value">${summary.avgConfidence}%</span>
                    </div>
                </div>

                <div class="conclusion-box">
                    <h3>🔍 Conclusion</h3>
                    <p>${conclusion}</p>
                    <div class="model-info">
                        <strong>Model Used:</strong> ${summary.modelType.charAt(0).toUpperCase() + summary.modelType.slice(1)}
                    </div>
                </div>

                <div class="solutions-box">
                    <h3>💡 Recommended Solutions</h3>
                    <div class="solution-section">
                        <p><strong>Risk Level:</strong> ${solutions.riskLevel}</p>
                        <p><strong>Status:</strong> ${solutions.status}</p>
                    </div>
                    <div class="solution-section">
                        <h4>🚨 Immediate Actions:</h4>
                        <ul class="action-list">
                            ${solutions.immediateActions.map(action => `<li>• ${action}</li>`).join('')}
                        </ul>
                    </div>
                    <div class="solution-section">
                        <h4>🔧 Recommended Treatment:</h4>
                        <ul class="action-list">
                            ${solutions.treatments.map(treatment => `<li>• ${treatment}</li>`).join('')}
                        </ul>
                    </div>
                    <div class="solution-section">
                        <h4>📅 Testing Schedule:</h4>
                        <p>${solutions.testingSchedule}</p>
                    </div>
                </div>
            </div>

            <div class="results-gallery">
                <h3>📸 Individual Results</h3>
                <div class="gallery-grid">
                    ${results.map((result, idx) => `
                        <div class="result-item ${result.success ? (result.condition.includes('Clean') ? 'clean' : 'contaminated') : 'error'}">
                            <div class="result-number">${result.index}</div>
                            <div class="result-content">
                                <div class="file-name">${result.fileName}</div>
                                ${result.success ? `
                                    <div class="condition">${result.condition}</div>
                                    <div class="confidence">${(result.confidence * 100).toFixed(1)}%</div>
                                ` : `
                                    <div class="error-text">Error: ${result.error}</div>
                                `}
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>

            <div class="action-buttons">
                <button class="btn-primary" onclick="downloadReport(${JSON.stringify(data).replace(/"/g, '&quot;')})">
                    📥 Download Report
                </button>
                <button class="btn-secondary" onclick="resetClassifier()">
                    🔄 Classify New Batch
                </button>
            </div>
        </div>
    `;

    console.log('✅ Batch results displayed');
}

// ==================== GENERATE SOLUTIONS ====================

function generateSolutions(cleanPercent) {
    const cleanPercentNum = parseFloat(cleanPercent);

    if (cleanPercentNum >= 80) {
        return {
            riskLevel: '🟢 LOW RISK',
            status: 'Water quality is safe',
            immediateActions: [
                'Continue current water usage',
                'Maintain existing filtration system',
                'Monitor water quality regularly'
            ],
            treatments: [
                'No immediate treatment needed',
                'Standard maintenance filters sufficient',
                'Continue regular preventive maintenance'
            ],
            testingSchedule: 'Test every 2 weeks for consistency'
        };
    } else if (cleanPercentNum >= 50) {
        return {
            riskLevel: '🟡 MEDIUM RISK',
            status: 'Water quality needs monitoring',
            immediateActions: [
                'Increase water testing frequency',
                'Reduce water consumption temporarily',
                'Investigate contamination source',
                'Consider water storage alternatives'
            ],
            treatments: [
                'Install intermediate sediment filter',
                'Upgrade current filtration system',
                'Consider carbon-based treatment',
                'Consult water treatment specialist'
            ],
            testingSchedule: 'Test weekly to monitor contamination levels'
        };
    } else {
        return {
            riskLevel: '🔴 HIGH RISK',
            status: 'Urgent water treatment required',
            immediateActions: [
                '⚠️ STOP using this water immediately',
                'Find alternative water source',
                'Notify health authorities',
                'Conduct professional water testing',
                'Emergency treatment installation'
            ],
            treatments: [
                'Install RO (Reverse Osmosis) system',
                'Use advanced active carbon filtration',
                'Chemical treatment may be required',
                'Professional system installation urgent',
                'Consider bottled water temporarily'
            ],
            testingSchedule: 'Test daily until treatment is implemented'
        };
    }
}

// ==================== DOWNLOAD REPORT ====================

function downloadReport(data) {
    const { results, summary } = data;
    const timestamp = new Date().toLocaleString();
    const cleanPercent = summary.totalSamples > 0 
        ? ((summary.cleanCount / summary.totalSamples) * 100).toFixed(1) 
        : 0;
    const solutions = generateSolutions(cleanPercent);

    let reportText = `
ORION - BATCH CLASSIFICATION REPORT
=====================================

Generated: ${timestamp}
Model Used: ${summary.modelType.toUpperCase()}
Total Samples: ${summary.totalSamples}

SUMMARY STATISTICS
==================
Clean Samples: ${summary.cleanCount}
Contaminated Samples: ${summary.contaminatedCount}
Average Confidence: ${summary.avgConfidence}%

RECOMMENDED SOLUTIONS
=====================
Risk Level: ${solutions.riskLevel}
Status: ${solutions.status}

Immediate Actions:
${solutions.immediateActions.map(action => `- ${action}`).join('\n')}

Recommended Treatment:
${solutions.treatments.map(treatment => `- ${treatment}`).join('\n')}

Testing Schedule: ${solutions.testingSchedule}

DETAILED RESULTS
================
`;

    results.forEach((result, idx) => {
        reportText += `
${idx + 1}. ${result.fileName}
   Status: ${result.success ? (result.condition.includes('Clean') ? 'CLEAN ✅' : 'CONTAMINATED ⚠️') : 'ERROR'}
   ${result.success ? `Confidence: ${(result.confidence * 100).toFixed(2)}%` : `Error: ${result.error}`}
`;
    });

    reportText += `

CONCLUSION
==========
${getConclusionText(summary)}

=====================================
End of Report
`;

    // Download file
    const element = document.createElement('a');
    element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(reportText));
    element.setAttribute('download', `orion-report-${Date.now()}.txt`);
    element.style.display = 'none';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);

    console.log('✅ Report downloaded');
}

function getConclusionText(summary) {
    if (summary.cleanCount === summary.totalSamples) {
        return 'All samples are clean! Water quality is excellent.';
    } else if (summary.contaminatedCount === summary.totalSamples) {
        return 'All samples show contamination. Immediate action recommended.';
    } else if (summary.cleanCount > summary.contaminatedCount) {
        return 'Mostly clean water detected, but some contamination found. Monitor water quality.';
    } else {
        return 'Significant contamination detected. Further investigation recommended.';
    }
}

// ==================== RESET CLASSIFIER ====================

function resetClassifier() {
    uploadedFiles = [];
    const gallery = document.getElementById('previewGallery');
    if (gallery) gallery.innerHTML = '';
    
    const resultCard = document.querySelector('.result-card');
    resultCard.innerHTML = `
        <div class="result-top">
            <div>
                <p>AI RESULT</p>
                <h2 id="resultClass">READY</h2>
            </div>
            <div class="confidence">
                <span id="confidenceText">0%</span>
            </div>
        </div>
        <div class="bar-group">
            <div class="bar-label">
                <span>Confidence</span>
                <span id="confidenceLabel">0%</span>
            </div>
            <div class="bar">
                <div class="bar-fill" id="confidenceBar"></div>
            </div>
        </div>
        <div class="summary-box">
            <h3>Analysis Summary</h3>
            <p id="summaryText">Waiting for images...</p>
        </div>
    `;

    console.log('🔄 Classifier reset');
}

// ==================== MODELS CAROUSEL ====================

const modelsData = [
    {
        emoji: '🍌',
        title: 'Banana Expert',
        description: 'Specialized CNN model for banana classification.',
        images: ['/photos/bc.jpg', '/photos/bi.jpg', '/photos/br.jpg']
    },
    {
        emoji: '🥚',
        title: 'Egg Classifier',
        description: 'Optimized AI for egg contamination analysis.',
        images: ['/photos/ec.jpg', '/photos/ei.jpg', '/photos/er.jpg']
    },
    {
        emoji: '🍊',
        title: 'Orange Pro',
        description: 'Advanced classification system for oranges.',
        images: ['/photos/oc.jpg', '/photos/oi.jpg', '/photos/or.jpg']
    }
];

let currentModelIndex = 1;

class ModelsCarousel {
    constructor() {
        this.init();
    }

    init() {
        this.centerEmoji = document.getElementById('center-emoji');
        this.centerTitle = document.getElementById('center-title');
        this.centerDescription = document.getElementById('center-description');
        this.modelImages = document.querySelector('.model-images');
        this.dots = document.querySelectorAll('.dot');
        this.focusedCard = document.querySelector('.focused-card');
        this.sideModels = document.querySelectorAll('.side-model-item');

        if (!this.centerEmoji || !this.focusedCard) {
            console.error('❌ Carousel: Required DOM elements not found!');
            return;
        }

        console.log('✅ Carousel initialized');
        this.attachEventListeners();
        this.updateDisplay(currentModelIndex);
    }

    attachEventListeners() {
        this.dots.forEach((dot, index) => {
            dot.addEventListener('click', () => {
                this.switchModel(index);
            });
        });

        this.sideModels.forEach((item) => {
            item.addEventListener('click', () => {
                const dataIndex = parseInt(item.getAttribute('data-index'));
                this.switchModel(dataIndex);
            });
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft') this.switchModel(currentModelIndex - 1);
            if (e.key === 'ArrowRight') this.switchModel(currentModelIndex + 1);
        });
    }

    switchModel(newIndex) {
        if (newIndex < 0) newIndex = modelsData.length - 1;
        if (newIndex >= modelsData.length) newIndex = 0;

        if (newIndex === currentModelIndex) return;

        if (this.focusedCard) {
            this.focusedCard.classList.add('transition-out');
        }

        setTimeout(() => {
            currentModelIndex = newIndex;
            this.updateDisplay(newIndex);
            
            if (this.focusedCard) {
                this.focusedCard.classList.remove('transition-out');
                this.focusedCard.classList.add('transition-in');

                setTimeout(() => {
                    this.focusedCard.classList.remove('transition-in');
                }, 500);
            }
        }, 250);
    }

    updateDisplay(index) {
        const model = modelsData[index];

        if (this.centerEmoji) this.centerEmoji.textContent = model.emoji;
        if (this.centerTitle) this.centerTitle.textContent = model.title;
        if (this.centerDescription) this.centerDescription.textContent = model.description;

        if (this.modelImages) {
            this.modelImages.innerHTML = model.images
                .map(img => `<img src="${img}" alt="${model.title} visualization" class="model-img">`)
                .join('');
        }

        this.dots.forEach((dot, i) => {
            dot.classList.toggle('active', i === index);
        });

        this.sideModels.forEach((item) => {
            const itemIndex = parseInt(item.getAttribute('data-index'));
            item.classList.toggle('active', itemIndex === index);
        });
    }
}

document.addEventListener('DOMContentLoaded', () => {
    console.log('📦 DOM Content Loaded, initializing carousel...');
    new ModelsCarousel();
});
// ==================== CAMERA ====================

let cameraStream = null;
let facingMode = 'environment'; // start with rear camera

async function openCamera() {
    const modal = document.getElementById('cameraModal');
    modal.classList.add('active');
    await startStream();
}

async function startStream() {
    // Stop any existing stream
    if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
        cameraStream = null;
    }

    setHint('Starting camera...');

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode, width: { ideal: 1280 }, height: { ideal: 960 } },
            audio: false
        });

        const video = document.getElementById('cameraFeed');
        video.srcObject = cameraStream;
        await video.play();
        setHint('Position the sample inside the frame');
    } catch (err) {
        console.error('Camera error:', err);
        if (err.name === 'NotAllowedError') {
            setHint('❌ Camera permission denied. Please allow access.');
        } else if (err.name === 'NotFoundError') {
            setHint('❌ No camera found on this device.');
        } else {
            setHint('❌ Could not start camera: ' + err.message);
        }
    }
}

function closeCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(t => t.stop());
        cameraStream = null;
    }
    document.getElementById('cameraModal').classList.remove('active');
    setHint('Position the sample inside the frame');
}

async function switchCamera() {
    facingMode = facingMode === 'environment' ? 'user' : 'environment';
    await startStream();
}

function captureAndClassify() {
    const video = document.getElementById('cameraFeed');
    const canvas = document.getElementById('cameraCanvas');

    if (!cameraStream || video.readyState < 2) {
        setHint('⚠️ Camera not ready yet, please wait.');
        return;
    }

    // Flash effect
    const viewfinder = document.querySelector('.camera-viewfinder');
    const flash = document.createElement('div');
    flash.className = 'camera-flash';
    viewfinder.appendChild(flash);
    setTimeout(() => flash.remove(), 400);

    // Draw frame to canvas
    canvas.width  = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext('2d').drawImage(video, 0, 0);

    // Convert to Blob and save to uploadedFiles — no classify yet
    canvas.toBlob((blob) => {
        if (!blob) {
            setHint('❌ Failed to capture image.');
            return;
        }

        const fileName = `camera-capture-${Date.now()}.jpg`;
        const file = new File([blob], fileName, { type: 'image/jpeg' });

        // Add to the uploadedFiles array
        uploadedFiles.push(file);

        // Show/update the preview gallery
        const galleryDiv = document.getElementById('previewGallery') || createPreviewGallery();
        const reader = new FileReader();
        reader.onload = (e) => {
            const item = document.createElement('div');
            item.className = 'preview-item';
            item.innerHTML = `
                <img src="${e.target.result}" alt="Capture ${uploadedFiles.length}">
                <span class="preview-count">${uploadedFiles.length}</span>
            `;
            galleryDiv.appendChild(item);
            galleryDiv.style.display = 'grid';
        };
        reader.readAsDataURL(file);

        setHint(`✅ Photo saved (${uploadedFiles.length} total). Capture more or close to classify.`);
        console.log(`📸 Camera photo added. Total: ${uploadedFiles.length}`);
    }, 'image/jpeg', 0.92);
}

function setHint(text) {
    const el = document.getElementById('cameraHint');
    if (el) el.textContent = text;
}

function resetCaptureBtn() {
    const btn = document.getElementById('captureBtn');
    if (btn) btn.classList.remove('scanning');
}

function showCameraResult(result, modelType) {
    const pct = (result.confidence * 100).toFixed(1);
    const isClean = result.condition.includes('Clean');
    const icon = isClean ? '✅' : '⚠️';

    // Add to uploadedFiles so the main classify flow also works
    // (just display the result card directly)
    const resultCard = document.querySelector('.result-card');
    if (!resultCard) return;

    resultCard.innerHTML = `
        <div class="result-top">
            <div>
                <p>AI RESULT — CAMERA</p>
                <h2 id="resultClass">${icon} ${result.condition.toUpperCase()}</h2>
            </div>
            <div class="confidence">
                <span id="confidenceText">${pct}%</span>
            </div>
        </div>
        <div class="bar-group">
            <div class="bar-label">
                <span>Confidence</span>
                <span id="confidenceLabel">${pct}%</span>
            </div>
            <div class="bar">
                <div class="bar-fill" id="confidenceBar" style="width:${pct}%"></div>
            </div>
        </div>
        <div class="summary-box">
            <h3>Analysis Summary</h3>
            <p id="summaryText">
                Model: <strong>${modelType.toUpperCase()}</strong><br>
                Source: <strong>Camera Capture</strong><br><br>
                ${isClean
                    ? '✅ Water appears <strong>clean</strong>. No iron contamination detected in this sample.'
                    : '⚠️ <strong>Iron contamination detected.</strong> Consider filtering or treatment before use.'}
            </p>
        </div>
        <div class="action-buttons" style="margin-top:20px;">
            <button class="btn-secondary" onclick="resetClassifier()">↩ Reset</button>
        </div>
    `;

    // Scroll to result
    resultCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// Close modal on backdrop click
document.addEventListener('DOMContentLoaded', () => {
    const modal = document.getElementById('cameraModal');
    if (modal) {
        modal.addEventListener('click', (e) => {
            if (e.target === modal) closeCamera();
        });
    }
});

// Close modal on Escape key
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && document.getElementById('cameraModal')?.classList.contains('active')) {
        closeCamera();
    }
});

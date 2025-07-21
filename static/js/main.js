/**
 * Golf Camera System - Frontend JavaScript
 * Modular, maintainable frontend code with proper error handling
 */

// === GLOBAL STATE ===
const AppState = {
    isRecording: false,
    autoRecordingEnabled: false,
    poseDetectionEnabled: true,
    modelsLoaded: false,
    
    // UI Elements
    elements: {
        recordBtn: null,
        autoBtn: null,
        poseBtn: null,
        reloadBtn: null,
        statusDiv: null,
        durationSelect: null,
        
        // Status displays
        predictedPose: null,
        currentStage: null,
        p1Confidence: null,
        p10Confidence: null,
        memoryEfficiency: null,
        framesInPool: null,
        uploadQueue: null,
        uploadSuccess: null
    }
};

// === UTILITY FUNCTIONS ===
const Utils = {
    /**
     * Show status message with styling
     */
    showStatus(message, type = 'success', duration = 5000) {
        const statusDiv = AppState.elements.statusDiv;
        if (!statusDiv) return;
        
        statusDiv.className = `status ${type}`;
        statusDiv.innerHTML = message;
        statusDiv.style.display = 'block';
        
        // Auto-hide after duration
        if (duration > 0) {
            setTimeout(() => {
                statusDiv.style.display = 'none';
            }, duration);
        }
    },
    
    /**
     * Log error with context
     */
    logError(operation, error) {
        console.error(`❌ ${operation}:`, error);
        this.showStatus(`❌ Error: ${operation} - ${error.message}`, 'error');
    },
    
    /**
     * Make API request with error handling
     */
    async apiRequest(url, options = {}) {
        try {
            const response = await fetch(url, {
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                },
                ...options
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return await response.json();
        } catch (error) {
            this.logError(`API request to ${url}`, error);
            throw error;
        }
    },
    
    /**
     * Update button state
     */
    updateButtonState(button, enabled, activeText, inactiveText) {
        if (!button) return;
        
        button.textContent = enabled ? activeText : inactiveText;
        button.className = enabled ? 'toggle-btn active' : 'toggle-btn';
    },
    
    /**
     * Set button loading state
     */
    setButtonLoading(button, loading, originalText = '') {
        if (!button) return;
        
        if (loading) {
            button.disabled = true;
            button.dataset.originalText = button.textContent;
            button.textContent = '🔄 Loading...';
            button.classList.add('loading');
        } else {
            button.disabled = false;
            button.textContent = button.dataset.originalText || originalText;
            button.classList.remove('loading');
        }
    }
};

// === RECORDING FUNCTIONS ===
const Recording = {
    /**
     * Start manual recording
     */
    async start() {
        if (AppState.isRecording) {
            Utils.showStatus('❌ Already recording', 'warning');
            return;
        }
        
        const duration = parseInt(AppState.elements.durationSelect?.value || 5);
        const recordBtn = AppState.elements.recordBtn;
        
        Utils.setButtonLoading(recordBtn, true);
        AppState.isRecording = true;
        
        try {
            const data = await Utils.apiRequest('/start_recording', {
                method: 'POST',
                body: JSON.stringify({ duration })
            });
            
            if (data.status === 'success') {
                Utils.showStatus('🔴 Recording in progress (optimized performance!)...', 'recording', 0);
                this.checkRecordingStatus();
            } else {
                throw new Error(data.message);
            }
        } catch (error) {
            Utils.logError('Start recording', error);
            Utils.setButtonLoading(recordBtn, false);
            AppState.isRecording = false;
        }
    },
    
    /**
     * Monitor recording status
     */
    checkRecordingStatus() {
        const interval = setInterval(async () => {
            try {
                const data = await Utils.apiRequest('/system_status');
                
                UI.updateDisplays(data);
                
                if (!data.is_recording) {
                    clearInterval(interval);
                    AppState.isRecording = false;
                    Utils.setButtonLoading(AppState.elements.recordBtn, false);
                    
                    const stopMessage = data.stop_recording_early ? 
                        ' (stopped early on P10 detection)' : ' (full duration)';
                    
                    Utils.showStatus(
                        `✅ Recording complete${stopMessage}!<br>` +
                        `📁 Local: ${data.video_path}<br>` +
                        `📤 Upload: Queued in background (${data.upload_queue_size} in queue)<br>` +
                        `💾 Memory efficiency: ${data.memory_efficiency}%`,
                        'success'
                    );
                }
            } catch (error) {
                Utils.logError('Recording status check', error);
                clearInterval(interval);
                Utils.setButtonLoading(AppState.elements.recordBtn, false);
                AppState.isRecording = false;
            }
        }, 1000);
    }
};

// === AI FUNCTIONS ===
const AI = {
    /**
     * Toggle auto recording
     */
    async toggleAutoRecording() {
        const autoBtn = AppState.elements.autoBtn;
        
        try {
            const data = await Utils.apiRequest('/toggle_auto_recording', { method: 'POST' });
            
            if (data.status === 'success') {
                AppState.autoRecordingEnabled = data.auto_recording_enabled;
                Utils.updateButtonState(
                    autoBtn,
                    AppState.autoRecordingEnabled,
                    '🎯 Auto Recording: ON',
                    '🎯 Auto Recording: OFF'
                );
                
                if (AppState.autoRecordingEnabled) {
                    Utils.showStatus(
                        '🤖 Auto recording enabled - Stand in P1 address position!',
                        'auto-enabled'
                    );
                } else {
                    Utils.showStatus('Auto recording disabled', 'success');
                }
            } else {
                throw new Error(data.message);
            }
        } catch (error) {
            Utils.logError('Toggle auto recording', error);
        }
    },
    
    /**
     * Toggle pose detection
     */
    async togglePoseDetection() {
        const poseBtn = AppState.elements.poseBtn;
        
        try {
            const data = await Utils.apiRequest('/toggle_pose_detection', { method: 'POST' });
            
            AppState.poseDetectionEnabled = data.pose_detection_enabled;
            Utils.updateButtonState(
                poseBtn,
                AppState.poseDetectionEnabled,
                '📊 Pose Detection: ON',
                '📊 Pose Detection: OFF'
            );
            
            Utils.showStatus(
                `Pose detection ${AppState.poseDetectionEnabled ? 'enabled' : 'disabled'}`,
                'success'
            );
        } catch (error) {
            Utils.logError('Toggle pose detection', error);
        }
    },
    
    /**
     * Reload AI models
     */
    async reloadModels(forceDownload = false) {
        const reloadBtn = AppState.elements.reloadBtn;
        Utils.setButtonLoading(reloadBtn, true);
        
        try {
            const data = await Utils.apiRequest('/reload_models', {
                method: 'POST',
                body: JSON.stringify({ force_download: forceDownload })
            });
            
            AppState.modelsLoaded = data.models_loaded;
            Utils.showStatus(`✅ ${data.message}`, data.status === 'success' ? 'success' : 'error');
        } catch (error) {
            Utils.logError('Reload models', error);
        } finally {
            Utils.setButtonLoading(reloadBtn, false);
        }
    }
};

// === DEBUG FUNCTIONS ===
const Debug = {
    /**
     * Get model information
     */
    async modelInfo() {
        try {
            const data = await Utils.apiRequest('/debug_model_info');
            
            if (data.model_info) {
                Utils.showStatus(
                    `<h4>🔍 Model Debug Info:</h4>` +
                    `<p><strong>Input Shape:</strong> ${data.model_info.input_shape}</p>` +
                    `<p><strong>Output Shape:</strong> ${data.model_info.output_shape}</p>` +
                    `<p><strong>Classes:</strong> ${data.model_info.class_names?.join(', ')}</p>` +
                    `<p>Check browser console for full details.</p>`,
                    'success'
                );
                console.log('🔍 Full Model Info:', data);
            } else {
                throw new Error(data.message || 'No model info received');
            }
        } catch (error) {
            Utils.logError('Get model info', error);
        }
    },
    
    /**
     * Test model with current pose
     */
    async testCurrentPose() {
        try {
            const data = await Utils.apiRequest('/test_model_with_pose', { method: 'POST' });
            
            if (data.success) {
                Utils.showStatus(
                    `<h4>🧪 Current Pose Test:</h4>` +
                    `<p><strong>Max Prediction:</strong> Index ${data.predicted_index} = ${data.predicted_confidence?.toFixed(4)}</p>` +
                    `<p><strong>Predicted Class:</strong> ${data.predicted_class}</p>` +
                    `<p>Check browser console for full details.</p>`,
                    'success'
                );
                console.log('🧪 Full Pose Test:', data);
            } else {
                throw new Error(data.error || 'Pose test failed');
            }
        } catch (error) {
            Utils.logError('Test current pose', error);
        }
    },
    
    /**
     * Get comprehensive system debug info
     */
    async systemDebug() {
        try {
            const data = await Utils.apiRequest('/system_debug');
            
            Utils.showStatus(
                `<h4>🔍 System Debug Info Retrieved:</h4>` +
                `<p>Check browser console for comprehensive system information.</p>`,
                'success'
            );
            console.log('🔍 System Debug Info:', data);
        } catch (error) {
            Utils.logError('Get system debug info', error);
        }
    }
};

// === UPLOAD & PERFORMANCE FUNCTIONS ===
const Performance = {
    /**
     * Check recent uploads
     */
    async checkUploads() {
        try {
            const data = await Utils.apiRequest('/recent_uploads');
            
            Utils.showStatus(
                `<h4>📤 Recent Uploads:</h4>` +
                `<p>Found ${Object.keys(data).length} recent uploads</p>` +
                `<p>Check browser console for full details.</p>`,
                'success'
            );
            console.log('📤 Recent Uploads:', data);
        } catch (error) {
            Utils.logError('Check uploads', error);
        }
    },
    
    /**
     * Check upload queue status
     */
    async checkUploadQueue() {
        try {
            const data = await Utils.apiRequest('/upload_queue_status');
            
            Utils.showStatus(
                `<h4>🔄 Upload Queue Status:</h4>` +
                `<p><strong>Queue Size:</strong> ${data.queue_size}</p>` +
                `<p><strong>Success Rate:</strong> ${data.stats?.success_rate}%</p>` +
                `<p>Check browser console for full details.</p>`,
                'success'
            );
            console.log('🔄 Upload Queue Status:', data);
        } catch (error) {
            Utils.logError('Check upload queue', error);
        }
    },
    
    /**
     * Check memory statistics
     */
    async checkMemoryStats() {
        try {
            const data = await Utils.apiRequest('/memory_stats');
            
            Utils.showStatus(
                `<h4>💾 Memory Efficiency Stats:</h4>` +
                `<p><strong>Frame Reuse Efficiency:</strong> ${data.frame_pool?.reuse_efficiency_percent}%</p>` +
                `<p><strong>Frames Reused:</strong> ${data.frame_pool?.frames_reused}</p>` +
                `<p><strong>Pool Size:</strong> ${data.frame_pool?.current_pool_size}</p>` +
                `<p>Check browser console for full details.</p>`,
                'success'
            );
            console.log('💾 Memory Stats:', data);
        } catch (error) {
            Utils.logError('Check memory stats', error);
        }
    }
};

// === UI UPDATE FUNCTIONS ===
const UI = {
    /**
     * Update all display elements
     */
    updateDisplays(data) {
        const elements = AppState.elements;
        
        // Update pose information
        if (elements.predictedPose) elements.predictedPose.textContent = data.predicted_class || 'Unknown';
        if (elements.p1Confidence) elements.p1Confidence.textContent = (data.p1_confidence || 0).toFixed(2);
        if (elements.p10Confidence) elements.p10Confidence.textContent = (data.p10_confidence || 0).toFixed(2);
        if (elements.currentStage) elements.currentStage.textContent = data.current_pose_stage || 'waiting';
        
        // Update performance metrics
        if (elements.memoryEfficiency) elements.memoryEfficiency.textContent = (data.memory_efficiency || 0) + '%';
        if (elements.framesInPool) elements.framesInPool.textContent = data.frames_in_pool || '--';
        if (elements.uploadQueue) elements.uploadQueue.textContent = data.upload_queue_size || '--';
        if (elements.uploadSuccess) elements.uploadSuccess.textContent = (data.upload_success_rate || 0) + '%';
        
        // Update global state
        AppState.modelsLoaded = data.models_loaded || false;
        AppState.autoRecordingEnabled = data.auto_recording_enabled || false;
        AppState.poseDetectionEnabled = data.pose_detection_enabled !== false;
    },
    
    /**
     * Initialize UI elements
     */
    initializeElements() {
        const elements = AppState.elements;
        
        // Control buttons
        elements.recordBtn = document.getElementById('recordBtn');
        elements.autoBtn = document.getElementById('autoBtn');
        elements.poseBtn = document.getElementById('poseBtn');
        elements.reloadBtn = document.getElementById('reloadBtn');
        elements.statusDiv = document.getElementById('status');
        elements.durationSelect = document.getElementById('duration');
        
        // Status displays
        elements.predictedPose = document.getElementById('predictedPose');
        elements.currentStage = document.getElementById('currentStage');
        elements.p1Confidence = document.getElementById('p1Confidence');
        elements.p10Confidence = document.getElementById('p10Confidence');
        elements.memoryEfficiency = document.getElementById('memoryEfficiency');
        elements.framesInPool = document.getElementById('framesInPool');
        elements.uploadQueue = document.getElementById('uploadQueue');
        elements.uploadSuccess = document.getElementById('uploadSuccess');
    },
    
    /**
     * Start periodic status updates
     */
    startPeriodicUpdates() {
        // Update every 2 seconds when not recording
        setInterval(async () => {
            if (!AppState.isRecording) {
                try {
                    const data = await Utils.apiRequest('/system_status');
                    this.updateDisplays(data);
                } catch (error) {
                    console.warn('Periodic status update failed:', error);
                }
            }
        }, 2000);
    }
};

// === GLOBAL FUNCTIONS (for HTML onclick handlers) ===
window.startRecording = () => Recording.start();
window.toggleAutoRecording = () => AI.toggleAutoRecording();
window.togglePoseDetection = () => AI.togglePoseDetection();
window.reloadModels = () => AI.reloadModels();
window.debugModelInfo = () => Debug.modelInfo();
window.testModelWithPose = () => Debug.testCurrentPose();
window.getSystemDebug = () => Debug.systemDebug();
window.checkUploads = () => Performance.checkUploads();
window.checkUploadQueue = () => Performance.checkUploadQueue();
window.checkMemoryStats = () => Performance.checkMemoryStats();

// === INITIALIZATION ===
document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 Golf Camera System - Frontend Initialized');
    
    // Initialize UI
    UI.initializeElements();
    UI.startPeriodicUpdates();
    
    // Load initial model
    AI.reloadModels();
    
    console.log('✅ Frontend initialization complete');
});

// === ERROR HANDLING ===
window.addEventListener('error', (event) => {
    console.error('🚨 Global error:', event.error);
    Utils.showStatus(`🚨 Application error: ${event.error?.message}`, 'error');
});

window.addEventListener('unhandledrejection', (event) => {
    console.error('🚨 Unhandled promise rejection:', event.reason);
    Utils.showStatus(`🚨 Network error: ${event.reason?.message}`, 'error');
});

// === EXPORT FOR TESTING ===
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { AppState, Utils, Recording, AI, Debug, Performance, UI };
}
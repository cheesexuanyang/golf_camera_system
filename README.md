# Golf Camera System - Modular Architecture

An advanced AI-powered golf swing analysis system built for Raspberry Pi with real-time pose detection, automatic recording, and cloud storage integration.

## Features

### Core Functionality
- **Automatic Recording**: AI-triggered recording based on golf pose detection (P1 address → swing sequence → P10 finish)
- **Real-time Pose Analysis**: 10-stage golf swing classification (P1-P10) using MediaPipe and TensorFlow
- **Cloud Integration**: Automatic background uploads to Google Cloud Storage
- **Web Interface**: Real-time video streaming with pose overlays and system controls
- **Memory Optimization**: Efficient frame pooling system for reduced memory allocation overhead

### Technical Highlights
- **Modular Architecture**: Clean separation of concerns across camera, AI, storage, and utility modules
- **Performance Optimized**: Frame skipping, memory pooling, and background processing
- **Error Resilient**: Comprehensive error handling and logging throughout the system
- **Scalable Design**: Easy to extend and maintain with clear module boundaries

## Project Structure

```
golf-camera-system/
├── ai/                     # AI/ML modules
│   ├── __init__.py
│   ├── model_manager.py    # TensorFlow model management
│   ├── pose_classifier.py  # Golf pose classification
│   └── pose_detector.py    # MediaPipe pose detection
├── camera/                 # Camera operations
│   ├── __init__.py
│   ├── camera_manager.py   # Raspberry Pi camera control
│   └── video_recorder.py   # Video recording with early stop
├── config/                 # Configuration management
│   ├── __init__.py
│   └── settings.py         # Centralized settings
├── storage/                # Cloud storage
│   ├── __init__.py
│   └── uploader.py         # Background GCS uploads
├── utils/                  # Utility modules
│   ├── __init__.py
│   ├── frame_pool.py       # Memory optimization
│   ├── helpers.py          # Common utilities
│   └── logger.py           # Centralized logging
├── static/                 # Web interface assets
│   ├── css/style.css
│   └── js/main.js
├── templates/              # HTML templates
│   └── index.html
├── app.py                  # Main Flask application
├── requirements.txt        # Python dependencies
└── README.md
```

## Installation

### Prerequisites
- Raspberry Pi 4 (recommended) with camera module
- Python 3.8+
- Google Cloud Storage account (for uploads)
- Minimum 4GB RAM recommended

### System Dependencies
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3-pip python3-venv
sudo apt install -y python3-opencv python3-numpy
sudo apt install -y libatlas-base-dev libhdf5-dev libhdf5-serial-dev
sudo apt install -y python3-picamera2

# Enable camera interface
sudo raspi-config
# Navigate to Interface Options > Camera > Enable
```

### Python Environment Setup
```bash
# Clone repository
git clone <your-repo-url>
cd golf-camera-system

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### Google Cloud Setup
```bash
# Install Google Cloud SDK (optional, for authentication)
curl https://sdk.cloud.google.com | bash
exec -l $SHELL

# Authenticate (or use service account key)
gcloud auth application-default login

# Or set service account key
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account-key.json"
```

## Configuration

### Basic Configuration
Edit `config/settings.py` to customize your setup:

```python
# === GOOGLE CLOUD STORAGE ===
BUCKET_NAME = "your-golf-swing-bucket"
GCS_MODEL_NAME = "your_model.keras"

# === CAMERA CONFIGURATION ===
CAMERA_PREVIEW_SIZE = (1280, 720)
CAMERA_RECORDING_SIZE = (1920, 1080)

# === AI MODEL CONFIGURATION ===
CLASS_NAMES = ['P1', 'P10', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9']
P1_CONFIDENCE_THRESHOLD = 0.7
P10_CONFIDENCE_THRESHOLD = 0.7

# === RECORDING DEFAULTS ===
DEFAULT_RECORDING_DURATION = 5  # seconds
AUTO_RECORDING_DURATION = 20    # seconds for auto-triggered recordings
```

### Directory Setup
The system will automatically create necessary directories:
- `/home/raspberry/Videos/` - Local video storage
- `/home/raspberry/models/` - AI model storage

### Model Setup
1. Train your golf pose classification model or obtain a pre-trained model
2. Upload your model to Google Cloud Storage as `best_model.keras`
3. The system will automatically download and load the model on startup

## Usage

### Starting the System
```bash
# Activate virtual environment
source venv/bin/activate

# Start the application
python app.py
```

The system will start on `http://localhost:5000` (or your Raspberry Pi's IP address).

### Web Interface

#### Main Controls
- **Start Recording**: Manual recording with configurable duration
- **Auto Recording**: AI-triggered recording based on pose detection
- **Pose Detection**: Toggle real-time pose analysis
- **Reload Model**: Refresh the AI model from cloud storage

#### Status Display
- Real-time pose classification (P1-P10)
- Confidence scores for key poses (P1 address, P10 finish)
- Memory efficiency and performance metrics
- Upload queue status

#### Debug Tools
- Model information and testing
- System diagnostics
- Upload monitoring
- Memory usage statistics

### Auto Recording Workflow
1. **Enable Auto Recording**: Click "Auto Recording: OFF" to enable
2. **Stand in Address Position**: System detects P1 pose with high confidence
3. **Begin Swing**: Movement from P1 triggers recording
4. **Automatic Stop**: P10 finish pose or 20-second timeout stops recording
5. **Background Upload**: Video automatically uploads to Google Cloud Storage

### Manual Recording
1. Select duration (3-20 seconds)
2. Click "Start Recording"
3. Perform your golf swing
4. Video saves locally and uploads in background

## API Endpoints

### Recording Control
- `POST /start_recording` - Start manual recording
- `POST /toggle_auto_recording` - Toggle auto recording
- `POST /toggle_pose_detection` - Toggle pose detection

### System Status
- `GET /system_status` - Comprehensive system status
- `GET /recording_status` - Recording status (legacy)
- `GET /upload_queue_status` - Upload queue information

### Debug & Monitoring
- `GET /debug_model_info` - AI model information
- `POST /test_model_with_pose` - Test model with current frame
- `GET /system_debug` - Comprehensive debug information
- `GET /memory_stats` - Memory efficiency statistics
- `GET /recent_uploads` - Recent upload history

### Model Management
- `POST /reload_models` - Reload AI model from cloud storage

## AI Model Requirements

### Input Format
- **Shape**: `(batch_size, 132)` - Flattened pose landmarks
- **Normalization**: Hip-centered coordinate system with distance normalization
- **Landmarks**: 33 MediaPipe pose landmarks × 4 coordinates (x, y, z, visibility)

### Output Format
- **Classes**: 10 golf swing poses (P1-P10)
- **Output**: Softmax probabilities for each class

### Training Data Format
The system expects landmarks normalized using this method:
```python
# Hip-centered normalization (from pose_classifier.py)
left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
center_x = (left_hip.x + right_hip.x) / 2
center_y = (left_hip.y + right_hip.y) / 2
max_distance = max([sqrt((lm.x - center_x)² + (lm.y - center_y)²) for lm in landmarks])

normalized_landmarks = [
    [(landmark.x - center_x) / max_distance,
     (landmark.y - center_y) / max_distance,
     landmark.z / max_distance,
     landmark.visibility] 
    for landmark in landmarks
].flatten()
```

## Performance Optimization

### Memory Management
- **Frame Pooling**: Pre-allocated frame buffers reduce garbage collection
- **Efficient Processing**: Frame skipping and interval-based processing
- **Background Operations**: Non-blocking uploads and AI inference

### Monitoring
- Real-time memory efficiency metrics
- Upload success rates and queue monitoring
- Performance timing for critical operations

## Troubleshooting

### Common Issues

#### Camera Not Detected
```bash
# Check camera connection
vcgencmd get_camera

# Enable camera interface
sudo raspi-config
```

#### Model Loading Fails
```bash
# Check Google Cloud authentication
gcloud auth list

# Verify bucket access
gsutil ls gs://your-bucket-name/

# Check model file exists
gsutil ls gs://your-bucket-name/best_model.keras
```

#### High Memory Usage
- Reduce `FRAME_POOL_SIZE` in settings
- Increase `FRAME_SKIP_INTERVAL` for less frequent processing
- Lower camera resolution in `CAMERA_PREVIEW_SIZE`

#### Upload Failures
- Verify internet connection
- Check Google Cloud Storage permissions
- Monitor upload queue: `GET /upload_queue_status`

### Debug Mode
Enable detailed logging by setting:
```python
LOG_LEVEL = "DEBUG"
FLASK_DEBUG = True
```

### System Diagnostics
Use the web interface debug tools:
- **System Debug**: Comprehensive system information
- **Test Current Pose**: Validate AI model with current frame
- **Memory Stats**: Detailed memory usage analysis

## Security Considerations

- **Network Access**: System binds to `0.0.0.0:5000` - consider firewall rules
- **File Permissions**: Ensure proper permissions for video and model directories
- **Cloud Credentials**: Secure your Google Cloud service account keys
- **Local Storage**: Videos are temporarily stored locally before upload

## Development

### Adding New Modules
The modular architecture makes it easy to extend:

1. **Create Module Directory**: Follow the pattern of existing modules
2. **Add `__init__.py`**: Export main classes/functions
3. **Import in Main App**: Add to `app.py` imports
4. **Update Configuration**: Add settings to `config/settings.py`

### Code Style
- Follow PEP 8 python style guidelines
- Use type hints where appropriate
- Include comprehensive error handling
- Add logging for important operations

### Testing
```bash
# Run with debug mode
FLASK_DEBUG=True python app.py

# Use web interface debug tools
# Monitor logs for issues
tail -f /var/log/your-app.log
```

## Key Benefits of This Architecture

### Modularity
- **Independent Components**: Each module has a single responsibility
- **Easy Testing**: Modules can be tested in isolation
- **Simple Maintenance**: Changes in one module don't affect others

### Performance
- **Memory Efficient**: Frame pooling reduces allocation overhead
- **Non-blocking Operations**: Background uploads and processing
- **Optimized AI Pipeline**: Smart frame processing intervals

### Reliability
- **Comprehensive Error Handling**: Graceful degradation on failures
- **Automatic Recovery**: System continues operation despite individual component failures
- **Detailed Logging**: Full visibility into system operation

### Scalability
- **Cloud Integration**: Automatic uploads and model management
- **Configurable Performance**: Tunable parameters for different hardware
- **Extensible Design**: Easy to add new features and capabilities

This modular architecture ensures your golf camera system is professional, maintainable, and ready for production use.

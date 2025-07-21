🏌️ Golf Camera System - Modular Architecture
A professional, modular AI-powered golf swing analysis system built for Raspberry Pi with clean separation of concerns and enterprise-grade architecture.

📁 Project Structure
golf_camera_system/
├── app.py                    # Main Flask application (entry point)
├── requirements.txt          # Dependencies
├── README.md                # This documentation
├── config/
│   ├── __init__.py
│   └── settings.py          # All configuration settings
├── camera/
│   ├── __init__.py
│   ├── camera_manager.py    # Camera operations & streaming
│   └── video_recorder.py    # Video recording logic
├── ai/
│   ├── __init__.py
│   ├── pose_detector.py     # MediaPipe pose detection
│   ├── model_manager.py     # TensorFlow model handling
│   └── pose_classifier.py   # Golf pose classification
├── storage/
│   ├── __init__.py
│   └── uploader.py          # Background upload system
├── utils/
│   ├── __init__.py
│   ├── frame_pool.py        # Memory management
│   ├── helpers.py           # Utility functions
│   └── logger.py            # Centralized logging
├── templates/
│   └── index.html           # Web interface template
└── static/
    ├── css/
    │   └── style.css        # Separated styles
    └── js/
        └── main.js          # Separated JavaScript

🚀 Features
Core Functionality
AI Golf Pose Detection: 10-class pose classification (P1-P10)
Auto Recording: Automatically starts/stops recording based on golf poses
Manual Recording: Configurable duration recording
Real-time Streaming: Live camera feed with pose overlays
Performance Optimizations
Background Uploads: Non-blocking uploads to Google Cloud Storage
Frame Pooling: Memory-efficient frame reuse system
Optimized Processing: Frame skipping and efficient encoding
Professional Features
Modular Architecture: Clean separation of concerns
Comprehensive Logging: Centralized logging with categories
Error Handling: Robust error handling throughout
Status Monitoring: Real-time system status and performance metrics

🛠️ Installation
Prerequisites
Raspberry Pi 4 (recommended) with camera module
Python 3.8+
Google Cloud Storage account and credentials
1. System Dependencies
bash
sudo apt update
sudo apt install python3-pip python3-opencv python3-numpy
sudo apt install libatlas-base-dev libhdf5-dev libhdf5-serial-dev
sudo apt install python3-picamera2
2. Python Dependencies
bash
cd golf_camera_system
pip3 install -r requirements.txt
3. Google Cloud Setup
bash
# Install Google Cloud SDK (optional)
curl https://sdk.cloud.google.com | bash

# Set up authentication
gcloud auth application-default login
4. Configuration
Edit config/settings.py to match your setup:

python
BUCKET_NAME = "your-gcs-bucket-name"
GCS_MODEL_NAME = "your-model-file.keras"
VIDEO_DIR = "/path/to/your/videos"

🎯 Usage
Starting the System
bash
python3 app.py
Web Interface
Navigate to http://your-pi-ip:5000 to access the web interface.

API Endpoints
GET / - Web interface
GET /video_feed - Live camera stream
POST /start_recording - Start manual recording
POST /toggle_auto_recording - Toggle auto recording
POST /toggle_pose_detection - Toggle pose detection
POST /reload_models - Reload AI model
GET /system_status - Get system status
GET /memory_stats - Get memory efficiency stats
GET /recent_uploads - Get upload history

🔧 Configuration
Core Settings (config/settings.py)
python
# Camera settings
CAMERA_PREVIEW_SIZE = (640, 480)
CAMERA_RECORDING_SIZE = (1280, 720)

# AI thresholds
P1_CONFIDENCE_THRESHOLD = 0.7
P10_CONFIDENCE_THRESHOLD = 0.7

# Performance settings
FRAME_POOL_SIZE = 8
UPLOAD_MAX_WORKERS = 2
Environment Variables
bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
export FLASK_ENV=production  # or development

📊 Monitoring
System Status
The web interface provides real-time monitoring of:

AI model status and predictions
Recording status and queue
Memory efficiency metrics
Upload queue and success rates
Camera status and performance
Logging
Logs are categorized by component:

📷 Camera operations
🧠 AI/ML operations
☁️ Upload operations
🎥 Recording operations
⚡ Performance metrics
🔧 Development
Adding New Features
1. New AI Model
python
# ai/new_model.py
from ai.model_manager import ModelManager

class NewModelManager(ModelManager):
    def __init__(self):
        super().__init__()
        # Custom implementation
2. New Storage Backend
python
# storage/new_backend.py
from storage.uploader import BackgroundUploader

class NewUploader(BackgroundUploader):
    def __init__(self):
        super().__init__()
        # Custom implementation
3. New Camera Features
python
# camera/new_feature.py
from camera.camera_manager import CameraManager

class EnhancedCameraManager(CameraManager):
    def __init__(self):
        super().__init__()
        # Custom implementation
Testing
bash
# Run individual module tests
python3 -m pytest tests/test_camera.py
python3 -m pytest tests/test_ai.py
python3 -m pytest tests/test_storage.py

# Run all tests
python3 -m pytest

🐛 Troubleshooting
Common Issues
Camera Not Working
bash
# Check camera detection
libcamera-hello

# Check picamera2 installation
python3 -c "from picamera2 import Picamera2; print('OK')"
AI Model Issues
Check model file exists in GCS bucket
Verify Google Cloud credentials
Check model format compatibility
Memory Issues
Monitor memory efficiency in web interface
Adjust FRAME_POOL_SIZE in settings
Check system memory with htop
Upload Issues
Verify GCS bucket permissions
Check network connectivity
Monitor upload queue status
Debug Mode
bash
# Enable debug logging
export FLASK_ENV=development
python3 app.py
System Diagnostics
Use the web interface debug endpoints:

Model Info: /debug_model_info
System Debug: /system_debug
Memory Stats: /memory_stats

📈 Performance Optimization
Memory Optimization
Frame pooling reduces memory allocation overhead
Configurable pool size based on available RAM
Efficient frame copying and reuse
Upload Optimization
Background uploads don't block recording
Retry logic with exponential backoff
Automatic cleanup of uploaded files
Processing Optimization
Frame skipping for real-time performance
Optimized JPEG encoding settings
Efficient pose processing pipeline

🤝 Contributing
Code Style
Follow PEP 8 conventions
Use type hints where appropriate
Add comprehensive docstrings
Include error handling
Module Guidelines
Keep modules focused on single responsibility
Use proper imports and exports
Add logging for important operations
Include comprehensive error handling
Pull Request Process
Create feature branch
Add tests for new functionality
Update documentation
Ensure all tests pass
Submit pull request
📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

🙏 Acknowledgments
MediaPipe team for pose detection
TensorFlow team for ML framework
Raspberry Pi Foundation for hardware platform
Flask team for web framework
📞 Support
For issues and questions:

Check the troubleshooting section
Review system logs
Use the debug endpoints
Create an issue with detailed logs and system info
Built with ❤️ for the golf community using professional software architecture patterns.


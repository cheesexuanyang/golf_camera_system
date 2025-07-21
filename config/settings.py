"""
Configuration settings for the Golf Camera System
All constants and configuration values are centralized here.
"""

import os

# === DIRECTORY CONFIGURATION ===
BASE_DIR = "/home/raspberry"
VIDEO_DIR = os.path.join(BASE_DIR, "Videos")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# === GOOGLE CLOUD STORAGE ===
BUCKET_NAME = "golf-swing-models"
GCS_VIDEO_PREFIX = "golf_videos"
GCS_MODEL_NAME = "best_model.keras"  # Adjust to your actual model name

# === CAMERA CONFIGURATION ===
CAMERA_PREVIEW_SIZE = (1280, 720)  
CAMERA_RECORDING_SIZE = (1920, 1080)  
CAMERA_FORMAT = "RGB888"
CAMERA_FPS = 30

# === AI MODEL CONFIGURATION ===
# Class names from refer.txt - TRY DIFFERENT ORDERS IF ALWAYS PREDICTING P10
# OPTION 1: Original order
CLASS_NAMES = ['P1', 'P10', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9']

# OPTION 2: If model was trained in numerical order, try:
# CLASS_NAMES = ['P1', 'P2', 'P3', 'P4', 'P5', 'P6', 'P7', 'P8', 'P9', 'P10']

# OPTION 3: If model has different order, check refer.txt for exact order used

# Ideal angles from refer.txt
IDEAL_SHOULDER_TILT = {
    'P1': 8, 'P2': 24, 'P3': 35, 'P4': 37, 'P5': 33, 
    'P6': 12, 'P7': 30, 'P8': 38, 'P9': 45, 'P10': 6
}

IDEAL_HIP_TILT = {
    'P1': 1, 'P2': 4, 'P3': 8, 'P4': 9, 'P5': 7, 
    'P6': 8, 'P7': 11, 'P8': 12, 'P9': 14, 'P10': 5
}

# Valid transitions from refer.txt
VALID_TRANSITIONS = {
    'P1': ['P1', 'P2'],
    'P10': ['P10'],
    'P2': ['P2', 'P3'],
    'P3': ['P3', 'P4'],
    'P4': ['P4', 'P5'],
    'P5': ['P5', 'P6'],
    'P6': ['P6', 'P7'],
    'P7': ['P7', 'P8'],
    'P8': ['P8', 'P9'],
    'P9': ['P9', 'P10']
}

# === POSE DETECTION THRESHOLDS ===
P1_CONFIDENCE_THRESHOLD = 0.7
P10_CONFIDENCE_THRESHOLD = 0.7
MIN_P1_DURATION = 1.0
SWING_DETECTION_WINDOW = 25.0  # seconds
COOLDOWN_PERIOD = 5.0  # seconds

# === MEDIAPIPE CONFIGURATION ===
MEDIAPIPE_CONFIG = {
    'static_image_mode': False,
    'model_complexity': 0,
    'enable_segmentation': False,
    'min_detection_confidence': 0.7,
    'min_tracking_confidence': 0.5
}

# === FRAME PROCESSING ===
FRAME_SKIP_INTERVAL = 1  # Process every 3rd frame
PROCESS_INTERVAL = 0.1   # Minimum time between processing (seconds)
JPEG_QUALITY = 60        # Lower quality for better performance

# === BACKGROUND UPLOAD CONFIGURATION ===
UPLOAD_MAX_WORKERS = 2
UPLOAD_MAX_RETRIES = 3
UPLOAD_RETRY_DELAY = 2   # Base delay for exponential backoff

# === FRAME POOL CONFIGURATION ===
FRAME_POOL_SIZE = 8
FRAME_SHAPE = (480, 640, 3)

# === RECORDING DEFAULTS ===
DEFAULT_RECORDING_DURATION = 5  # seconds
AUTO_RECORDING_DURATION = 20    # seconds for auto-triggered recordings

# === FLASK CONFIGURATION ===
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = False

# === LOGGING CONFIGURATION ===
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

def ensure_directories():
    """Create necessary directories if they don't exist"""
    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

def get_model_path():
    """Get the full path to the classification model"""
    return os.path.join(MODEL_DIR, "classification_model.keras")

def get_video_filename(timestamp_str):
    """Generate video filename with timestamp"""
    return f"golf_swing_{timestamp_str}.mp4"

def get_gcs_blob_name(filename):
    """Generate GCS blob name for video file"""
    return f"{GCS_VIDEO_PREFIX}/{filename}"
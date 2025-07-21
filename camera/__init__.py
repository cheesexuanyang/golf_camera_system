# camera/__init__.py
"""
Camera module for Golf Camera System
Handles Raspberry Pi camera operations and video recording
"""

from .camera_manager import CameraManager
from .video_recorder import VideoRecorder

__all__ = ['CameraManager', 'VideoRecorder']
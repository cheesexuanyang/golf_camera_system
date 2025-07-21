# ai/__init__.py
"""
AI module for Golf Camera System
Provides pose detection and golf swing classification
"""

from .pose_detector import PoseDetector
from .model_manager import ModelManager
from .pose_classifier import GolfPoseClassifier

__all__ = ['PoseDetector', 'ModelManager', 'GolfPoseClassifier']
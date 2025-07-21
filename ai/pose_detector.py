"""
MediaPipe pose detection module
Handles pose landmark detection and drawing
"""

import cv2
import time
from typing import Optional, Tuple, Any
import mediapipe as mp

from config.settings import MEDIAPIPE_CONFIG
from utils.logger import setup_logger, log_ai_event, log_error, log_success

logger = setup_logger(__name__)

class PoseDetector:
    """MediaPipe pose detection wrapper with error handling and optimization"""
    
    def __init__(self):
        """Initialize MediaPipe pose detector"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose_landmarks = self.mp_pose.PoseLandmark
        self.pose: Optional[Any] = None
        self.is_initialized = False
        
        self.initialize()
    
    def initialize(self) -> bool:
        """
        Initialize MediaPipe pose detector with error handling
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Close existing pose detector if it exists
            if self.pose is not None:
                self.pose.close()
            
            # Create new pose detector with configuration
            self.pose = self.mp_pose.Pose(**MEDIAPIPE_CONFIG)
            
            self.is_initialized = True
            log_success(logger, "MediaPipe pose detector initialized", 
                       f"config: {MEDIAPIPE_CONFIG}")
            return True
            
        except Exception as e:
            log_error(logger, "PoseDetector.initialize", e)
            self.is_initialized = False
            return False
    
    def detect_pose(self, frame) -> Tuple[Any, Optional[Any]]:
        """
        Detect pose landmarks in frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (processed_frame, pose_results)
        """
        if not self.is_initialized or self.pose is None:
            return frame, None
        
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame = rgb_frame.copy()  # Ensure frame is writable
            
            # Process with MediaPipe
            results = self.pose.process(rgb_frame)
            
            return frame, results
            
        except Exception as e:
            log_error(logger, "PoseDetector.detect_pose", e)
            return frame, None
    
    def draw_landmarks(self, frame, results) -> None:
        """
        Draw pose landmarks on frame
        
        Args:
            frame: Frame to draw on (modified in place)
            results: MediaPipe results object
        """
        try:
            if results and results.pose_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame, 
                    results.pose_landmarks, 
                    self.mp_pose.POSE_CONNECTIONS,
                    # Optional: Customize drawing style
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=2
                    ),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(0, 255, 255), thickness=2
                    )
                )
        
        except Exception as e:
            log_error(logger, "PoseDetector.draw_landmarks", e)
    
    def get_landmarks(self, results) -> Optional[list]:
        """
        Extract landmark list from results
        
        Args:
            results: MediaPipe results object
            
        Returns:
            List of landmarks or None if no pose detected
        """
        try:
            if results and results.pose_landmarks:
                return results.pose_landmarks.landmark
            return None
            
        except Exception as e:
            log_error(logger, "PoseDetector.get_landmarks", e)
            return None
    
    def has_pose(self, results) -> bool:
        """
        Check if pose was detected in results
        
        Args:
            results: MediaPipe results object
            
        Returns:
            True if pose detected, False otherwise
        """
        try:
            return results is not None and results.pose_landmarks is not None
        except:
            return False
    
    def get_landmark_by_name(self, landmarks, landmark_name: str):
        """
        Get specific landmark by name
        
        Args:
            landmarks: Landmark list from MediaPipe results
            landmark_name: Name of landmark (e.g., 'LEFT_HIP', 'RIGHT_HIP')
            
        Returns:
            Landmark object or None if not found
        """
        try:
            landmark_enum = getattr(self.mp_pose_landmarks, landmark_name.upper())
            if landmark_enum.value < len(landmarks):
                return landmarks[landmark_enum.value]
            return None
            
        except Exception as e:
            log_error(logger, "PoseDetector.get_landmark_by_name", e, 
                     f"landmark_name={landmark_name}")
            return None
    
    def get_hip_landmarks(self, landmarks) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Get left and right hip landmarks (commonly used for golf pose analysis)
        
        Args:
            landmarks: Landmark list from MediaPipe results
            
        Returns:
            Tuple of (left_hip, right_hip) landmarks
        """
        try:
            left_hip = self.get_landmark_by_name(landmarks, 'LEFT_HIP')
            right_hip = self.get_landmark_by_name(landmarks, 'RIGHT_HIP')
            return left_hip, right_hip
            
        except Exception as e:
            log_error(logger, "PoseDetector.get_hip_landmarks", e)
            return None, None
    
    def get_shoulder_landmarks(self, landmarks) -> Tuple[Optional[Any], Optional[Any]]:
        """
        Get left and right shoulder landmarks
        
        Args:
            landmarks: Landmark list from MediaPipe results
            
        Returns:
            Tuple of (left_shoulder, right_shoulder) landmarks
        """
        try:
            left_shoulder = self.get_landmark_by_name(landmarks, 'LEFT_SHOULDER')
            right_shoulder = self.get_landmark_by_name(landmarks, 'RIGHT_SHOULDER')
            return left_shoulder, right_shoulder
            
        except Exception as e:
            log_error(logger, "PoseDetector.get_shoulder_landmarks", e)
            return None, None
    
    def reinitialize(self) -> bool:
        """
        Reinitialize the pose detector (useful for recovery from errors)
        
        Returns:
            True if reinitialization successful
        """
        log_ai_event(logger, "Reinitializing pose detector")
        return self.initialize()
    
    def get_detector_info(self) -> dict:
        """
        Get information about the detector configuration
        
        Returns:
            Dictionary with detector information
        """
        return {
            'initialized': self.is_initialized,
            'config': MEDIAPIPE_CONFIG,
            'available_landmarks': len(self.mp_pose_landmarks) if self.mp_pose_landmarks else 0
        }
    
    def cleanup(self) -> None:
        """Clean up MediaPipe resources"""
        try:
            if self.pose is not None:
                self.pose.close()
                self.pose = None
            self.is_initialized = False
            log_ai_event(logger, "Pose detector cleaned up")
            
        except Exception as e:
            log_error(logger, "PoseDetector.cleanup", e)
    
    def __del__(self):
        """Cleanup when detector is destroyed"""
        self.cleanup()
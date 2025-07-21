"""
Golf pose classification module
Combines pose detection and ML model for golf swing analysis
"""

import math
import numpy as np
from typing import Tuple, Optional, Dict, Any
import mediapipe as mp

from config.settings import (
    CLASS_NAMES, VALID_TRANSITIONS, 
    P1_CONFIDENCE_THRESHOLD, P10_CONFIDENCE_THRESHOLD
)
from ai.pose_detector import PoseDetector
from ai.model_manager import ModelManager
from utils.logger import setup_logger, log_ai_event, log_error

logger = setup_logger(__name__)

class GolfPoseClassifier:
    """Golf-specific pose classification combining MediaPipe and TensorFlow"""
    
    def __init__(self):
        """Initialize the golf pose classifier"""
        self.pose_detector = PoseDetector()
        self.model_manager = ModelManager()
        
        # State tracking
        self.previous_class_index = -1
        self.last_predicted_class = "Unknown"
        self.last_p1_confidence = 0.0
        self.last_p10_confidence = 0.0
        
        # Validation settings
        self.validate_transitions = False  # Can be disabled for debugging
        
        log_ai_event(logger, "Golf pose classifier initialized")
    
    def initialize(self) -> bool:
        """
        Initialize all components
        
        Returns:
            True if initialization successful
        """
        try:
            # Initialize pose detector
            if not self.pose_detector.initialize():
                return False
            
            # Load model
            if not self.model_manager.load_model():
                return False
            
            log_ai_event(logger, "Golf pose classifier ready")
            return True
            
        except Exception as e:
            log_error(logger, "GolfPoseClassifier.initialize", e)
            return False
    
    def normalize_landmarks(self, landmarks) -> Optional[np.ndarray]:
        """
        Normalize landmarks using the same method as refer.txt
        
        Args:
            landmarks: Raw landmarks from MediaPipe
            
        Returns:
            Normalized pose landmarks array or None if error
        """
        if not landmarks or len(landmarks) < 33:
            return None
        
        try:
            # Get hip landmarks for center calculation (same as refer.txt)
            left_hip = landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP.value]
            right_hip = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP.value]
            
            # Calculate center point
            center_x = (left_hip.x + right_hip.x) / 2
            center_y = (left_hip.y + right_hip.y) / 2
            
            # Calculate max distance from center
            max_distance = max([
                math.sqrt((lm.x - center_x)**2 + (lm.y - center_y)**2) 
                for lm in landmarks
            ])
            
            # Prevent division by zero
            if max_distance == 0:
                max_distance = 1.0
            
            # Normalize landmarks (same as refer.txt)
            pose_landmarks = np.array([
                [(landmark.x - center_x) / max_distance,
                 (landmark.y - center_y) / max_distance,
                 landmark.z / max_distance,
                 landmark.visibility] 
                for landmark in landmarks
            ]).flatten()
            
            return pose_landmarks
            
        except Exception as e:
            log_error(logger, "GolfPoseClassifier.normalize_landmarks", e)
            return None
    
    def is_valid_transition(self, current_class_index: int, 
                           previous_class_index: int) -> bool:
        """
        Check if the transition between poses is valid (from refer.txt)
        
        Args:
            current_class_index: Index of current predicted class
            previous_class_index: Index of previous predicted class
            
        Returns:
            True if transition is valid
        """
        if not self.validate_transitions or previous_class_index == -1:
            return True
        
        if (current_class_index < 0 or current_class_index >= len(CLASS_NAMES) or
            previous_class_index < 0 or previous_class_index >= len(CLASS_NAMES)):
            return False
        
        try:
            previous_class = CLASS_NAMES[previous_class_index]
            current_class = CLASS_NAMES[current_class_index]
            
            return current_class in VALID_TRANSITIONS[previous_class]
            
        except (KeyError, IndexError) as e:
            log_error(logger, "GolfPoseClassifier.is_valid_transition", e,
                     f"current={current_class_index}, previous={previous_class_index}")
            return True  # Allow transition if validation fails
    
    def classify_pose(self, frame) -> Tuple[Any, str, float, float, float]:
        """
        Classify golf pose from frame
        
        Args:
            frame: Input frame (BGR format)
            
        Returns:
            Tuple of (processed_frame, predicted_class, confidence, p1_confidence, p10_confidence)
        """
        try:
            # Detect pose landmarks
            processed_frame, results = self.pose_detector.detect_pose(frame)
            
            # If no pose detected, return previous values
            if not self.pose_detector.has_pose(results):
                return processed_frame, self.last_predicted_class, 0.0, \
                       self.last_p1_confidence, self.last_p10_confidence
            
            # Draw landmarks on frame
            self.pose_detector.draw_landmarks(processed_frame, results)
            
            # Get landmarks and normalize
            landmarks = self.pose_detector.get_landmarks(results)
            normalized_landmarks = self.normalize_landmarks(landmarks)
            
            if normalized_landmarks is None:
                return processed_frame, self.last_predicted_class, 0.0, \
                       self.last_p1_confidence, self.last_p10_confidence
            
            # Prepare for model prediction
            landmarks_batch = np.expand_dims(normalized_landmarks, axis=0)
            
            # Run model prediction
            prediction = self.model_manager.predict(landmarks_batch)
            
            if prediction is None:
                return processed_frame, self.last_predicted_class, 0.0, \
                       self.last_p1_confidence, self.last_p10_confidence
            
            # Get predicted class
            predicted_class, confidence, current_class_index = \
                self.model_manager.get_class_prediction(prediction[0])
            
            # Validate transition if enabled
            if self.validate_transitions and not self.is_valid_transition(
                current_class_index, self.previous_class_index):
                # Invalid transition, return previous prediction
                return processed_frame, self.last_predicted_class, 0.0, \
                       self.last_p1_confidence, self.last_p10_confidence
            
            # Update state
            self.previous_class_index = current_class_index
            self.last_predicted_class = predicted_class
            
            # Get P1 and P10 specific confidences
            p1_confidence = self.model_manager.get_specific_class_confidence(
                prediction[0], 'P1')
            p10_confidence = self.model_manager.get_specific_class_confidence(
                prediction[0], 'P10')
            
            self.last_p1_confidence = p1_confidence
            self.last_p10_confidence = p10_confidence
            
            # Log high-confidence predictions for debugging
            if confidence > 0.7:
                logger.debug(f"🔍 Pose: {predicted_class} (conf: {confidence:.3f}) - "
                           f"P1: {p1_confidence:.3f}, P10: {p10_confidence:.3f}")
            
            return processed_frame, predicted_class, confidence, p1_confidence, p10_confidence
            
        except Exception as e:
            log_error(logger, "GolfPoseClassifier.classify_pose", e)
            return frame, self.last_predicted_class, 0.0, \
                   self.last_p1_confidence, self.last_p10_confidence
    
    def is_p1_detected(self, p1_confidence: float = None) -> bool:
        """
        Check if P1 pose is detected with sufficient confidence
        
        Args:
            p1_confidence: P1 confidence score (uses last known if None)
            
        Returns:
            True if P1 detected with sufficient confidence
        """
        confidence = p1_confidence if p1_confidence is not None else self.last_p1_confidence
        return confidence > P1_CONFIDENCE_THRESHOLD
    
    def is_p10_detected(self, p10_confidence: float = None) -> bool:
        """
        Check if P10 pose is detected with sufficient confidence
        
        Args:
            p10_confidence: P10 confidence score (uses last known if None)
            
        Returns:
            True if P10 detected with sufficient confidence
        """
        confidence = p10_confidence if p10_confidence is not None else self.last_p10_confidence
        return confidence > P10_CONFIDENCE_THRESHOLD
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        Get current classifier state
        
        Returns:
            Dictionary with current state information
        """
        return {
            'predicted_class': self.last_predicted_class,
            'p1_confidence': self.last_p1_confidence,
            'p10_confidence': self.last_p10_confidence,
            'previous_class_index': self.previous_class_index,
            'p1_detected': self.is_p1_detected(),
            'p10_detected': self.is_p10_detected(),
            'transition_validation_enabled': self.validate_transitions,
            'pose_detector_initialized': self.pose_detector.is_initialized,
            'model_loaded': self.model_manager.is_loaded
        }
    
    def reset_state(self) -> None:
        """Reset classifier state"""
        self.previous_class_index = -1
        self.last_predicted_class = "Unknown"
        self.last_p1_confidence = 0.0
        self.last_p10_confidence = 0.0
        log_ai_event(logger, "Classifier state reset")
    
    def enable_transition_validation(self, enabled: bool = True) -> None:
        """
        Enable or disable transition validation
        
        Args:
            enabled: Whether to validate pose transitions
        """
        self.validate_transitions = enabled
        log_ai_event(logger, f"Transition validation {'enabled' if enabled else 'disabled'}")
    
    def test_with_current_frame(self, frame) -> Dict[str, Any]:
        """
        Test classifier with current frame and return detailed results
        
        Args:
            frame: Input frame for testing
            
        Returns:
            Dictionary with detailed test results
        """
        try:
            # Process frame
            processed_frame, results = self.pose_detector.detect_pose(frame)
            
            if not self.pose_detector.has_pose(results):
                return {'error': 'No pose detected in current frame'}
            
            # Get and normalize landmarks
            landmarks = self.pose_detector.get_landmarks(results)
            normalized_landmarks = self.normalize_landmarks(landmarks)
            
            if normalized_landmarks is None:
                return {'error': 'Failed to normalize landmarks'}
            
            # Run prediction
            landmarks_batch = np.expand_dims(normalized_landmarks, axis=0)
            prediction = self.model_manager.predict(landmarks_batch)
            
            if prediction is None:
                return {'error': 'Model prediction failed'}
            
            # Get detailed results
            predicted_class, confidence, class_index = \
                self.model_manager.get_class_prediction(prediction[0])
            all_confidences = self.model_manager.get_all_class_confidences(prediction[0])
            
            return {
                'success': True,
                'normalized_landmarks_shape': normalized_landmarks.shape,
                'normalized_landmarks_sample': normalized_landmarks[:20].tolist(),
                'raw_prediction': prediction[0].tolist(),
                'predicted_class': predicted_class,
                'predicted_confidence': confidence,
                'predicted_index': class_index,
                'all_class_confidences': all_confidences,
                'transition_valid': self.is_valid_transition(class_index, self.previous_class_index)
            }
            
        except Exception as e:
            log_error(logger, "GolfPoseClassifier.test_with_current_frame", e)
            return {'error': str(e)}
    
    def get_debug_info(self) -> Dict[str, Any]:
        """
        Get comprehensive debug information
        
        Returns:
            Dictionary with debug information
        """
        return {
            'classifier_state': self.get_current_state(),
            'pose_detector_info': self.pose_detector.get_detector_info(),
            'model_info': self.model_manager.get_model_info(),
            'class_names': CLASS_NAMES,
            'valid_transitions': VALID_TRANSITIONS,
            'thresholds': {
                'p1_confidence': P1_CONFIDENCE_THRESHOLD,
                'p10_confidence': P10_CONFIDENCE_THRESHOLD
            }
        }
    
    def reload_model(self, force_download: bool = False) -> bool:
        """
        Reload the ML model
        
        Args:
            force_download: Whether to re-download from GCS
            
        Returns:
            True if reload successful
        """
        self.reset_state()
        return self.model_manager.reload_model(force_download)
    
    def cleanup(self) -> None:
        """Clean up all resources"""
        try:
            self.pose_detector.cleanup()
            self.model_manager.cleanup()
            self.reset_state()
            log_ai_event(logger, "Golf pose classifier cleaned up")
            
        except Exception as e:
            log_error(logger, "GolfPoseClassifier.cleanup", e)
    
    def __del__(self):
        """Cleanup when classifier is destroyed"""
        self.cleanup()
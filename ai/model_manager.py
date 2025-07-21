"""
TensorFlow model management for golf pose classification
Handles model downloading, loading, and inference
"""

import os
import numpy as np
import tensorflow as tf
from typing import Optional, Tuple, Dict, Any
from google.cloud import storage

from config.settings import (
    BUCKET_NAME, MODEL_DIR, GCS_MODEL_NAME, 
    get_model_path, CLASS_NAMES
)
from utils.logger import setup_logger, log_ai_event, log_error, log_success

logger = setup_logger(__name__)

class ModelManager:
    """TensorFlow model manager with download and caching capabilities"""
    
    def __init__(self):
        """Initialize model manager"""
        self.classification_model: Optional[tf.keras.Model] = None
        self.is_loaded = False
        self.model_path = get_model_path()
        self.class_names = CLASS_NAMES
        
        # Ensure model directory exists
        os.makedirs(MODEL_DIR, exist_ok=True)
    
    def download_model_from_gcs(self, force_download: bool = False) -> bool:
        """
        Download model from Google Cloud Storage
        
        Args:
            force_download: If True, download even if model exists locally
            
        Returns:
            True if download successful, False otherwise
        """
        try:
            # Check if model already exists and skip download unless forced
            if os.path.exists(self.model_path) and not force_download:
                log_ai_event(logger, "Model already exists locally", self.model_path)
                return True
            
            log_ai_event(logger, "Downloading model from GCS", 
                        f"gs://{BUCKET_NAME}/{GCS_MODEL_NAME}")
            
            # Download from GCS
            client = storage.Client()
            bucket = client.bucket(BUCKET_NAME)
            blob = bucket.blob(GCS_MODEL_NAME)
            
            if not blob.exists():
                raise FileNotFoundError(f"Model not found in GCS: gs://{BUCKET_NAME}/{GCS_MODEL_NAME}")
            
            blob.download_to_filename(self.model_path)
            
            # Verify file was downloaded
            if os.path.exists(self.model_path):
                file_size_mb = os.path.getsize(self.model_path) / (1024 * 1024)
                log_success(logger, "Model downloaded successfully", 
                           f"{file_size_mb:.1f}MB → {self.model_path}")
                return True
            else:
                raise FileNotFoundError("Model file not found after download")
                
        except Exception as e:
            log_error(logger, "ModelManager.download_model_from_gcs", e)
            return False
    
    def load_model(self) -> bool:
        """
        Load the TensorFlow model
        
        Returns:
            True if loading successful, False otherwise
        """
        try:
            if not os.path.exists(self.model_path):
                log_ai_event(logger, "Model file not found, attempting download")
                if not self.download_model_from_gcs():
                    return False
            
            log_ai_event(logger, "Loading TensorFlow model", self.model_path)
            
            # Load the model
            self.classification_model = tf.keras.models.load_model(self.model_path)
            
            self.is_loaded = True
            
            # Log model information
            input_shape = self.classification_model.input_shape
            output_shape = self.classification_model.output_shape
            
            log_success(logger, "Model loaded successfully")
            log_ai_event(logger, "Model details", 
                        f"Input: {input_shape}, Output: {output_shape}")
            
            return True
            
        except Exception as e:
            log_error(logger, "ModelManager.load_model", e)
            self.is_loaded = False
            self.classification_model = None
            return False
    
    def predict(self, landmarks_batch: np.ndarray) -> Optional[np.ndarray]:
        """
        Run inference on normalized landmarks
        
        Args:
            landmarks_batch: Preprocessed landmarks array with batch dimension
            
        Returns:
            Prediction array or None if error
        """
        if not self.is_loaded or self.classification_model is None:
            log_error(logger, "ModelManager.predict", 
                     Exception("Model not loaded"), "Call load_model() first")
            return None
        
        try:
            # Validate input shape
            expected_shape = self.classification_model.input_shape
            if landmarks_batch.shape[1:] != expected_shape[1:]:
                raise ValueError(f"Input shape mismatch: expected {expected_shape}, "
                               f"got {landmarks_batch.shape}")
            
            # Run prediction
            prediction = self.classification_model.predict(landmarks_batch, verbose=0)
            
            return prediction
            
        except Exception as e:
            log_error(logger, "ModelManager.predict", e, 
                     f"input_shape={landmarks_batch.shape}")
            return None
    
    def get_class_prediction(self, prediction: np.ndarray) -> Tuple[str, float, int]:
        """
        Convert raw prediction to class name and confidence
        
        Args:
            prediction: Raw prediction array from model
            
        Returns:
            Tuple of (class_name, confidence, class_index)
        """
        try:
            class_index = np.argmax(prediction)
            confidence = float(prediction[class_index])
            
            if class_index < len(self.class_names):
                class_name = self.class_names[class_index]
            else:
                class_name = f"Unknown_Index_{class_index}"
            
            return class_name, confidence, class_index
            
        except Exception as e:
            log_error(logger, "ModelManager.get_class_prediction", e)
            return "Unknown", 0.0, -1
    
    def get_specific_class_confidence(self, prediction: np.ndarray, 
                                    class_name: str) -> float:
        """
        Get confidence for a specific class
        
        Args:
            prediction: Raw prediction array from model
            class_name: Name of class to get confidence for
            
        Returns:
            Confidence score for the specified class
        """
        try:
            if class_name in self.class_names:
                class_index = self.class_names.index(class_name)
                if class_index < len(prediction):
                    return float(prediction[class_index])
            return 0.0
            
        except Exception as e:
            log_error(logger, "ModelManager.get_specific_class_confidence", e, 
                     f"class_name={class_name}")
            return 0.0
    
    def get_all_class_confidences(self, prediction: np.ndarray) -> Dict[str, float]:
        """
        Get confidence scores for all classes
        
        Args:
            prediction: Raw prediction array from model
            
        Returns:
            Dictionary mapping class names to confidence scores
        """
        try:
            confidences = {}
            for i, class_name in enumerate(self.class_names):
                if i < len(prediction):
                    confidences[class_name] = float(prediction[i])
                else:
                    confidences[class_name] = 0.0
            
            return confidences
            
        except Exception as e:
            log_error(logger, "ModelManager.get_all_class_confidences", e)
            return {}
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get detailed model information for debugging
        
        Returns:
            Dictionary with model information
        """
        info = {
            'is_loaded': self.is_loaded,
            'model_path': self.model_path,
            'class_names': self.class_names,
            'num_classes': len(self.class_names)
        }
        
        if self.is_loaded and self.classification_model is not None:
            try:
                info.update({
                    'input_shape': str(self.classification_model.input_shape),
                    'output_shape': str(self.classification_model.output_shape),
                    'model_summary': self._get_model_summary()
                })
            except Exception as e:
                log_error(logger, "ModelManager.get_model_info", e)
                info['error'] = str(e)
        
        return info
    
    def _get_model_summary(self) -> list:
        """Get model layer information"""
        try:
            summary = []
            for i, layer in enumerate(self.classification_model.layers):
                layer_info = {
                    'layer': i,
                    'name': layer.name,
                    'type': type(layer).__name__
                }
                
                if hasattr(layer, 'output_shape'):
                    layer_info['output_shape'] = str(layer.output_shape)
                
                summary.append(layer_info)
            
            return summary
            
        except Exception as e:
            log_error(logger, "ModelManager._get_model_summary", e)
            return []
    
    def test_model_with_dummy_data(self) -> Dict[str, Any]:
        """
        Test model with dummy data for debugging
        
        Returns:
            Dictionary with test results
        """
        if not self.is_loaded or self.classification_model is None:
            return {'error': 'Model not loaded'}
        
        try:
            # Create dummy input matching expected shape
            input_shape = self.classification_model.input_shape
            dummy_input = np.random.rand(1, input_shape[1])
            
            # Run prediction
            dummy_prediction = self.predict(dummy_input)
            
            if dummy_prediction is not None:
                class_name, confidence, class_index = self.get_class_prediction(dummy_prediction[0])
                all_confidences = self.get_all_class_confidences(dummy_prediction[0])
                
                return {
                    'success': True,
                    'input_shape': dummy_input.shape,
                    'output_shape': dummy_prediction.shape,
                    'predicted_class': class_name,
                    'predicted_confidence': confidence,
                    'predicted_index': class_index,
                    'all_confidences': all_confidences
                }
            else:
                return {'error': 'Prediction failed'}
                
        except Exception as e:
            log_error(logger, "ModelManager.test_model_with_dummy_data", e)
            return {'error': str(e)}
    
    def reload_model(self, force_download: bool = False) -> bool:
        """
        Reload the model (optionally re-downloading from GCS)
        
        Args:
            force_download: If True, re-download model from GCS
            
        Returns:
            True if reload successful
        """
        log_ai_event(logger, "Reloading model", f"force_download={force_download}")
        
        # Cleanup current model
        if self.classification_model is not None:
            del self.classification_model
            self.classification_model = None
        
        self.is_loaded = False
        
        # Re-download if requested
        if force_download:
            if not self.download_model_from_gcs(force_download=True):
                return False
        
        # Load model
        return self.load_model()
    
    def cleanup(self) -> None:
        """Clean up model resources"""
        try:
            if self.classification_model is not None:
                del self.classification_model
                self.classification_model = None
            
            self.is_loaded = False
            log_ai_event(logger, "Model manager cleaned up")
            
        except Exception as e:
            log_error(logger, "ModelManager.cleanup", e)
    
    def __del__(self):
        """Cleanup when model manager is destroyed"""
        self.cleanup()
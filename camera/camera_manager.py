"""
Camera management module for Raspberry Pi camera operations
Handles camera initialization, streaming, and mode switching
"""

import cv2
import time
import numpy as np
from typing import Generator, Optional, Tuple
from picamera2 import Picamera2
from libcamera import Transform

from config.settings import (
    CAMERA_PREVIEW_SIZE, CAMERA_RECORDING_SIZE, CAMERA_FORMAT,
    FRAME_SKIP_INTERVAL, PROCESS_INTERVAL, JPEG_QUALITY
)
from utils.logger import setup_logger, log_camera_event, log_error, log_success
from utils.frame_pool import FramePool

logger = setup_logger(__name__)

class CameraManager:
    """Raspberry Pi camera manager with streaming and recording capabilities"""
    
    def __init__(self, frame_pool: Optional[FramePool] = None):
        """
        Initialize camera manager
        
        Args:
            frame_pool: Optional frame pool for memory optimization
        """
        self.picam2: Optional[Picamera2] = None
        self.is_initialized = False
        self.current_mode = "preview"  # "preview" or "recording"
        self.frame_pool = frame_pool
        
        # Performance tracking
        self.frame_skip_counter = 0
        self.last_process_time = time.time()
        
        # JPEG encoding settings
        self.jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        
        self.initialize_camera()
    
    def initialize_camera(self) -> bool:
        """
        Initialize the Raspberry Pi camera
        
        Returns:
            True if initialization successful
        """
        try:
            if self.picam2 is not None:
                self.cleanup()
            
            log_camera_event(logger, "Initializing camera")
            
            self.picam2 = Picamera2()
            
            # Configure for preview mode initially
            config = self.picam2.create_preview_configuration(
                main={"size": CAMERA_PREVIEW_SIZE, "format": CAMERA_FORMAT},
                transform=Transform(vflip=True)
            )
            
            self.picam2.configure(config)
            self.picam2.start()
            
            # Allow camera to warm up
            time.sleep(2)
            
            self.is_initialized = True
            self.current_mode = "preview"
            
            log_success(logger, "Camera initialized", 
                       f"mode: {self.current_mode}, size: {CAMERA_PREVIEW_SIZE}")
            return True
            
        except Exception as e:
            log_error(logger, "CameraManager.initialize_camera", e)
            self.is_initialized = False
            return False
    
    def switch_to_preview_mode(self) -> bool:
        """
        Switch camera to preview mode (lower resolution, optimized for streaming)
        
        Returns:
            True if switch successful
        """
        if not self.is_initialized or self.current_mode == "preview":
            return True
        
        try:
            log_camera_event(logger, "Switching to preview mode")
            
            self.picam2.stop()
            
            config = self.picam2.create_preview_configuration(
                main={"size": CAMERA_PREVIEW_SIZE, "format": CAMERA_FORMAT},
                transform=Transform(vflip=True)
            )
            
            self.picam2.configure(config)
            self.picam2.start()
            time.sleep(1.0)
            
            self.current_mode = "preview"
            
            log_success(logger, "Switched to preview mode", 
                       f"size: {CAMERA_PREVIEW_SIZE}")
            return True
            
        except Exception as e:
            log_error(logger, "CameraManager.switch_to_preview_mode", e)
            return False
    
    def switch_to_recording_mode(self) -> bool:
        """
        Switch camera to recording mode (higher resolution)
        
        Returns:
            True if switch successful
        """
        if not self.is_initialized or self.current_mode == "recording":
            return True
        
        try:
            log_camera_event(logger, "Switching to recording mode")
            
            self.picam2.stop()
            
            config = self.picam2.create_video_configuration(
                main={"size": CAMERA_RECORDING_SIZE, "format": CAMERA_FORMAT},
                transform=Transform(vflip=True) 
            )
            
            self.picam2.configure(config)
            self.picam2.start()
            time.sleep(1.0)
            
            self.current_mode = "recording"
            
            log_success(logger, "Switched to recording mode", 
                       f"size: {CAMERA_RECORDING_SIZE}")
            return True
            
        except Exception as e:
            log_error(logger, "CameraManager.switch_to_recording_mode", e)
            return False
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from camera
        
        Returns:
            Frame array or None if error
        """
        if not self.is_initialized or self.picam2 is None:
            return None
        
        try:
            frame = self.picam2.capture_array()
            return frame
            
        except Exception as e:
            log_error(logger, "CameraManager.capture_frame", e)
            return None
    
    def generate_stream_frames(self, process_callback=None) -> Generator[bytes, None, None]:
        """
        Generate frames for video streaming with optional processing
        
        Args:
            process_callback: Optional function to process frames 
                             Should return (processed_frame, should_skip_frame)
        
        Yields:
            JPEG frame bytes for streaming
        """
        if not self.is_initialized:
            log_error(logger, "CameraManager.generate_stream_frames", 
                     Exception("Camera not initialized"))
            return
        
        while True:
            try:
                time.sleep(0.03)  # ~30 FPS
                
                # Get frame from pool if available, otherwise use camera frame directly
                if self.frame_pool:
                    working_frame = self.frame_pool.get_frame()
                    camera_frame = self.capture_frame()
                    
                    if camera_frame is None:
                        if self.frame_pool:
                            self.frame_pool.return_frame(working_frame)
                        continue
                    
                    # Copy camera data to working frame efficiently
                    if working_frame.shape == camera_frame.shape:
                        np.copyto(working_frame, camera_frame)
                    else:
                        working_frame = camera_frame.copy()
                else:
                    working_frame = self.capture_frame()
                    if working_frame is None:
                        continue
                
                self.frame_skip_counter += 1
                current_time = time.time()
                
                # Process frame with callback if provided
                should_process = (self.frame_skip_counter % FRAME_SKIP_INTERVAL == 0 and 
                                (current_time - self.last_process_time) > PROCESS_INTERVAL)
                
                if process_callback and should_process:
                    try:
                        processed_result = process_callback(working_frame)
                        
                        # Handle different callback return formats
                        if isinstance(processed_result, tuple):
                            working_frame = processed_result[0]
                            # Additional return values can be handled by caller
                        else:
                            working_frame = processed_result
                        
                        self.last_process_time = current_time
                        
                    except Exception as e:
                        log_error(logger, "CameraManager.generate_stream_frames.process_callback", e)
                        # Continue with unprocessed frame
                
                # Encode frame to JPEG
                ret, buffer = cv2.imencode('.jpg', working_frame, self.jpeg_params)
                
                # Return frame to pool if using frame pooling
                if self.frame_pool:
                    self.frame_pool.return_frame(working_frame)
                
                if not ret:
                    continue
                
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
                
            except Exception as e:
                log_error(logger, "CameraManager.generate_stream_frames", e)
                
                # Make sure to return frame to pool even on error
                if self.frame_pool and 'working_frame' in locals():
                    self.frame_pool.return_frame(working_frame)
                
                time.sleep(0.1)
    
    def add_overlay_text(self, frame: np.ndarray, text: str, 
                        position: Tuple[int, int] = (10, 30),
                        color: Tuple[int, int, int] = (0, 255, 0),
                        thickness: int = 2) -> None:
        """
        Add text overlay to frame
        
        Args:
            frame: Frame to add text to (modified in place)
            text: Text to add
            position: (x, y) position for text
            color: BGR color tuple
            thickness: Text thickness
        """
        try:
            cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, color, thickness)
        except Exception as e:
            log_error(logger, "CameraManager.add_overlay_text", e)
    
    def add_recording_indicator(self, frame: np.ndarray) -> None:
        """
        Add recording indicator to frame
        
        Args:
            frame: Frame to add indicator to (modified in place)
        """
        try:
            # Add "RECORDING" text
            cv2.putText(frame, "RECORDING", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add red circle indicator
            cv2.circle(frame, (580, 30), 8, (0, 0, 255), -1)
            
        except Exception as e:
            log_error(logger, "CameraManager.add_recording_indicator", e)
    
    def get_camera_info(self) -> dict:
        """
        Get camera information and status
        
        Returns:
            Dictionary with camera information
        """
        info = {
            'initialized': self.is_initialized,
            'current_mode': self.current_mode,
            'preview_size': CAMERA_PREVIEW_SIZE,
            'recording_size': CAMERA_RECORDING_SIZE,
            'format': CAMERA_FORMAT,
            'jpeg_quality': JPEG_QUALITY
        }
        
        if self.is_initialized and self.picam2:
            try:
                # Add runtime camera information
                info.update({
                    'camera_model': str(self.picam2.camera_model) if hasattr(self.picam2, 'camera_model') else 'Unknown',
                    'frame_skip_counter': self.frame_skip_counter,
                    'using_frame_pool': self.frame_pool is not None
                })
                
                if self.frame_pool:
                    info['frame_pool_stats'] = self.frame_pool.get_efficiency_stats()
                    
            except Exception as e:
                log_error(logger, "CameraManager.get_camera_info", e)
                info['error'] = str(e)
        
        return info
    
    def reset_performance_counters(self) -> None:
        """Reset performance tracking counters"""
        self.frame_skip_counter = 0
        self.last_process_time = time.time()
        log_camera_event(logger, "Performance counters reset")
    
    def test_capture(self) -> dict:
        """
        Test camera capture functionality
        
        Returns:
            Dictionary with test results
        """
        if not self.is_initialized:
            return {'success': False, 'error': 'Camera not initialized'}
        
        try:
            start_time = time.time()
            frame = self.capture_frame()
            capture_time = time.time() - start_time
            
            if frame is not None:
                return {
                    'success': True,
                    'frame_shape': frame.shape,
                    'capture_time_ms': round(capture_time * 1000, 2),
                    'current_mode': self.current_mode
                }
            else:
                return {'success': False, 'error': 'Failed to capture frame'}
                
        except Exception as e:
            log_error(logger, "CameraManager.test_capture", e)
            return {'success': False, 'error': str(e)}
    
    def reinitialize(self) -> bool:
        """
        Reinitialize the camera (useful for recovery from errors)
        
        Returns:
            True if reinitialization successful
        """
        log_camera_event(logger, "Reinitializing camera")
        self.cleanup()
        return self.initialize_camera()
    
    def cleanup(self) -> None:
        """Clean up camera resources"""
        try:
            if self.picam2 is not None:
                self.picam2.stop()
                self.picam2.close()
                self.picam2 = None
            
            self.is_initialized = False
            self.current_mode = "preview"
            
            log_camera_event(logger, "Camera cleaned up")
            
        except Exception as e:
            log_error(logger, "CameraManager.cleanup", e)
    
    def __del__(self):
        """Cleanup when camera manager is destroyed"""
        self.cleanup()
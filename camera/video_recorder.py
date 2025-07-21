"""
Video recording module for golf swing capture
Handles recording operations with early stop capability
"""

import os
import time
import threading
import requests  # ADD THIS IMPORT
from datetime import datetime
from typing import Optional, Callable
from picamera2.encoders import H264Encoder
from picamera2.outputs import FfmpegOutput

from config.settings import (
    VIDEO_DIR, DEFAULT_RECORDING_DURATION, AUTO_RECORDING_DURATION,
    get_video_filename, get_gcs_blob_name, BUCKET_NAME
)
from camera.camera_manager import CameraManager
from storage.uploader import BackgroundUploader
from utils.logger import setup_logger, log_recording_event, log_error, log_success

logger = setup_logger(__name__)

DJANGO_SERVER_URL = "http://127.0.0.1:8000/"  # Update with your Django server
DJANGO_UPLOAD_ENDPOINT = f"{DJANGO_SERVER_URL}/home/upload_from_pi/"

class VideoRecorder:
    """Video recording manager with Django upload integration"""
    
    def __init__(self, camera_manager: CameraManager, 
                 uploader: BackgroundUploader):
        """
        Initialize video recorder
        
        Args:
            camera_manager: Camera manager instance
            uploader: Background uploader instance (kept for backward compatibility)
        """
        self.camera_manager = camera_manager
        self.uploader = uploader  # Keep for potential fallback
        
        # Recording state
        self.is_recording = False
        self.recording_thread: Optional[threading.Thread] = None
        self.current_video_path: Optional[str] = None
        self.current_destination_blob: Optional[str] = None
        self.stop_recording_early = False
        
        # Callbacks for state changes
        self.on_recording_started: Optional[Callable] = None
        self.on_recording_stopped: Optional[Callable] = None
        
        # Ensure video directory exists
        os.makedirs(VIDEO_DIR, exist_ok=True)
        
        log_recording_event(logger, "Video recorder initialized with Django integration")
    
    def start_recording(self, duration: int, user_id: str, assignee_id: str = None, auto_triggered: bool = False) -> bool:
        """
        Enhanced to accept both operator and assignee IDs
        """
        if self.is_recording:
            return False
        
        if not user_id:
            return False

        # Default assignee to user_id if not provided
        if not assignee_id:
            assignee_id = user_id

        try:
            self.is_recording = True
            self.stop_recording_early = False
            
            # Pass both IDs to the recording thread
            self.recording_thread = threading.Thread(
                target=self._record_video,
                args=(duration, user_id, assignee_id, auto_triggered) 
            )
            self.recording_thread.start()
            
            return True
        except Exception as e:
            self.is_recording = False
            return False

    def _upload_to_django(self, video_path: str, operator_id: str, assignee_id: str) -> bool:
        """
        NEW METHOD: Upload video to Django server instead of direct GCS
        
        Args:
            video_path: Local path to video file
            operator_id: Who recorded the video
            assignee_id: Who the video is for
            
        Returns:
            True if upload successful
        """
        try:
            log_recording_event(logger, "Starting Django upload", 
                              f"operator: {operator_id}, assignee: {assignee_id}")
            
            # Prepare file and data for upload
            with open(video_path, 'rb') as video_file:
                files = {'file': video_file}
                data = {
                    'operator_id': operator_id,
                    'assignee_id': assignee_id,
                    'video_type': 'face-on'  # Default type
                }
                
                # POST to Django
                response = requests.post(
                    DJANGO_UPLOAD_ENDPOINT,
                    files=files,
                    data=data,
                    timeout=60  # 60 second timeout
                )
            
            if response.status_code == 200:
                response_data = response.json()
                log_success(logger, "Django upload successful", 
                          f"response: {response_data.get('message', 'Success')}")
                
                # Clean up local file after successful upload
                try:
                    os.remove(video_path)
                    log_recording_event(logger, "Local file cleaned up", video_path)
                except Exception as cleanup_error:
                    log_error(logger, "Failed to clean up local file", cleanup_error)
                
                return True
            else:
                log_error(logger, "Django upload failed", 
                         Exception(f"HTTP {response.status_code}: {response.text}"))
                return False
                
        except requests.exceptions.Timeout:
            log_error(logger, "Django upload timeout", 
                     Exception("Upload request timed out after 60 seconds"))
            return False
        except requests.exceptions.ConnectionError:
            log_error(logger, "Django upload connection error", 
                     Exception(f"Could not connect to Django server at {DJANGO_UPLOAD_ENDPOINT}"))
            return False
        except Exception as e:
            log_error(logger, "Django upload error", e)
            return False

    def _fallback_to_gcs(self, video_path: str, video_filename: str) -> bool:
        """
        FALLBACK METHOD: Direct GCS upload if Django fails
        
        Args:
            video_path: Local path to video file
            video_filename: Name of video file
            
        Returns:
            True if upload successful
        """
        try:
            log_recording_event(logger, "Falling back to direct GCS upload")
            
            blob_name = get_gcs_blob_name(video_filename)
            upload_id = self.uploader.queue_upload(
                video_path, 
                BUCKET_NAME, 
                blob_name
            )
            
            log_recording_event(logger, "GCS fallback upload queued", f"ID: {upload_id}")
            return True
            
        except Exception as e:
            log_error(logger, "GCS fallback upload failed", e)
            return False
    
    def stop_recording_early_signal(self) -> None:
        """Signal to stop recording early (e.g., when P10 detected)"""
        if self.is_recording:
            self.stop_recording_early = True
            log_recording_event(logger, "Early stop signal sent")
    
    def wait_for_recording_completion(self, timeout: float = 30.0) -> bool:
        """
        Wait for current recording to complete
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if recording completed within timeout
        """
        if not self.is_recording or not self.recording_thread:
            return True
        
        try:
            self.recording_thread.join(timeout=timeout)
            return not self.recording_thread.is_alive()
            
        except Exception as e:
            log_error(logger, "VideoRecorder.wait_for_recording_completion", e)
            return False
    
    def _record_video(self, duration: int, user_id: str, assignee_id: str, auto_triggered: bool) -> None:
        """
        MODIFIED: Internal method that uploads to Django first, GCS as fallback
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create filename (keep same format)
        if user_id == assignee_id:
            # Same person recording for themselves
            video_filename = f"{user_id}_swing_{timestamp}.mp4"
        else:
            # Different people (coach recording for student)
            video_filename = f"{user_id}_{assignee_id}_swing_{timestamp}.mp4"
        
        self.current_video_path = os.path.join(VIDEO_DIR, video_filename)
        self.current_destination_blob = get_gcs_blob_name(video_filename)
        
        upload_success = False
        
        try:
            # Switch camera to recording mode
            if not self.camera_manager.switch_to_recording_mode():
                raise Exception("Failed to switch camera to recording mode")
            
            time.sleep(0.2)  # Brief pause after mode switch
            
            # Start recording
            encoder = H264Encoder(10000000)  # 10 Mbps bitrate
            video_output = FfmpegOutput(self.current_video_path)
            
            self.camera_manager.picam2.start_recording(encoder, output=video_output)
            
            log_recording_event(logger, "Recording started", 
                            f"file: {video_filename}, operator: {user_id}, assignee: {assignee_id}")
            
            # Record for specified duration or until early stop
            start_time = time.time()
            while time.time() - start_time < duration:
                time.sleep(0.1)  # Check every 100ms
                if self.stop_recording_early:
                    elapsed = time.time() - start_time
                    log_recording_event(logger, "Early stop detected", 
                                    f"recorded: {elapsed:.1f}s")
                    break
            
            # Stop recording
            self.camera_manager.picam2.stop_recording()
            
            # Log recording completion
            actual_duration = time.time() - start_time
            if self.stop_recording_early:
                log_success(logger, "Recording completed early", 
                        f"{actual_duration:.1f}s of {duration}s")
            else:
                log_success(logger, "Recording completed full duration", 
                        f"{actual_duration:.1f}s")
            
            # Switch back to preview mode
            if not self.camera_manager.switch_to_preview_mode():
                log_error(logger, "VideoRecorder._record_video", 
                        Exception("Failed to switch back to preview mode"))
            
            # MODIFIED: Try Django upload first, fallback to GCS
            upload_success = self._upload_to_django(
                self.current_video_path, 
                user_id,  # operator_id
                assignee_id  # assignee_id
            )
            
            if not upload_success:
                log_recording_event(logger, "Django upload failed, trying GCS fallback")
                upload_success = self._fallback_to_gcs(
                    self.current_video_path,
                    video_filename
                )
            
        except Exception as e:
            log_error(logger, "VideoRecorder._record_video", e)
            
            # Try to recover camera to preview mode
            try:
                self.camera_manager.switch_to_preview_mode()
                log_recording_event(logger, "Camera recovered to preview mode")
            except Exception as recovery_error:
                log_error(logger, "VideoRecorder._record_video.recovery", recovery_error)
        
        finally:
            self.is_recording = False
            
            # Call callback if set
            if self.on_recording_stopped:
                try:
                    self.on_recording_stopped(upload_success)
                except Exception as e:
                    log_error(logger, "VideoRecorder.on_recording_stopped callback", e)
    
    def get_recording_status(self) -> dict:
        """
        Get current recording status
        
        Returns:
            Dictionary with recording status information
        """
        status = {
            'is_recording': self.is_recording,
            'stop_early_signal': self.stop_recording_early,
            'current_video_path': self.current_video_path if not self.is_recording else None,
            'current_blob_name': self.current_destination_blob if not self.is_recording else None,
            'video_directory': VIDEO_DIR,
            'default_duration': DEFAULT_RECORDING_DURATION,
            'auto_duration': AUTO_RECORDING_DURATION
        }
        
        # Add thread status if recording
        if self.recording_thread:
            status['thread_alive'] = self.recording_thread.is_alive()
        
        # Add recent upload info
        if not self.is_recording and self.current_destination_blob:
            status['gcs_path'] = f"gs://{BUCKET_NAME}/{self.current_destination_blob}"
        
        return status
    
    def get_recent_recordings(self, limit: int = 5) -> list:
        """
        Get list of recent recording files
        
        Args:
            limit: Maximum number of files to return
            
        Returns:
            List of recent recording file information
        """
        try:
            recordings = []
            
            if not os.path.exists(VIDEO_DIR):
                return recordings
            
            # Get all .mp4 files in video directory
            files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
            
            # Sort by modification time (newest first)
            files.sort(key=lambda x: os.path.getmtime(os.path.join(VIDEO_DIR, x)), 
                      reverse=True)
            
            # Get info for recent files
            for filename in files[:limit]:
                filepath = os.path.join(VIDEO_DIR, filename)
                try:
                    stat = os.stat(filepath)
                    recordings.append({
                        'filename': filename,
                        'filepath': filepath,
                        'size_mb': round(stat.st_size / (1024 * 1024), 1),
                        'modified_time': stat.st_mtime,
                        'gcs_blob': get_gcs_blob_name(filename)
                    })
                except Exception as e:
                    log_error(logger, "VideoRecorder.get_recent_recordings.file_stat", e, 
                             f"file: {filename}")
            
            return recordings
            
        except Exception as e:
            log_error(logger, "VideoRecorder.get_recent_recordings", e)
            return []
    
    def cleanup_old_recordings(self, max_age_hours: int = 24) -> int:
        """
        Clean up old recording files
        
        Args:
            max_age_hours: Maximum age in hours for local recordings
            
        Returns:
            Number of files cleaned up
        """
        try:
            if not os.path.exists(VIDEO_DIR):
                return 0
            
            cutoff_time = time.time() - (max_age_hours * 3600)
            cleaned_count = 0
            
            for filename in os.listdir(VIDEO_DIR):
                if filename.endswith('.mp4'):
                    filepath = os.path.join(VIDEO_DIR, filename)
                    try:
                        if os.path.getmtime(filepath) < cutoff_time:
                            file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
                            os.remove(filepath)
                            cleaned_count += 1
                            log_recording_event(logger, "Old recording cleaned up", 
                                              f"{filename} ({file_size_mb:.1f}MB)")
                    except Exception as e:
                        log_error(logger, "VideoRecorder.cleanup_old_recordings.file", e, 
                                 f"file: {filename}")
            
            if cleaned_count > 0:
                log_success(logger, f"Cleaned up {cleaned_count} old recordings")
            
            return cleaned_count
            
        except Exception as e:
            log_error(logger, "VideoRecorder.cleanup_old_recordings", e)
            return 0
    
    def get_stats(self) -> dict:
        """
        Get video recorder statistics
        
        Returns:
            Dictionary with recorder statistics
        """
        try:
            stats = {
                'total_recordings': 0,
                'total_size_mb': 0,
                'video_directory': VIDEO_DIR,
                'directory_exists': os.path.exists(VIDEO_DIR)
            }
            
            if os.path.exists(VIDEO_DIR):
                mp4_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith('.mp4')]
                stats['total_recordings'] = len(mp4_files)
                
                total_size = 0
                for filename in mp4_files:
                    try:
                        filepath = os.path.join(VIDEO_DIR, filename)
                        total_size += os.path.getsize(filepath)
                    except:
                        pass
                
                stats['total_size_mb'] = round(total_size / (1024 * 1024), 1)
            
            return stats
            
        except Exception as e:
            log_error(logger, "VideoRecorder.get_stats", e)
            return {'error': str(e)}
    
    def force_stop_recording(self) -> bool:
        """
        Force stop current recording (emergency stop)
        
        Returns:
            True if stop was successful
        """
        if not self.is_recording:
            return True
        
        try:
            log_recording_event(logger, "Force stopping recording")
            
            # Signal early stop
            self.stop_recording_early = True
            
            # Wait for recording to stop
            if self.wait_for_recording_completion(timeout=5.0):
                log_success(logger, "Recording force stopped successfully")
                return True
            else:
                log_error(logger, "VideoRecorder.force_stop_recording", 
                         Exception("Recording did not stop within timeout"))
                return False
                
        except Exception as e:
            log_error(logger, "VideoRecorder.force_stop_recording", e)
            return False
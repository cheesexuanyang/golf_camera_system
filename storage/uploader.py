"""
Background upload system for Google Cloud Storage
Handles non-blocking file uploads with retry logic and status tracking
"""

import os
import time
import queue
import threading
from typing import Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor
from google.cloud import storage

from config.settings import UPLOAD_MAX_WORKERS, UPLOAD_MAX_RETRIES, UPLOAD_RETRY_DELAY
from utils.logger import setup_logger, log_upload_event, log_error, log_success

logger = setup_logger(__name__)

class BackgroundUploader:
    """Background upload system with retry logic and status tracking"""
    
    def __init__(self, max_workers: int = UPLOAD_MAX_WORKERS):
        """
        Initialize the background uploader
        
        Args:
            max_workers: Maximum number of concurrent upload threads
        """
        self.upload_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.running = True
        self.upload_status: Dict[str, Dict[str, Any]] = {}
        self.max_workers = max_workers
        
        # Start background worker
        self.worker_thread = threading.Thread(target=self._upload_worker, daemon=True)
        self.worker_thread.start()
        
        log_success(logger, "Background uploader started", 
                   f"max_workers={max_workers}")
    
    def _upload_worker(self) -> None:
        """Background worker that processes upload queue"""
        while self.running:
            try:
                task = self.upload_queue.get(timeout=1)
                if task:
                    upload_id, local_path, bucket_name, blob_name = task
                    self._do_upload(upload_id, local_path, bucket_name, blob_name)
                    self.upload_queue.task_done()
            
            except queue.Empty:
                continue
            except Exception as e:
                log_error(logger, "BackgroundUploader._upload_worker", e)
    
    def _do_upload(self, upload_id: str, local_path: str, 
                   bucket_name: str, blob_name: str) -> None:
        """
        Perform the actual upload with retry logic
        
        Args:
            upload_id: Unique identifier for this upload
            local_path: Local file path
            bucket_name: GCS bucket name
            blob_name: GCS blob name
        """
        log_upload_event(logger, f"Starting upload {upload_id}", 
                        f"local: {local_path} → gs://{bucket_name}/{blob_name}")
        
        self.upload_status[upload_id] = {
            'status': 'uploading',
            'start_time': time.time(),
            'local_path': local_path,
            'blob_name': blob_name,
            'bucket_name': bucket_name,
            'attempt': 0
        }
        
        for attempt in range(UPLOAD_MAX_RETRIES):
            try:
                self.upload_status[upload_id]['attempt'] = attempt + 1
                
                # Perform upload
                client = storage.Client()
                bucket = client.bucket(bucket_name)
                blob = bucket.blob(blob_name)
                
                log_upload_event(logger, f"Uploading {blob_name}", 
                               f"attempt {attempt + 1}/{UPLOAD_MAX_RETRIES}")
                
                blob.upload_from_filename(local_path)
                
                # Success!
                self.upload_status[upload_id].update({
                    'status': 'completed',
                    'end_time': time.time(),
                    'gcs_path': f"gs://{bucket_name}/{blob_name}"
                })
                
                upload_duration = time.time() - self.upload_status[upload_id]['start_time']
                log_success(logger, f"Upload completed in {upload_duration:.1f}s", 
                           f"{blob_name}")
                
                # Clean up local file
                self._cleanup_local_file(local_path)
                return  # Success - exit retry loop
                
            except Exception as e:
                log_error(logger, f"BackgroundUploader._do_upload (attempt {attempt + 1})", 
                         e, f"file: {local_path}")
                
                if attempt == UPLOAD_MAX_RETRIES - 1:
                    # Final attempt failed
                    self.upload_status[upload_id].update({
                        'status': 'failed',
                        'end_time': time.time(),
                        'error': str(e),
                        'final_attempt': attempt + 1
                    })
                    log_error(logger, "BackgroundUploader", 
                             Exception(f"Upload failed after {UPLOAD_MAX_RETRIES} attempts: {e}"),
                             f"file: {local_path}")
                else:
                    # Wait before retry with exponential backoff
                    wait_time = UPLOAD_RETRY_DELAY ** attempt
                    log_upload_event(logger, f"Retrying in {wait_time}s", 
                                   f"attempt {attempt + 1} failed")
                    time.sleep(wait_time)
    
    def _cleanup_local_file(self, local_path: str) -> None:
        """
        Clean up local file after successful upload
        
        Args:
            local_path: Path to local file to delete
        """
        try:
            if os.path.exists(local_path):
                file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
                os.remove(local_path)
                log_success(logger, f"Local file cleaned up", 
                           f"{local_path} ({file_size_mb:.1f}MB)")
            
        except Exception as e:
            log_error(logger, "BackgroundUploader._cleanup_local_file", e, 
                     f"file: {local_path}")
    
    def queue_upload(self, local_path: str, bucket_name: str, blob_name: str) -> str:
        """
        Queue a file for background upload
        
        Args:
            local_path: Path to local file
            bucket_name: GCS bucket name
            blob_name: GCS blob name
            
        Returns:
            Upload ID for tracking
        """
        upload_id = f"upload_{int(time.time() * 1000)}"
        
        # Validate file exists
        if not os.path.exists(local_path):
            error_msg = f"File not found: {local_path}"
            self.upload_status[upload_id] = {
                'status': 'failed',
                'error': error_msg,
                'queue_time': time.time()
            }
            log_error(logger, "BackgroundUploader.queue_upload", 
                     FileNotFoundError(error_msg))
            return upload_id
        
        # Add to queue
        self.upload_queue.put((upload_id, local_path, bucket_name, blob_name))
        
        file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
        self.upload_status[upload_id] = {
            'status': 'queued',
            'queue_time': time.time(),
            'local_path': local_path,
            'blob_name': blob_name,
            'bucket_name': bucket_name,
            'file_size_mb': round(file_size_mb, 1)
        }
        
        log_upload_event(logger, f"Queued for upload", 
                        f"{blob_name} ({file_size_mb:.1f}MB) - ID: {upload_id}")
        return upload_id
    
    def get_upload_info(self, upload_id: str) -> Dict[str, Any]:
        """
        Get status information for specific upload
        
        Args:
            upload_id: Upload identifier
            
        Returns:
            Upload status dictionary
        """
        return self.upload_status.get(upload_id, {'status': 'not_found'})
    
    def get_recent_uploads(self, limit: int = 5) -> Dict[str, Dict[str, Any]]:
        """
        Get recent upload history
        
        Args:
            limit: Maximum number of uploads to return
            
        Returns:
            Dictionary of recent uploads
        """
        recent = sorted(
            self.upload_status.items(), 
            key=lambda x: x[1].get('queue_time', 0), 
            reverse=True
        )
        return dict(recent[:limit])
    
    def get_queue_size(self) -> int:
        """Get current upload queue size"""
        return self.upload_queue.qsize()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive uploader statistics
        
        Returns:
            Dictionary with uploader statistics
        """
        completed = sum(1 for status in self.upload_status.values() 
                       if status.get('status') == 'completed')
        failed = sum(1 for status in self.upload_status.values() 
                    if status.get('status') == 'failed')
        in_progress = sum(1 for status in self.upload_status.values() 
                         if status.get('status') == 'uploading')
        queued = self.get_queue_size()
        
        return {
            'total_uploads': len(self.upload_status),
            'completed': completed,
            'failed': failed,
            'in_progress': in_progress,
            'queued': queued,
            'success_rate': round((completed / max(1, completed + failed)) * 100, 1),
            'queue_size': queued,
            'max_workers': self.max_workers,
            'worker_running': self.running
        }
    
    def clear_old_status(self, max_age_hours: int = 24) -> int:
        """
        Clear old upload status entries
        
        Args:
            max_age_hours: Maximum age in hours for status entries
            
        Returns:
            Number of entries cleared
        """
        cutoff_time = time.time() - (max_age_hours * 3600)
        cleared = 0
        
        # Create list of keys to remove (can't modify dict during iteration)
        to_remove = []
        for upload_id, status in self.upload_status.items():
            queue_time = status.get('queue_time', 0)
            end_time = status.get('end_time', 0)
            
            # Remove if both queued and completed/failed before cutoff
            if queue_time < cutoff_time and (end_time < cutoff_time or end_time == 0):
                to_remove.append(upload_id)
        
        # Remove old entries
        for upload_id in to_remove:
            del self.upload_status[upload_id]
            cleared += 1
        
        if cleared > 0:
            log_success(logger, f"Cleared {cleared} old upload status entries")
        
        return cleared
    
    def shutdown(self) -> None:
        """Gracefully shutdown the uploader"""
        log_upload_event(logger, "Shutting down background uploader")
        self.running = False
        
        # Wait for current uploads to complete (with timeout)
        try:
            self.upload_queue.join()  # Wait for queue to empty
            self.executor.shutdown(wait=True, timeout=30)
            self.worker_thread.join(timeout=5)
        except Exception as e:
            log_error(logger, "BackgroundUploader.shutdown", e)
        
        log_success(logger, "Background uploader shutdown complete")
    
    def __del__(self):
        """Cleanup when uploader is destroyed"""
        if hasattr(self, 'running') and self.running:
            self.shutdown()
"""
Frame pooling system for memory optimization
Provides efficient frame reuse to reduce memory allocation/deallocation overhead
"""

import queue
import numpy as np
from typing import Tuple, Dict, Any
from utils.logger import setup_logger, log_success, log_error

logger = setup_logger(__name__)

class FramePool:
    """Memory-efficient frame reuse system"""
    
    def __init__(self, pool_size: int = 8, frame_shape: Tuple[int, int, int] = (480, 640, 3)):
        """
        Initialize the frame pool
        
        Args:
            pool_size: Number of frames to pre-allocate
            frame_shape: Shape of frames (height, width, channels)
        """
        self.pool = queue.Queue(maxsize=pool_size)
        self.frame_shape = frame_shape
        self.pool_size = pool_size
        self.stats = {
            'total_created': 0,
            'total_reused': 0,
            'current_pool_size': 0
        }
        
        try:
            # Pre-allocate frames for the pool
            for _ in range(pool_size):
                frame = np.zeros(frame_shape, dtype=np.uint8)
                self.pool.put(frame)
            
            self.stats['current_pool_size'] = pool_size
            log_success(logger, f"Frame pool initialized", 
                       f"{pool_size} frames ({frame_shape})")
        
        except Exception as e:
            log_error(logger, "FramePool.__init__", e, 
                     f"pool_size={pool_size}, frame_shape={frame_shape}")
            raise
    
    def get_frame(self) -> np.ndarray:
        """
        Get a frame from pool or create new one if empty
        
        Returns:
            numpy array frame ready for use
        """
        try:
            frame = self.pool.get_nowait()
            self.stats['total_reused'] += 1
            self.stats['current_pool_size'] = self.pool.qsize()
            return frame
        
        except queue.Empty:
            # Pool empty, create new frame
            self.stats['total_created'] += 1
            return np.zeros(self.frame_shape, dtype=np.uint8)
    
    def return_frame(self, frame: np.ndarray) -> None:
        """
        Return frame to pool for reuse
        
        Args:
            frame: numpy array frame to return to pool
        """
        try:
            # Validate frame shape
            if frame.shape != self.frame_shape:
                logger.warning(f"Frame shape mismatch: expected {self.frame_shape}, "
                             f"got {frame.shape}. Frame not returned to pool.")
                return
            
            # Optional: Clear frame data (usually not needed for camera frames)
            # frame.fill(0)  # Uncomment if you want to clear frame data
            
            self.pool.put_nowait(frame)
            self.stats['current_pool_size'] = self.pool.qsize()
            
        except queue.Full:
            # Pool full, frame will be garbage collected naturally
            pass
    
    def get_efficiency_stats(self) -> Dict[str, Any]:
        """
        Get memory efficiency statistics
        
        Returns:
            Dictionary with efficiency metrics
        """
        total_frames = self.stats['total_reused'] + self.stats['total_created']
        efficiency = (self.stats['total_reused'] / max(1, total_frames)) * 100
        
        return {
            'frames_reused': self.stats['total_reused'],
            'frames_created': self.stats['total_created'],
            'total_frames_processed': total_frames,
            'reuse_efficiency_percent': round(efficiency, 1),
            'current_pool_size': self.stats['current_pool_size'],
            'max_pool_size': self.pool_size
        }
    
    def reset_stats(self) -> None:
        """Reset efficiency statistics"""
        self.stats = {
            'total_created': 0,
            'total_reused': 0,
            'current_pool_size': self.pool.qsize()
        }
        logger.info("📊 Frame pool statistics reset")
    
    def get_memory_usage_mb(self) -> float:
        """
        Estimate memory usage of the frame pool in MB
        
        Returns:
            Estimated memory usage in megabytes
        """
        bytes_per_frame = np.prod(self.frame_shape) * np.dtype(np.uint8).itemsize
        total_bytes = self.stats['current_pool_size'] * bytes_per_frame
        return total_bytes / (1024 * 1024)  # Convert to MB
    
    def resize_pool(self, new_size: int) -> None:
        """
        Resize the frame pool (add or remove frames)
        
        Args:
            new_size: New target pool size
        """
        current_size = self.pool.qsize()
        
        try:
            if new_size > current_size:
                # Add frames
                for _ in range(new_size - current_size):
                    if not self.pool.full():
                        frame = np.zeros(self.frame_shape, dtype=np.uint8)
                        self.pool.put_nowait(frame)
                
                log_success(logger, f"Frame pool expanded", 
                           f"{current_size} → {self.pool.qsize()} frames")
            
            elif new_size < current_size:
                # Remove frames
                removed = 0
                while self.pool.qsize() > new_size and not self.pool.empty():
                    try:
                        self.pool.get_nowait()
                        removed += 1
                    except queue.Empty:
                        break
                
                log_success(logger, f"Frame pool reduced", 
                           f"removed {removed} frames, now {self.pool.qsize()}")
            
            self.pool_size = new_size
            self.stats['current_pool_size'] = self.pool.qsize()
            
        except Exception as e:
            log_error(logger, "FramePool.resize_pool", e, 
                     f"new_size={new_size}, current_size={current_size}")
    
    def __del__(self):
        """Cleanup when frame pool is destroyed"""
        try:
            # Clear the pool
            while not self.pool.empty():
                self.pool.get_nowait()
        except:
            pass  # Ignore errors during cleanup
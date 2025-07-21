"""
Utility helper functions for the Golf Camera System
Common functionality shared across modules
"""

import os
import time
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union, List
from pathlib import Path

from utils.logger import setup_logger, log_error

logger = setup_logger(__name__)

def get_timestamp_string(format_type: str = "filename") -> str:
    """
    Get formatted timestamp string
    
    Args:
        format_type: Type of format ("filename", "display", "iso")
        
    Returns:
        Formatted timestamp string
    """
    now = datetime.now()
    
    if format_type == "filename":
        return now.strftime("%Y%m%d_%H%M%S")
    elif format_type == "display":
        return now.strftime("%Y-%m-%d %H:%M:%S")
    elif format_type == "iso":
        return now.isoformat()
    else:
        return now.strftime("%Y%m%d_%H%M%S")

def ensure_directory(directory_path: Union[str, Path]) -> bool:
    """
    Ensure directory exists, create if it doesn't
    
    Args:
        directory_path: Path to directory
        
    Returns:
        True if directory exists or was created successfully
    """
    try:
        Path(directory_path).mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        log_error(logger, "ensure_directory", e, f"path: {directory_path}")
        return False

def get_file_size_mb(file_path: Union[str, Path]) -> float:
    """
    Get file size in megabytes
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB, or 0 if file doesn't exist
    """
    try:
        return os.path.getsize(file_path) / (1024 * 1024)
    except (OSError, FileNotFoundError):
        return 0.0

def get_file_age_hours(file_path: Union[str, Path]) -> float:
    """
    Get file age in hours
    
    Args:
        file_path: Path to file
        
    Returns:
        Age in hours, or 0 if file doesn't exist
    """
    try:
        mtime = os.path.getmtime(file_path)
        age_seconds = time.time() - mtime
        return age_seconds / 3600
    except (OSError, FileNotFoundError):
        return 0.0

def cleanup_old_files(directory: Union[str, Path], 
                     max_age_hours: float = 24,
                     file_pattern: str = "*.mp4") -> int:
    """
    Clean up old files in directory
    
    Args:
        directory: Directory to clean
        max_age_hours: Maximum age in hours
        file_pattern: File pattern to match (glob pattern)
        
    Returns:
        Number of files deleted
    """
    try:
        directory_path = Path(directory)
        if not directory_path.exists():
            return 0
        
        cutoff_time = time.time() - (max_age_hours * 3600)
        deleted_count = 0
        
        for file_path in directory_path.glob(file_pattern):
            try:
                if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                    file_path.unlink()
                    deleted_count += 1
                    logger.info(f"🗑️ Deleted old file: {file_path.name}")
            except Exception as e:
                log_error(logger, "cleanup_old_files.file", e, f"file: {file_path}")
        
        return deleted_count
        
    except Exception as e:
        log_error(logger, "cleanup_old_files", e, f"directory: {directory}")
        return 0

def get_directory_stats(directory: Union[str, Path], 
                       file_pattern: str = "*") -> Dict[str, Any]:
    """
    Get statistics about files in directory
    
    Args:
        directory: Directory to analyze
        file_pattern: File pattern to match
        
    Returns:
        Dictionary with directory statistics
    """
    try:
        directory_path = Path(directory)
        if not directory_path.exists():
            return {"exists": False, "file_count": 0, "total_size_mb": 0}
        
        files = list(directory_path.glob(file_pattern))
        total_size = sum(f.stat().st_size for f in files if f.is_file())
        
        return {
            "exists": True,
            "file_count": len(files),
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "directory": str(directory_path),
            "pattern": file_pattern
        }
        
    except Exception as e:
        log_error(logger, "get_directory_stats", e, f"directory: {directory}")
        return {"exists": False, "error": str(e)}

def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def format_bytes(bytes_value: int) -> str:
    """
    Format bytes in human-readable format
    
    Args:
        bytes_value: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f}{unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f}PB"

def calculate_file_hash(file_path: Union[str, Path], 
                       algorithm: str = "md5") -> Optional[str]:
    """
    Calculate file hash
    
    Args:
        file_path: Path to file
        algorithm: Hash algorithm ("md5", "sha256", etc.)
        
    Returns:
        Hash string or None if error
    """
    try:
        hash_obj = hashlib.new(algorithm)
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_obj.update(chunk)
        
        return hash_obj.hexdigest()
        
    except Exception as e:
        log_error(logger, "calculate_file_hash", e, f"file: {file_path}")
        return None

def safe_json_loads(json_string: str, default: Any = None) -> Any:
    """
    Safely parse JSON string with default fallback
    
    Args:
        json_string: JSON string to parse
        default: Default value if parsing fails
        
    Returns:
        Parsed JSON or default value
    """
    try:
        return json.loads(json_string)
    except (json.JSONDecodeError, TypeError):
        return default

def safe_json_dumps(obj: Any, default: Any = "{}") -> str:
    """
    Safely serialize object to JSON string
    
    Args:
        obj: Object to serialize
        default: Default string if serialization fails
        
    Returns:
        JSON string or default value
    """
    try:
        return json.dumps(obj, indent=2, default=str)
    except (TypeError, ValueError):
        return default

def clamp(value: Union[int, float], 
          min_value: Union[int, float], 
          max_value: Union[int, float]) -> Union[int, float]:
    """
    Clamp value between min and max
    
    Args:
        value: Value to clamp
        min_value: Minimum value
        max_value: Maximum value
        
    Returns:
        Clamped value
    """
    return max(min_value, min(value, max_value))

def get_system_info() -> Dict[str, Any]:
    """
    Get basic system information
    
    Returns:
        Dictionary with system information
    """
    import platform
    import sys
    
    info = {
        "platform": platform.platform(),
        "python_version": sys.version,
        "architecture": platform.architecture(),
        "processor": platform.processor(),
        "hostname": platform.node(),
        "timestamp": get_timestamp_string("iso")
    }
    
    # Add memory info if psutil is available
    try:
        import psutil
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        info.update({
            "memory_total_mb": memory.total // (1024 * 1024),
            "memory_available_mb": memory.available // (1024 * 1024),
            "memory_percent": memory.percent,
            "disk_total_gb": disk.total // (1024 * 1024 * 1024),
            "disk_free_gb": disk.free // (1024 * 1024 * 1024),
            "disk_percent": (disk.used / disk.total) * 100
        })
    except ImportError:
        info["note"] = "Install psutil for detailed system info"
    
    return info

def validate_config_value(value: Any, 
                         expected_type: type, 
                         default_value: Any,
                         min_value: Optional[Union[int, float]] = None,
                         max_value: Optional[Union[int, float]] = None) -> Any:
    """
    Validate and sanitize configuration value
    
    Args:
        value: Value to validate
        expected_type: Expected type
        default_value: Default value if validation fails
        min_value: Minimum value for numeric types
        max_value: Maximum value for numeric types
        
    Returns:
        Validated value or default
    """
    try:
        # Type check
        if not isinstance(value, expected_type):
            if expected_type in (int, float):
                value = expected_type(value)
            else:
                return default_value
        
        # Range check for numeric types
        if expected_type in (int, float):
            if min_value is not None:
                value = max(value, min_value)
            if max_value is not None:
                value = min(value, max_value)
        
        return value
        
    except (ValueError, TypeError):
        return default_value

def create_backup_filename(original_path: Union[str, Path]) -> str:
    """
    Create backup filename with timestamp
    
    Args:
        original_path: Original file path
        
    Returns:
        Backup filename
    """
    path = Path(original_path)
    timestamp = get_timestamp_string("filename")
    return f"{path.stem}_backup_{timestamp}{path.suffix}"

def retry_operation(operation_func, max_retries: int = 3, 
                   delay: float = 1.0, backoff_factor: float = 2.0):
    """
    Retry operation with exponential backoff
    
    Args:
        operation_func: Function to retry
        max_retries: Maximum number of retries
        delay: Initial delay between retries
        backoff_factor: Delay multiplier for each retry
        
    Returns:
        Result of operation or raises last exception
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return operation_func()
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                time.sleep(delay * (backoff_factor ** attempt))
                logger.warning(f"Operation failed (attempt {attempt + 1}/{max_retries + 1}): {e}")
            else:
                logger.error(f"Operation failed after {max_retries + 1} attempts")
    
    raise last_exception

def get_available_disk_space(path: Union[str, Path]) -> Dict[str, float]:
    """
    Get available disk space for given path
    
    Args:
        path: Path to check
        
    Returns:
        Dictionary with disk space information in MB
    """
    try:
        import shutil
        total, used, free = shutil.disk_usage(path)
        
        return {
            "total_mb": total / (1024 * 1024),
            "used_mb": used / (1024 * 1024),
            "free_mb": free / (1024 * 1024),
            "used_percent": (used / total) * 100
        }
    except Exception as e:
        log_error(logger, "get_available_disk_space", e, f"path: {path}")
        return {"error": str(e)}

def is_port_available(port: int, host: str = "localhost") -> bool:
    """
    Check if network port is available
    
    Args:
        port: Port number to check
        host: Host to check
        
    Returns:
        True if port is available
    """
    import socket
    
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            return result != 0  # Port is available if connection fails
    except Exception:
        return False

def create_unique_filename(base_path: Union[str, Path], 
                          extension: str = ".mp4") -> str:
    """
    Create unique filename by appending number if file exists
    
    Args:
        base_path: Base path without extension
        extension: File extension
        
    Returns:
        Unique filename
    """
    counter = 1
    base = Path(base_path)
    filename = f"{base}{extension}"
    
    while Path(filename).exists():
        filename = f"{base}_{counter}{extension}"
        counter += 1
    
    return filename
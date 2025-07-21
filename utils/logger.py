"""
Centralized logging configuration for the Golf Camera System
"""

import logging
import sys
from config.settings import LOG_LEVEL, LOG_FORMAT

def setup_logger(name: str, level: str = LOG_LEVEL) -> logging.Logger:
    """
    Create a configured logger instance
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only add handler if logger doesn't have one (prevents duplicate logs)
    if not logger.handlers:
        # Create console handler
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(LOG_FORMAT)
        handler.setFormatter(formatter)
        
        # Add handler to logger
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, level.upper()))
        
        # Prevent propagation to avoid duplicate logs
        logger.propagate = False
    
    return logger

def log_performance(logger: logging.Logger, operation: str, duration: float):
    """
    Log performance metrics in a consistent format
    
    Args:
        logger: Logger instance
        operation: Description of the operation
        duration: Time taken in seconds
    """
    logger.info(f"⚡ Performance: {operation} took {duration:.3f}s")

def log_camera_event(logger: logging.Logger, event: str, details: str = ""):
    """
    Log camera-related events with consistent formatting
    
    Args:
        logger: Logger instance
        event: Camera event description
        details: Additional details
    """
    symbol = "📷"
    if details:
        logger.info(f"{symbol} Camera: {event} - {details}")
    else:
        logger.info(f"{symbol} Camera: {event}")

def log_ai_event(logger: logging.Logger, event: str, details: str = ""):
    """
    Log AI/ML related events with consistent formatting
    
    Args:
        logger: Logger instance
        event: AI event description
        details: Additional details
    """
    symbol = "🧠"
    if details:
        logger.info(f"{symbol} AI: {event} - {details}")
    else:
        logger.info(f"{symbol} AI: {event}")

def log_upload_event(logger: logging.Logger, event: str, details: str = ""):
    """
    Log upload-related events with consistent formatting
    
    Args:
        logger: Logger instance
        event: Upload event description
        details: Additional details
    """
    symbol = "☁️"
    if details:
        logger.info(f"{symbol} Upload: {event} - {details}")
    else:
        logger.info(f"{symbol} Upload: {event}")

def log_recording_event(logger: logging.Logger, event: str, details: str = ""):
    """
    Log recording-related events with consistent formatting
    
    Args:
        logger: Logger instance
        event: Recording event description
        details: Additional details
    """
    symbol = "🎥"
    if details:
        logger.info(f"{symbol} Recording: {event} - {details}")
    else:
        logger.info(f"{symbol} Recording: {event}")

def log_error(logger: logging.Logger, module: str, error: Exception, context: str = ""):
    """
    Log errors with consistent formatting and context
    
    Args:
        logger: Logger instance
        module: Module where error occurred
        error: Exception object
        context: Additional context about what was happening
    """
    error_msg = f"❌ Error in {module}: {str(error)}"
    if context:
        error_msg += f" (Context: {context})"
    
    logger.error(error_msg, exc_info=True)

def log_startup(logger: logging.Logger, component: str, status: str = "started"):
    """
    Log component startup with consistent formatting
    
    Args:
        logger: Logger instance
        component: Component name
        status: Status message
    """
    logger.info(f"🚀 {component} {status}")

def log_success(logger: logging.Logger, operation: str, details: str = ""):
    """
    Log successful operations with consistent formatting
    
    Args:
        logger: Logger instance
        operation: Operation description
        details: Additional details
    """
    symbol = "✅"
    if details:
        logger.info(f"{symbol} Success: {operation} - {details}")
    else:
        logger.info(f"{symbol} Success: {operation}")
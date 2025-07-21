# storage/__init__.py
"""
Storage module for Golf Camera System
Handles background uploads to Google Cloud Storage
"""

from .uploader import BackgroundUploader

__all__ = ['BackgroundUploader']
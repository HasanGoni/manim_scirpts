"""
Development configuration for faster Manim debugging
"""

# Development settings for ultra-fast rendering
DEV_CONFIG = {
    "quality": "low_quality",  # Options: low_quality, medium_quality, high_quality, ultra_quality
    "preview": True,
    "show_in_file_browser": False,
    "verbosity": "WARNING",  # Reduce log output
    "frame_rate": 15,  # Lower frame rate for faster rendering
    "pixel_height": 480,  # Lower resolution
    "pixel_width": 854,
    "max_files_cached": 50,  # Reduce cache size
}

# Production settings for final rendering
PROD_CONFIG = {
    "quality": "high_quality",
    "preview": True,
    "show_in_file_browser": False,
    "verbosity": "INFO",
    "frame_rate": 30,
    "pixel_height": 1080,
    "pixel_width": 1920,
    "max_files_cached": 200,
}

def get_dev_config():
    """Get development configuration"""
    return DEV_CONFIG

def get_prod_config():
    """Get production configuration"""
    return PROD_CONFIG

# Quick access functions
def is_dev_mode():
    """Check if we're in development mode"""
    return True  # Change this based on your environment

def get_current_config():
    """Get current configuration based on mode"""
    return get_dev_config() if is_dev_mode() else get_prod_config() 
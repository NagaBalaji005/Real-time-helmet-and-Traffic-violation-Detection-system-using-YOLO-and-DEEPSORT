#!/usr/bin/env python3
"""
FIXED Centralized Configuration
All hardcoded values moved here for easy management
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "output"
UPLOAD_DIR = BASE_DIR / "uploads"
LOGS_DIR = BASE_DIR / "logs"

# Create directories
for directory in [DATA_DIR, MODELS_DIR, OUTPUT_DIR, UPLOAD_DIR, LOGS_DIR]:
    directory.mkdir(exist_ok=True)

# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# Model paths (priority order)
MODEL_PATHS = [
    BASE_DIR / "runs/final_refined_model/weights/best.pt",
    BASE_DIR / "runs/final_model_refined/weights/best.pt",
    BASE_DIR / "runs/final_model/weights/best.pt",
    BASE_DIR / "yolov8s.pt"
]

# Detection parameters
DETECTION_CONFIG = {
    'model_size': 640,
    'confidence_threshold': 0.6,  # Standardized confidence threshold
    'nms_threshold': 0.45,
    'max_det': 300
}

# Classes
CLASSES = {
    0: 'no_helmet',
    1: 'mobile_usage',
    2: 'number_plate',
    3: 'helmet',
    4: 'triple_riding',
    5: 'traffic_violation',
    6: 'overspeed'
}

# ============================================================================
# OCR CONFIGURATION
# ============================================================================

OCR_CONFIG = {
    'use_gpu': False,
    'use_angle_cls': True,
    'lang': 'en',
    'det_db_thresh': 0.3,
    'det_db_box_thresh': 0.5,
    'rec_batch_num': 6,
    'min_confidence': 0.6,  # Minimum OCR confidence to accept
    'min_quality_score': 5.0  # Minimum quality score (out of 15)
}

# Valid Indian state codes
VALID_STATE_CODES = {
    'AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN', 
    'GA', 'GJ', 'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 
    'MH', 'MN', 'MP', 'MZ', 'NL', 'OD', 'OR', 'PB', 'PY', 'RJ', 
    'SK', 'TN', 'TR', 'TS', 'UK', 'UP', 'WB'
}

# ============================================================================
# VIOLATION CONFIGURATION
# ============================================================================

VIOLATION_CONFIG = {
    'violation_debounce_frames': 5,  # Minimum frames before confirming violation
    'max_distance_ratio': 0.4,  # Maximum distance for plate matching (40% of diagonal)
    'helmet_proximity_distance': 120,  # Maximum distance for helmet-to-rider matching (pixels)
    'min_violation_confidence': 0.5  # Minimum confidence for violation detection
}

# Violation types and display names
VIOLATION_TYPES = {
    'helmet': 'Helmet Worn',
    'no_helmet': 'No Helmet',
    'mobile_usage': 'Phone Usage',
    'phone_usage': 'Phone Usage',
    'triple_riding': 'Triple Riding',
    'traffic_violation': 'Traffic Rules Violation',
    'overspeed': 'Overspeed'
}

# Severity mapping
SEVERITY_MAP = {
    'helmet': 'info',
    'no_helmet': 'medium',
    'mobile_usage': 'high',
    'phone_usage': 'high',
    'triple_riding': 'medium',
    'traffic_violation': 'high',
    'overspeed': 'high'
}

# ============================================================================
# TRACKING CONFIGURATION
# ============================================================================

TRACKING_CONFIG = {
    'max_age': 30,  # Maximum frames to keep lost tracks
    'min_hits': 3,  # Minimum hits before track is confirmed
    'iou_threshold': 0.3,  # IoU threshold for matching
    'max_tracking_distance': 200  # Maximum distance for track matching (pixels)
}

# ============================================================================
# SPEED ESTIMATION CONFIGURATION
# ============================================================================

SPEED_CONFIG = {
    'pixels_per_meter': 50,  # Calibration factor (needs camera-specific calibration)
    'speed_limit': 50,  # Default speed limit (km/h)
    'speed_threshold': 60,  # Speed above which to trigger violation (km/h)
    'min_track_length': 5,  # Minimum track length for speed calculation
    'smoothing_window': 10,  # Window size for speed smoothing
    'enable_speed_detection': False  # DISABLED by default (needs calibration)
}

# ============================================================================
# FILE UPLOAD CONFIGURATION
# ============================================================================

UPLOAD_CONFIG = {
    'max_file_size': 100 * 1024 * 1024,  # 100MB
    'allowed_image_types': ['image/jpeg', 'image/png', 'image/jpg'],
    'allowed_video_types': ['video/mp4', 'video/avi', 'video/mov'],
    'image_timeout': 300,  # 5 minutes
    'video_timeout': 600   # 10 minutes
}

# ============================================================================
# DATABASE CONFIGURATION
# ============================================================================

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'traffic_violations'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres')
}

# ============================================================================
# API CONFIGURATION
# ============================================================================

API_CONFIG = {
    'host': '127.0.0.1',
    'port': 8081,
    'reload': False,
    'log_level': 'info',
    'cors_origins': ['*']
}

# ============================================================================
# CAMERA CONFIGURATION
# ============================================================================

CAMERA_CONFIG = {
    'default_camera_id': os.getenv('DEFAULT_CAMERA_ID', 'CAM001'),
    'default_location': os.getenv('DEFAULT_LOCATION', 'Main Street Intersection'),
    'fps': 30  # Default FPS for video processing
}

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

VISUALIZATION_CONFIG = {
    'colors': {
        'number_plate': (0, 255, 0),      # Green
        'helmet': (0, 255, 0),            # Green
        'no_helmet': (0, 165, 255),       # Orange
        'mobile_usage': (0, 165, 255),    # Orange
        'phone_usage': (0, 165, 255),     # Orange
        'triple_riding': (255, 0, 255),   # Magenta
        'traffic_violation': (255, 0, 255),  # Magenta
        'overspeed': (0, 255, 255)        # Yellow
    },
    'box_thickness': 2,
    'text_font': 'cv2.FONT_HERSHEY_SIMPLEX',
    'text_scale': 0.6,
    'text_thickness': 2
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'file': {
            'class': 'logging.FileHandler',
            'filename': str(LOGS_DIR / 'traffic_violation.log'),
            'formatter': 'standard',
            'level': 'INFO'
        },
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard',
            'level': 'INFO'
        }
    },
    'root': {
        'handlers': ['file', 'console'],
        'level': 'INFO'
    }
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_model_path():
    """Get the first available model path"""
    for path in MODEL_PATHS:
        if path.exists():
            return str(path)
    return "yolov8s.pt"


def get_output_path(filename: str, prefix: str = "annotated") -> Path:
    """Get output path for processed file"""
    return OUTPUT_DIR / f"{prefix}_{filename}"


def validate_config():
    """Validate configuration"""
    errors = []
    
    # Check critical directories
    if not MODELS_DIR.exists():
        errors.append(f"Models directory not found: {MODELS_DIR}")
    
    # Check model files
    model_exists = any(path.exists() for path in MODEL_PATHS)
    if not model_exists:
        errors.append("No model files found")
    
    # Check database connection (if required)
    if os.getenv('REQUIRE_DB', 'false').lower() == 'true':
        if not DB_CONFIG['host']:
            errors.append("Database host not configured")
    
    return errors


# Validate on import
config_errors = validate_config()
if config_errors:
    import warnings
    for error in config_errors:
        warnings.warn(f"Configuration warning: {error}")
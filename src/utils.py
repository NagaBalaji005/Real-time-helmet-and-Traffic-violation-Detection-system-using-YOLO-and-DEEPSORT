"""
Utility functions for Traffic Violation Detection System
"""

import cv2
import numpy as np
import math
from typing import List, Tuple, Dict, Any
import os
from datetime import datetime

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format
    Returns:
        IoU value between 0 and 1
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def preprocess_image(image: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> np.ndarray:
    """
    Preprocess image for model input
    Args:
        image: Input image
        target_size: Target size for resizing
    Returns:
        Preprocessed image
    """
    # Resize image
    resized = cv2.resize(image, target_size)
    
    # Normalize to [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    return normalized

def draw_detections(image: np.ndarray, detections: List[Dict], classes: Dict) -> np.ndarray:
    """
    Draw detection boxes and labels on image
    Args:
        image: Input image
        detections: List of detection dictionaries
        classes: Class mapping dictionary
    Returns:
        Image with drawn detections
    """
    result = image.copy()
    
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        conf = det['confidence']
        class_id = det['class_id']
        class_name = classes.get(class_id, f'class_{class_id}')
        
        # Draw bounding box
        color = get_class_color(class_id)
        cv2.rectangle(result, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Draw label
        label = f'{class_name}: {conf:.2f}'
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
        cv2.rectangle(result, (int(x1), int(y1) - label_size[1] - 10), 
                     (int(x1) + label_size[0], int(y1)), color, -1)
        cv2.putText(result, label, (int(x1), int(y1) - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    
    return result

def get_class_color(class_id: int) -> Tuple[int, int, int]:
    """
    Get color for class visualization
    Args:
        class_id: Class ID
    Returns:
        BGR color tuple
    """
    colors = [
        (0, 255, 0),    # Green for rider
        (255, 0, 0),    # Blue for helmet
        (0, 0, 255),    # Red for phone
        (255, 255, 0),  # Cyan for number_plate
        (255, 0, 255)   # Magenta for vehicle
    ]
    return colors[class_id % len(colors)]

def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points
    Args:
        point1: First point (x, y)
        point2: Second point (x, y)
    Returns:
        Distance in pixels
    """
    return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

def crop_bbox(image: np.ndarray, bbox: List[float], padding: int = 5) -> np.ndarray:
    """
    Crop image using bounding box with padding
    Args:
        image: Input image
        bbox: Bounding box [x1, y1, x2, y2]
        padding: Padding pixels around the bbox
    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    x1 = max(0, int(bbox[0]) - padding)
    y1 = max(0, int(bbox[1]) - padding)
    x2 = min(w, int(bbox[2]) + padding)
    y2 = min(h, int(bbox[3]) + padding)
    
    return image[y1:y2, x1:x2]

def enhance_plate_image(image: np.ndarray) -> np.ndarray:
    """
    Enhance number plate image for better OCR
    Args:
        image: Input plate image
    Returns:
        Enhanced image
    """
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    # Apply bilateral filter to reduce noise
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                 cv2.THRESH_BINARY, 11, 2)
    
    return thresh

def save_violation_image(image: np.ndarray, violation_type: str, plate_number: str = None) -> str:
    """
    Save violation evidence image
    Args:
        image: Image to save
        violation_type: Type of violation
        plate_number: Number plate text
    Returns:
        Path to saved image
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plate_suffix = f"_{plate_number}" if plate_number else ""
    filename = f"{violation_type}_{timestamp}{plate_suffix}.jpg"
    
    # Create output directory if it doesn't exist
    os.makedirs("output/violations", exist_ok=True)
    filepath = os.path.join("output/violations", filename)
    
    cv2.imwrite(filepath, image)
    return filepath

def validate_plate_format(plate_text: str) -> bool:
    """
    Validate number plate format (basic validation)
    Args:
        plate_text: Extracted plate text
    Returns:
        True if format is valid
    """
    if not plate_text:
        return False
    
    # Remove spaces and convert to uppercase
    plate = plate_text.replace(" ", "").upper()
    
    # Basic validation patterns (adjust for your country's format)
    patterns = [
        r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$',  # Format: KA01AB1234
        r'^[A-Z]{2}\d{1,2}[A-Z]{1,2}\d{4}$',  # Format: KA1A1234
        r'^\d{2}[A-Z]{2}\d{4}$',  # Format: 12AB1234
    ]
    
    import re
    for pattern in patterns:
        if re.match(pattern, plate):
            return True
    
    return False

def smooth_speed(speeds: List[float], window_size: int = 5) -> float:
    """
    Smooth speed values using moving average
    Args:
        speeds: List of speed values
        window_size: Window size for averaging
    Returns:
        Smoothed speed value
    """
    if not speeds:
        return 0.0
    
    if len(speeds) < window_size:
        return sum(speeds) / len(speeds)
    
    return sum(speeds[-window_size:]) / window_size

def create_output_directories():
    """
    Create necessary output directories
    """
    directories = [
        "output",
        "output/violations", 
        "output/processed_videos",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

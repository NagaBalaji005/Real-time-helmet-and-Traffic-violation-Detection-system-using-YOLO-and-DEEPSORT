#!/usr/bin/env python3
"""
FINAL Indian Number Plate OCR using PaddleOCR
- Processes cropped number plate regions for accurate extraction
- Better small text recognition than EasyOCR
- No destructive preprocessing
- Plate grammar enforcement
- Post-OCR correction & reconstruction
"""

import cv2
import re
import sys
import numpy as np
from paddleocr import PaddleOCR

# ---------------------------
# CONFIG
# ---------------------------
ALLOWED_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Prioritize 2-letter series (most common format)
PLATE_REGEX_2_LETTER = r"([A-Z]{2})([0-9]{2})([A-Z]{2})([0-9]{4})"  # AA BB CC DDDD
PLATE_REGEX_FLEXIBLE = r"([A-Z]{2})([0-9]{2})([A-Z]{1,2})([0-9]{4})"  # Fallback

VALID_STATES = {
    'AN', 'AP', 'AR', 'AS', 'BR', 'CG', 'CH', 'DD', 'DL', 'DN', 
    'GA', 'GJ', 'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD', 
    'MH', 'MN', 'MP', 'MZ', 'NL', 'OD', 'OR', 'PB', 'PY', 'RJ', 
    'SK', 'TN', 'TR', 'TS', 'UK', 'UP', 'WB'
}

CONFUSION_MAP = {
    'O': '0', 'Q': '0',
    'I': '1', 'L': '1',
    'Z': '2',
    'S': '5',
    'B': '8',
    'G': '6',
    'J': '3',
    'T': '7'
}

# ---------------------------
# PREPROCESSING (PADDLE SAFE)
# ---------------------------
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # PaddleOCR benefits from strong upscaling
    gray = cv2.resize(
        gray, None,
        fx=3.0, fy=3.0,
        interpolation=cv2.INTER_CUBIC
    )

    # Gentle denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # Contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    return gray

# ---------------------------
# PLATE RECONSTRUCTION WITH VALIDATION
# ---------------------------
def reconstruct_plate(text):
    """Reconstruct plate with strict validation - prioritize 2-letter series"""
    cleaned = re.sub(r'[^A-Z0-9]', '', text.upper())
    
    # Try 2-letter series first (most common)
    match = re.search(PLATE_REGEX_2_LETTER, cleaned)
    if match:
        state, rto, series, number = match.groups()
        # Validate: state must be valid, number must be 4 digits, no leading 0
        if state in VALID_STATES and rto.isdigit() and len(rto) == 2:
            if number.isdigit() and len(number) == 4 and number[0] != '0':
                # Check if series has invalid chars (O, I)
                if series.isalpha() and not any(c in 'OI' for c in series):
                    return f"{state} {rto} {series} {number}"
    
    # Try flexible regex (1-2 letter series)
    match = re.search(PLATE_REGEX_FLEXIBLE, cleaned)
    if match:
        state, rto, series, number = match.groups()
        if state in VALID_STATES and rto.isdigit() and len(rto) == 2:
            if number.isdigit() and len(number) == 4 and number[0] != '0':
                if series and series.isalpha() and not any(c in 'OI' for c in series):
                    return f"{state} {rto} {series} {number}"
                elif not series:
                    # No series - but check if raw text length suggests series should exist
                    if len(cleaned) >= 9:
                        return None  # Reject - missing series when expected
                    return f"{state} {rto} {number}"
    
    return None

# ---------------------------
# MAIN OCR - PROCESS CROPPED REGION
# ---------------------------
def extract_plate(image_path, bbox=None):
    """
    Extract plate from image or cropped region
    
    Args:
        image_path: Path to image file OR numpy array (if bbox provided, image_path should be array)
        bbox: Optional [x1, y1, x2, y2] bounding box to crop region
    """
    print("\n" + "="*80)
    print("FINAL INDIAN NUMBER PLATE OCR (PADDLEOCR) - CROPPED REGION")
    print("="*80)

    # Load image or use provided array
    if isinstance(image_path, np.ndarray):
        img = image_path
    else:
        img = cv2.imread(image_path)
        if img is None:
            print("❌ Cannot load image")
            return None
    
    # Crop to bbox if provided
    if bbox:
        x1, y1, x2, y2 = map(int, bbox)
        # Add padding
        padding = 5
        h, w = img.shape[:2]
        y1 = max(0, y1 - padding)
        y2 = min(h, y2 + padding)
        x1 = max(0, x1 - padding)
        x2 = min(w, x2 + padding)
        img = img[y1:y2, x1:x2]
        print(f"📦 Cropped region: {x2-x1}x{y2-y1} pixels")
    
    if img.size == 0 or img.shape[0] < 5 or img.shape[1] < 10:
        print("❌ Cropped region too small")
        return None

    processed = preprocess(img)
    if bbox:
        cv2.imwrite("debug_preprocessed_paddle.jpg", processed)
        print("Saved: debug_preprocessed_paddle.jpg")

    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='en',
        show_log=False
    )

    result = ocr.ocr(processed, cls=True)

    if not result or not result[0]:
        print("❌ No OCR detections")
        return None

    print("\nRAW OCR:")
    combined = ""
    for line in result[0]:
        text = line[1][0]
        conf = line[1][1]
        print(f"  '{text}' ({conf:.2%})")
        combined += text

    print(f"\nCombined Raw: {combined}")
    cleaned = re.sub(r'[^A-Z0-9]', '', combined.upper())
    print(f"Cleaned:      {cleaned}")

    plate = reconstruct_plate(cleaned)

    print("\n" + "-"*80)
    if plate:
        # Final validation: reject if raw length suggests missing series
        cleaned_len = len(cleaned)
        parts = plate.split()
        has_series = len(parts) == 4  # AA BB CC DDDD format
        if cleaned_len >= 9 and not has_series:
            print(f"⚠️ Rejected: Raw length {cleaned_len} suggests series should exist, but plate '{plate}' has no series")
            return None
        print(f"✅ FINAL PLATE: {plate}")
        return plate
    else:
        print("⚠️ Could not confidently reconstruct full plate")
        return None
    print("-"*80)

# ---------------------------
# ENTRY POINT
# ---------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python plate_ocr_paddle.py <image_path> [x1 y1 x2 y2]")
        print("  For cropped region: python plate_ocr_paddle.py image.jpg 100 200 300 250")
        sys.exit(1)

    image_path = sys.argv[1]
    
    # Check if bbox provided
    if len(sys.argv) == 6:
        bbox = [int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])]
        extract_plate(image_path, bbox=bbox)
    else:
        extract_plate(image_path)

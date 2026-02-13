"""
Main Pipeline for Traffic Violation Detection
"""

import cv2
import numpy as np
import argparse
import time
import uuid
from datetime import datetime
from typing import List, Dict, Optional
import os

from .detector import Detector
from .tracker import Tracker
from .speed import SpeedEstimator
from .ocr import PlateOCR
from .database import DatabaseManager
from .utils import draw_detections, crop_bbox, save_violation_image, create_output_directories
from .config import CLASSES, VIOLATION_CONFIG

class TrafficViolationPipeline:
    def __init__(self, model_path: str = None, device: str = 'auto'):
        """
        Initialize traffic violation detection pipeline
        Args:
            model_path: Path to trained YOLO model
            device: Device to run inference on
        """
        # Create output directories
        create_output_directories()
        
        # Initialize components
        self.detector = Detector(model_path, device)
        self.tracker = Tracker()
        self.speed_estimator = SpeedEstimator()
        self.ocr = PlateOCR()
        self.db_manager = DatabaseManager()
        
        # Pipeline state
        self.frame_count = 0
        self.violations = []
        self.active_tracks = set()
        
        # Violation tracking
        self.violation_history = {}
        self.debounce_frames = VIOLATION_CONFIG['violation_debounce_frames']
        
        # Camera and location info
        self.camera_id = os.getenv('DEFAULT_CAMERA_ID', 'CAM001')
        self.location = os.getenv('DEFAULT_LOCATION', 'Main Street Intersection')
        
        print("Traffic Violation Pipeline initialized")
    
    def process_video(self, video_path: str, output_path: str = None, 
                     show_display: bool = True, save_output: bool = True):
        """
        Process video file for traffic violations
        Args:
            video_path: Path to input video
            output_path: Path to save output video
            show_display: Whether to show real-time display
            save_output: Whether to save output video
        """
        # Open video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {width}x{height}, {fps} FPS, {total_frames} frames")
        
        # Update speed estimator with video FPS
        self.speed_estimator.fps = fps
        
        # Initialize video writer
        writer = None
        if save_output and output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Process frames
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame = self.process_frame(frame)
            
            # Save output
            if save_output and writer:
                writer.write(processed_frame)
            
            # Display
            if show_display:
                cv2.imshow('Traffic Violation Detection', processed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Progress update
            self.frame_count += 1
            if self.frame_count % 100 == 0:
                elapsed = time.time() - start_time
                fps_processed = self.frame_count / elapsed
                print(f"Processed {self.frame_count}/{total_frames} frames ({fps_processed:.1f} FPS)")
        
        # Cleanup
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        # Print results
        self.print_results()
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Process single frame
        Args:
            frame: Input frame
        Returns:
            Processed frame with annotations
        """
        # Run detection
        detections = self.detector.predict(frame)
        
        # Update tracker
        tracked_objects = self.tracker.update(detections, frame)
        
        # Update active tracks
        self.active_tracks = {track['track_id'] for track in tracked_objects}
        
        # Process violations
        frame_violations = self.detect_violations(frame, detections, tracked_objects)
        
        # Draw results
        annotated_frame = self.draw_results(frame, detections, tracked_objects, frame_violations)
        
        # Cleanup old tracks
        self.tracker.cleanup_old_tracks()
        self.speed_estimator.cleanup_old_tracks(list(self.active_tracks))
        
        return annotated_frame
    
    def detect_violations(self, frame: np.ndarray, detections: List[Dict], 
                         tracked_objects: List[Dict]) -> List[Dict]:
        """
        Detect violations in current frame
        Args:
            frame: Current frame
            detections: Detection results
            tracked_objects: Tracked objects
        Returns:
            List of violations detected
        """
        violations = []
        
        # Detect helmet violations (no_helmet class)
        helmet_violations = self.detector.detect_helmet_violation(detections)
        violations.extend(helmet_violations)
        
        # Detect mobile phone usage violations (mobile_usage class)
        phone_violations = self.detector.detect_phone_violation(detections)
        violations.extend(phone_violations)
        
        # Detect traffic signal violations (traffic_violation class)
        traffic_violations = self.detector.detect_traffic_violation(detections)
        violations.extend(traffic_violations)
        
        # Detect overspeed violations (overspeed class)
        overspeed_violations = self.detector.detect_overspeed_violation(detections)
        violations.extend(overspeed_violations)
        
        # Detect speed violations from tracking
        speed_violations = self.detect_speed_violations(tracked_objects)
        violations.extend(speed_violations)
        
        # Process violations with debouncing
        confirmed_violations = self.debounce_violations(violations)
        
        # Extract number plates for violations
        for violation in confirmed_violations:
            plate_text = self.extract_plate_for_violation(frame, violation, detections)
            if plate_text:
                violation['plate_number'] = plate_text
                self.log_violation(violation, frame)
        
        return confirmed_violations
    
    def detect_speed_violations(self, tracked_objects: List[Dict]) -> List[Dict]:
        """
        Detect speed violations
        Args:
            tracked_objects: List of tracked objects
        Returns:
            List of speed violations
        """
        violations = []
        vehicle_tracks = self.tracker.get_vehicle_tracks(tracked_objects)
        
        for track in vehicle_tracks:
            track_id = track['track_id']
            speed = self.speed_estimator.calculate_speed(track['history'], track_id)
            
            if speed is not None:
                speed_violation = self.speed_estimator.detect_speed_violation(speed)
                
                if speed_violation['is_violation']:
                    violation = {
                        'type': 'speed_violation',
                        'track_id': track_id,
                        'speed': speed,
                        'speed_limit': speed_violation['speed_limit'],
                        'excess_speed': speed_violation['excess_speed'],
                        'bbox': track['bbox'],
                        'confidence': track['confidence'],
                        'timestamp': None
                    }
                    violations.append(violation)
        
        return violations
    
    def debounce_violations(self, violations: List[Dict]) -> List[Dict]:
        """
        Debounce violations to avoid false positives
        Args:
            violations: List of violations
        Returns:
            Confirmed violations
        """
        confirmed_violations = []

        for violation in violations:
            # Create unique key for violation - prioritize plate number for better deduplication
            plate_number = violation.get('plate_number', 'unknown')

            if plate_number != 'unknown' and plate_number:
                # Use plate number + violation type for unique identification
                key = f"{plate_number}_{violation['type']}"
            elif violation['type'] == 'speed_violation':
                key = f"{violation['type']}_{violation['track_id']}"
            else:
                # Fallback to bbox position for violations without plate
                bbox = violation.get('rider_bbox', violation.get('phone_bbox', violation.get('bbox')))
                key = f"{violation['type']}_{int(bbox[0])}_{int(bbox[1])}"

            # Update violation history
            if key not in self.violation_history:
                self.violation_history[key] = {
                    'count': 0,
                    'first_seen': self.frame_count,
                    'violation': violation,
                    'last_logged_frame': 0
                }

            self.violation_history[key]['count'] += 1

            # Confirm violation if seen for enough frames and not recently logged
            frames_since_last_log = self.frame_count - self.violation_history[key]['last_logged_frame']
            if (self.violation_history[key]['count'] >= self.debounce_frames and
                frames_since_last_log > 300):  # Minimum 300 frames (10 seconds at 30fps) between same violations
                confirmed_violations.append(violation)
                self.violation_history[key]['last_logged_frame'] = self.frame_count

        return confirmed_violations
    
    def extract_plate_for_violation(self, frame: np.ndarray, violation: Dict,
                                   detections: List[Dict]) -> Optional[str]:
        """
        Extract number plate for a violation
        Args:
            frame: Current frame
            violation: Violation information
            detections: Detection results
        Returns:
            Number plate text or None
        """
        # Get number plate detections
        plate_detections = self.detector.get_detections_by_class(detections, 'number_plate')

        if not plate_detections:
            return None

        # Find plate closest to violation
        violation_bbox = violation.get('rider_bbox', violation.get('bbox'))
        best_plate = None
        best_distance = float('inf')

        for plate_det in plate_detections:
            plate_bbox = plate_det['bbox']
            distance = self._calculate_bbox_distance(violation_bbox, plate_bbox)

            if distance < best_distance:
                best_distance = distance
                best_plate = plate_det

        if best_plate and best_distance < 200:  # Distance threshold
            # Crop plate image
            plate_image = crop_bbox(frame, best_plate['bbox'])

            # Run OCR
            ocr_result = self.ocr.read_plate(plate_image)

            if ocr_result:
                plate_text = ocr_result['text']
                confidence = ocr_result.get('confidence', 0.0)

                # Only use plates with reasonable confidence
                if confidence < 0.6:
                    print(f"Low confidence plate detection ({confidence:.2f}): {plate_text} - skipping")
                    return None

                # Validate and format plate
                if ocr_result['is_valid']:
                    return self._format_plate_number(plate_text)
                else:
                    # Try to correct common OCR errors only for high confidence detections
                    if confidence >= 0.8:
                        corrected_plate = self._correct_plate_format(plate_text)
                        if corrected_plate:
                            return corrected_plate
                    else:
                        print(f"Plate format invalid and confidence too low ({confidence:.2f}): {plate_text} - skipping")
                        return None

        return None
    
    def _calculate_bbox_distance(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate distance between two bounding boxes
        Args:
            bbox1: First bounding box
            bbox2: Second bounding box
        Returns:
            Distance in pixels
        """
        center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
        center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)

        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

    def _format_plate_number(self, plate_text: str) -> str:
        """
        Format plate number to strict Indian standard format
        Args:
            plate_text: Raw plate text
        Returns:
            Formatted plate number or None if invalid
        """
        import re

        # Clean the text: remove spaces, convert to uppercase, keep only alphanumeric
        cleaned = re.sub(r'[^A-Z0-9]', '', plate_text.upper())

        # Strict Indian format validation:
        # Part 1: Two-letter state code (AA)
        # Part 2: Two-digit RTO number (BB)
        # Part 3: Single or multiple letters (CC...) - can be 1, 2, or 3 letters, excluding 'O' and 'I'
        # Part 4: Unique number between 1 and 9999 (DDDD)

        # Pattern: AA BB CC DDDD (e.g., TN 09 CF 7722)
        # This becomes: TN09CF7722
        if re.match(r'^[A-Z]{2}\d{2}[A-Z]{1,3}\d{1,4}$', cleaned):
            # Extract parts
            state_code = cleaned[:2]      # AA
            rto_number = cleaned[2:4]     # BB
            letters = cleaned[4:-4] if len(cleaned) > 6 else ''  # CC...
            numbers = cleaned[-4:]        # DDDD

            # Validate parts
            if len(state_code) == 2 and state_code.isalpha():
                if len(rto_number) == 2 and rto_number.isdigit():
                    # Letters part: 1-3 letters, no 'O' or 'I'
                    if 0 <= len(letters) <= 3 and not any(c in 'OI' for c in letters):
                        # Numbers part: 1-4 digits, not starting with 0 unless it's exactly 4 digits
                        if numbers.isdigit() and 1 <= len(numbers) <= 4:
                            if len(numbers) < 4 and numbers.startswith('0'):
                                return None  # Leading zero not allowed for shorter numbers
                            # Format as AA BB CC DDDD
                            formatted = f"{state_code} {rto_number}"
                            if letters:
                                formatted += f" {letters}"
                            formatted += f" {numbers.zfill(4)}"
                            return formatted

        return None  # Invalid format

    def _correct_plate_format(self, plate_text: str) -> Optional[str]:
        """
        Try to correct common OCR errors in plate format
        Args:
            plate_text: Raw plate text
        Returns:
            Corrected plate text or None
        """
        import re

        # Clean the text
        cleaned = re.sub(r'[^A-Z0-9]', '', plate_text.upper())

        # Only apply corrections if the text is close to a valid format
        # Check if it matches patterns with some tolerance for OCR errors

        patterns = [
            (r'^[A-Z]{2}[OIZSBG]\d[A-Z]{2}\d{4}$', 'KA01AB1234'),  # KA01AB1234 with OCR errors
            (r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$', 'KA01AB1234'),       # Already correct
            (r'^[A-Z]{2}\d{1}[A-Z]{2}\d{4}$', 'KA1AB1234'),        # KA1AB1234
            (r'^\d{2}[A-Z]{2}\d{4}$', '12AB1234'),                 # 12AB1234
        ]

        # First check if already valid
        for pattern, example in patterns:
            if re.match(pattern[0], cleaned):
                return cleaned

        # Apply conservative corrections only for specific cases
        corrections = {
            'O': '0',
            'I': '1',
            'Z': '2',
            'S': '5',
            'B': '8',
            'G': '6'
        }

        # Try corrections but only if result matches a valid pattern
        corrected = ''.join(corrections.get(c, c) for c in cleaned)

        # Validate the corrected result
        valid_patterns = [
            r'^[A-Z]{2}\d{2}[A-Z]{2}\d{4}$',  # KA01AB1234
            r'^[A-Z]{2}\d{1}[A-Z]{2}\d{4}$',  # KA1AB1234
            r'^\d{2}[A-Z]{2}\d{4}$',          # 12AB1234
        ]

        for pattern in valid_patterns:
            if re.match(pattern, corrected):
                return corrected

        # If no valid correction found, return None to indicate unreliable detection
        return None
    
    def log_violation(self, violation: Dict, frame: np.ndarray):
        """
        Log violation with evidence and store in database
        Args:
            violation: Violation information
            frame: Current frame
        """
        start_time = time.time()
        
        # Generate unique violation ID
        violation_id = f"VIO_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
        
        # Extract number plate if not already present
        if not violation.get('plate_number'):
            plate_number = self.extract_plate_for_violation(frame, violation, [])
            if plate_number:
                violation['plate_number'] = plate_number
        
        # Save violation image
        violation_image = self._create_violation_image(frame, violation)
        image_path = save_violation_image(
            violation_image, 
            violation['type'], 
            violation.get('plate_number')
        )
        
        # Determine severity based on violation type
        severity_map = {
            'helmet': 'medium',
            'phone': 'high',
            'speed': 'high' if violation.get('speed_excess', 0) > 20 else 'medium',
            'red_light': 'critical'
        }
        severity = severity_map.get(violation['type'], 'medium')

        # Prepare violation data for database
        violation_data = {
            'violation_type': violation['type'],
            'severity': severity,
            'description': self._generate_violation_description(violation),
            'number_plate': violation.get('plate_number', 'Unknown'),
            'vehicle_type': violation.get('vehicle_type', 'unknown'),
            'vehicle_color': violation.get('vehicle_color'),
            'speed_limit': 40,  # Fixed speed limit for all vehicles
            'actual_speed': violation.get('actual_speed'),
            'speed_unit': 'km/h',
            'location': self.location,
            'camera_id': self.camera_id,
            'image_path': image_path,
            'confidence_score': violation.get('confidence', 0.0),
            'timestamp': datetime.now()
        }
        
        # Store in database
        try:
            db_violation = self.db_manager.add_violation(violation_data)
            if db_violation:
                print(f"✓ Violation stored in DB: {violation['type']} - {violation.get('plate_number', 'Unknown')}")
            else:
                print(f"✗ Failed to store violation in DB: {violation['type']}")
        except Exception as e:
            print(f"✗ Database error: {e}")
        
        # Add to local list for display
        violation['image_path'] = image_path
        violation['violation_id'] = violation_id
        self.violations.append(violation)
        
        print(f"Violation logged: {violation['type']} - {violation.get('plate_number', 'Unknown')}")
    
    def _generate_violation_description(self, violation: Dict) -> str:
        """Generate detailed description of violation"""
        violation_type = violation['type']
        
        if violation_type == 'no_helmet':
            return f"Rider detected without helmet. Confidence: {violation.get('confidence', 0):.2f}"
        elif violation_type == 'mobile_usage':
            return f"Driver detected using mobile phone while driving. Confidence: {violation.get('confidence', 0):.2f}"
        elif violation_type == 'traffic_violation':
            return f"Traffic signal violation detected. Confidence: {violation.get('confidence', 0):.2f}"
        elif violation_type == 'overspeed':
            return f"Vehicle detected exceeding speed limit. Confidence: {violation.get('confidence', 0):.2f}"
        elif violation_type == 'speed_violation':
            speed_excess = violation.get('excess_speed', 0)
            return f"Vehicle exceeding speed limit by {speed_excess:.1f} km/h. Speed: {violation.get('speed', 0):.1f} km/h, Limit: {violation.get('speed_limit', 0)} km/h"
        else:
            return f"Traffic violation detected: {violation_type}. Confidence: {violation.get('confidence', 0):.2f}"
    
    def _create_violation_image(self, frame: np.ndarray, violation: Dict) -> np.ndarray:
        """
        Create violation evidence image
        Args:
            frame: Current frame
            violation: Violation information
        Returns:
            Violation image
        """
        # Crop around violation area
        bbox = violation.get('rider_bbox', violation.get('bbox'))
        if bbox:
            # Expand bbox for better context
            x1, y1, x2, y2 = bbox
            margin = 50
            x1 = max(0, int(x1) - margin)
            y1 = max(0, int(y1) - margin)
            x2 = min(frame.shape[1], int(x2) + margin)
            y2 = min(frame.shape[0], int(y2) + margin)
            
            return frame[y1:y2, x1:x2]
        
        return frame
    
    def draw_results(self, frame: np.ndarray, detections: List[Dict], 
                    tracked_objects: List[Dict], violations: List[Dict]) -> np.ndarray:
        """
        Draw detection and tracking results on frame
        Args:
            frame: Input frame
            detections: Detection results
            tracked_objects: Tracked objects
            violations: Violations
        Returns:
            Annotated frame
        """
        # Draw detections
        frame = draw_detections(frame, detections, CLASSES)
        
        # Draw tracking information
        for track in tracked_objects:
            bbox = track['bbox']
            track_id = track['track_id']
            class_name = track['class_name']
            
            # Draw track ID
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id}", (int(bbox[0]), int(bbox[1]) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw speed if available
            speed = self.speed_estimator.calculate_speed(track['history'], track_id)
            if speed is not None:
                speed_text = f"{speed:.1f} km/h"
                cv2.putText(frame, speed_text, (int(bbox[0]), int(bbox[3]) + 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
        
        # Draw violations
        for violation in violations:
            bbox = violation.get('rider_bbox', violation.get('bbox'))
            if bbox:
                # Draw violation box
                color = (0, 0, 255)  # Red for violations
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 3)
                
                # Draw violation text
                violation_text = f"{violation['type']}"
                if violation.get('plate_number'):
                    violation_text += f" - {violation['plate_number']}"
                
                cv2.putText(frame, violation_text, (int(bbox[0]), int(bbox[1]) - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Draw statistics
        self._draw_statistics(frame)
        
        return frame
    
    def _draw_statistics(self, frame: np.ndarray):
        """
        Draw processing statistics on frame
        Args:
            frame: Input frame
        """
        stats = [
            f"Frame: {self.frame_count}",
            f"Tracks: {len(self.active_tracks)}",
            f"Violations: {len(self.violations)}"
        ]
        
        y_offset = 30
        for stat in stats:
            cv2.putText(frame, stat, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
    
    def print_results(self):
        """
        Print processing results
        """
        print("\n" + "="*50)
        print("PROCESSING RESULTS")
        print("="*50)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Total violations detected: {len(self.violations)}")
        
        # Violation breakdown
        violation_types = {}
        for violation in self.violations:
            v_type = violation['type']
            violation_types[v_type] = violation_types.get(v_type, 0) + 1
        
        print("\nViolation breakdown:")
        for v_type, count in violation_types.items():
            print(f"  {v_type}: {count}")
        
        print("\nDetailed violations:")
        for i, violation in enumerate(self.violations, 1):
            print(f"{i}. {violation['type']} - {violation.get('plate_number', 'Unknown')} - {violation['timestamp']}")

def main():
    """
    Main function for command line usage
    """
    parser = argparse.ArgumentParser(description='Traffic Violation Detection Pipeline')
    parser.add_argument('--video', type=str, required=True, help='Path to input video')
    parser.add_argument('--model', type=str, default=None, help='Path to YOLO model')
    parser.add_argument('--output', type=str, default=None, help='Path to output video')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cpu/cuda/auto)')
    parser.add_argument('--no-display', action='store_true', help='Disable display')
    parser.add_argument('--no-save', action='store_true', help='Disable output saving')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = TrafficViolationPipeline(args.model, args.device)
    
    # Process video
    pipeline.process_video(
        args.video,
        args.output,
        show_display=not args.no_display,
        save_output=not args.no_save
    )

if __name__ == "__main__":
    main()
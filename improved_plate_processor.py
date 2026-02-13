#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Plate Processor - Video Processing with PaddleOCR
FULLY CORRECTED VERSION - Fixed Issues:
1. Video rotation issue
2. No helmet detection
3. Speed calculation
4. Overspeed detection using max speed
5. Proper violation mapping to plates
6. Terminal output for plates, speeds, and violations
7. ✅ Store actual plate numbers instead of track IDs in violations
8. ✅ Filter out false/low-confidence traffic violations
9. ✅ Properly log overspeed violations with plate numbers in JSON
10. ✅ Skip violations without valid plate numbers from being logged
"""

import cv2
import os
import sys
import json
from pathlib import Path
from datetime import datetime
import argparse
import numpy as np
from collections import defaultdict
import re
import time
import torch

# PaddleOCR can hang on startup doing online model-hoster checks when offline.
os.environ.setdefault("DISABLE_MODEL_SOURCE_CHECK", "True")

# Fix Windows console encoding for emoji characters
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# Fix for PyTorch 2.6 compatibility
try:
    from ultralytics.nn.tasks import torch_safe_load as original_torch_safe_load, temporary_modules
    from ultralytics.utils.checks import check_suffix
    from ultralytics.utils.downloads import attempt_download_asset
    
    def patched_torch_safe_load(weight):
        """Patched version of torch_safe_load that works with PyTorch 2.6"""
        check_suffix(file=weight, suffix='.pt')
        file = attempt_download_asset(weight)
        try:
            with temporary_modules({
                    'ultralytics.yolo.utils': 'ultralytics.utils',
                    'ultralytics.yolo.v8': 'ultralytics.models.yolo',
                    'ultralytics.yolo.data': 'ultralytics.data'}):
                return torch.load(file, map_location='cpu', weights_only=False), file
        except Exception as e:
            return original_torch_safe_load(weight)
    
    import ultralytics.nn.tasks
    ultralytics.nn.tasks.torch_safe_load = patched_torch_safe_load
    print("✅ Applied PyTorch 2.6 compatibility patch")
except Exception as e:
    print(f"⚠️  Could not apply PyTorch 2.6 patch: {e}")

from ultralytics import YOLO

# COCO class IDs for vehicles (pretrained YOLOv8) - used for tracking and speed
COCO_VEHICLE_IDS = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

# Import PaddleOCR-based OCR
try:
    from src.ocr import PlateOCR
    USE_OCR = True
    print("✅ Using PaddleOCR from src/ocr.py")
except ImportError:
    try:
        from ocr import PlateOCR
        USE_OCR = True
        print("✅ Using PaddleOCR from ocr.py")
    except ImportError:
        USE_OCR = False
        print("❌ PaddleOCR not available")


class VehicleTracker:
    """Track vehicles using bounding box IoU matching"""
    
    def __init__(self, iou_threshold=0.3, max_age=10):
        self.tracks = {}
        self.next_id = 1
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.frame_count = 0
    
    def update(self, detections, detected_plates=None):
        """Update tracks with new detections"""
        self.frame_count += 1
        current_frame_tracks = {}
        
        unmatched_detections = list(range(len(detections)))
        unmatched_tracks = list(self.tracks.keys())
        matched_pairs = []
        
        # Build cost matrix (IoU scores)
        if unmatched_tracks and unmatched_detections:
            cost_matrix = np.zeros((len(unmatched_tracks), len(unmatched_detections)))
            for i, track_id in enumerate(unmatched_tracks):
                track_bbox = self.tracks[track_id]['bbox']
                for j, det_idx in enumerate(unmatched_detections):
                    det_bbox = detections[det_idx]['bbox']
                    cost_matrix[i, j] = calculate_iou(track_bbox, det_bbox)
            
            # Simple greedy matching
            matched_pairs = []
            for _ in range(min(len(unmatched_tracks), len(unmatched_detections))):
                if cost_matrix.size == 0:
                    break
                max_idx = np.unravel_index(np.argmax(cost_matrix), cost_matrix.shape)
                max_iou = cost_matrix[max_idx]
                
                if max_iou >= self.iou_threshold:
                    track_idx = unmatched_tracks[max_idx[0]]
                    det_idx = unmatched_detections[max_idx[1]]
                    matched_pairs.append((track_idx, det_idx))
                    cost_matrix = np.delete(cost_matrix, max_idx[0], axis=0)
                    cost_matrix = np.delete(cost_matrix, max_idx[1], axis=1)
                    unmatched_tracks.remove(track_idx)
                    unmatched_detections.remove(det_idx)
                else:
                    break
        
        # Update matched tracks
        for track_id, det_idx in matched_pairs:
            det = detections[det_idx]
            self.tracks[track_id]['bbox'] = det['bbox']
            self.tracks[track_id]['last_seen'] = self.frame_count
            self.tracks[track_id]['class'] = det.get('class', 'unknown')
            self.tracks[track_id]['confidence'] = det.get('confidence', 0.0)
            
            # Match plate to this track
            if detected_plates:
                best_plate = None
                best_score = 0
                track_bbox = self.tracks[track_id]['bbox']
                track_center = ((track_bbox[0] + track_bbox[2]) / 2, (track_bbox[1] + track_bbox[3]) / 2)
                
                for plate in detected_plates:
                    plate_bbox = plate['bbox']
                    plate_center = ((plate_bbox[0] + plate_bbox[2]) / 2, (plate_bbox[1] + plate_bbox[3]) / 2)
                    
                    distance = np.sqrt((track_center[0] - plate_center[0])**2 + (track_center[1] - plate_center[1])**2)
                    expanded_iou = calculate_expanded_iou(track_bbox, plate_bbox, expansion_factor=2.5)
                    
                    distance_score = max(0, 1.0 - (distance / 200.0))
                    combined_score = (expanded_iou * 0.7) + (distance_score * 0.3)
                    
                    if combined_score > best_score and (expanded_iou > 0.1 or distance < 150):
                        best_score = combined_score
                        best_plate = plate
                
                if best_plate:
                    # Only overwrite if confidence improves (prevents OCR variants)
                    new_plate = best_plate['text']
                    new_conf = float(best_plate.get('confidence', 0.0) or 0.0)
                    old_conf = float(self.tracks[track_id].get('plate_confidence', 0.0) or 0.0)
                    if (not self.tracks[track_id].get('plate')) or (new_conf >= old_conf):
                        self.tracks[track_id]['plate'] = new_plate
                        self.tracks[track_id]['plate_confidence'] = new_conf
            
            current_frame_tracks[track_id] = self.tracks[track_id].copy()
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            det = detections[det_idx]
            track_id = self.next_id
            self.next_id += 1
            
            plate_text = None
            plate_conf = 0.0
            if detected_plates:
                det_bbox = det['bbox']
                det_center = ((det_bbox[0] + det_bbox[2]) / 2, (det_bbox[1] + det_bbox[3]) / 2)
                best_plate = None
                best_score = 0
                
                for plate in detected_plates:
                    plate_bbox = plate['bbox']
                    plate_center = ((plate_bbox[0] + plate_bbox[2]) / 2, (plate_bbox[1] + plate_bbox[3]) / 2)
                    
                    distance = np.sqrt((det_center[0] - plate_center[0])**2 + (det_center[1] - plate_center[1])**2)
                    expanded_iou = calculate_expanded_iou(det_bbox, plate_bbox, expansion_factor=2.5)
                    
                    distance_score = max(0, 1.0 - (distance / 200.0))
                    combined_score = (expanded_iou * 0.7) + (distance_score * 0.3)
                    
                    if combined_score > best_score and (expanded_iou > 0.1 or distance < 150):
                        best_score = combined_score
                        best_plate = plate
                
                if best_plate:
                    plate_text = best_plate['text']
                    plate_conf = best_plate.get('confidence', 0.0)
            
            self.tracks[track_id] = {
                'bbox': det['bbox'],
                'last_seen': self.frame_count,
                'first_seen': self.frame_count,
                'class': det.get('class', 'unknown'),
                'confidence': det.get('confidence', 0.0),
                'plate': plate_text,
                'plate_confidence': plate_conf,
                'position_history': []
            }
            current_frame_tracks[track_id] = self.tracks[track_id].copy()
        
        # Remove old tracks
        tracks_to_remove = []
        for track_id, track_data in self.tracks.items():
            if self.frame_count - track_data['last_seen'] > self.max_age:
                tracks_to_remove.append(track_id)
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        return current_frame_tracks


class ImprovedViolationTracker:
    """Track violations with PaddleOCR integration"""
    
    def __init__(self):
        if USE_OCR:
            try:
                self.reader = PlateOCR(use_gpu=False)
                print("✅ PaddleOCR initialized for number plate detection")
            except Exception as e:
                print(f"❌ Failed to initialize PaddleOCR: {e}")
                self.reader = None
        else:
            self.reader = None
            print("❌ PaddleOCR not available")
        
        self.vehicle_tracker = VehicleTracker(iou_threshold=0.3, max_age=15)
        self.vehicles = {}
        self.violation_log = []
        # Track -> best plate seen (avoid OCR plate variants per same vehicle)
        # {track_id: {"plate": str, "confidence": float}}
        self.best_plate_by_track = {}
        
        self.violation_types = {
            'no_helmet': 'No Helmet',
            'mobile_usage': 'Phone Usage',
            'phone_usage': 'Phone Usage',
            'triple_riding': 'Triple Riding',
            'traffic_violation': 'Traffic Rules Violation',
            'overspeed': 'Overspeed'
        }
        
        # FIXED: Speed tracking - use max speed approach
        self.speed_limit = 40  # km/h (overspeed threshold)
        self.pixels_per_meter = 15  # FIXED: More realistic calibration (50 was too high)
        self.fps = 30
        self.max_speed_detected = 0  # Track maximum speed in video
        
        self._ocr_cache = {}
        self._last_ocr_frame = {}
        self._ocr_interval = 3  # FIXED: Run OCR more frequently (was 5)
    
    def detect_number_plates(self, frame, bbox_list, verbose=False, debug_dir=None, frame_idx=None):
        """Detect number plates using PaddleOCR"""
        if not self.reader:
            return []
        
        plates = []
        seen_plates = set()
        
        bbox_list = sorted(bbox_list, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]), reverse=True)
        
        for idx, bbox in enumerate(bbox_list):
            x1, y1, x2, y2 = map(int, bbox)
            
            paddings = [10, 20, 30]
            best_result = None
            best_pad = None
            last_raw = None

            bbox_hash = f"{x1}_{y1}_{x2}_{y2}"
            cached_result = self._ocr_cache.get(bbox_hash)
            if cached_result and cached_result.get('is_valid', False):
                plate_text = cached_result['text']
                if plate_text not in seen_plates:
                    plates.append({
                        'text': plate_text,
                        'confidence': cached_result.get('confidence', 0.0),
                        'quality_score': cached_result.get('score', 0.0),
                        'bbox': [x1, y1, x2, y2],
                        'raw_text': cached_result.get('raw_text', plate_text),
                        'method': 'Cached',
                        'parts': cached_result.get('parts', {})
                    })
                    seen_plates.add(plate_text)
                    if verbose:
                        print(f"  ✅ Plate {idx+1}: {plate_text} (cached)")
                continue
            
            for padding in paddings:
                y1_pad = max(0, y1 - padding)
                y2_pad = min(frame.shape[0], y2 + padding)
                x1_pad = max(0, x1 - padding)
                x2_pad = min(frame.shape[1], x2 + padding)
                roi = frame[y1_pad:y2_pad, x1_pad:x2_pad]

                if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 20:
                    continue

                try:
                    ocr_result = self.reader.read_plate(roi, verbose=verbose)
                    last_raw = ocr_result.get('raw_text') if isinstance(ocr_result, dict) else None

                    if ocr_result and ocr_result.get('is_valid', False):
                        best_result = ocr_result
                        best_pad = padding
                        break
                except Exception:
                    continue

            self._ocr_cache[bbox_hash] = best_result

            if best_result and best_result.get('is_valid', False):
                plate_text = best_result['text']
                confidence = best_result.get('confidence', 0.0)

                if plate_text not in seen_plates:
                    plates.append({
                        'text': plate_text,
                        'confidence': min(0.95, confidence),
                        'quality_score': best_result.get('score', 0.0),
                        'bbox': [x1, y1, x2, y2],
                        'raw_text': best_result.get('raw_text', plate_text),
                        'method': f"{best_result.get('method', 'PaddleOCR')}+pad{best_pad}",
                        'parts': best_result.get('parts', {})
                    })
                    seen_plates.add(plate_text)
                    if verbose:
                        print(f"  ✅ Plate {idx+1}: {plate_text} (pad={best_pad}, conf={confidence:.2%})")
            else:
                raw_msg = f"raw='{last_raw}'" if last_raw else "raw=<none>"
                if verbose:
                    print(f"  ❌ OCR failed for plate {idx+1} (bbox={x1},{y1},{x2},{y2}) {raw_msg}")
        
        return plates
    
    def get_vehicle_key(self, track_id, plate_number=None, plate_confidence: float | None = None):
        """
        Always use a stable key per track (TRACK_{id}) to prevent duplicates.
        Plate (if any) is stored as metadata and canonicalized by best confidence.
        """
        vehicle_key = f"TRACK_{track_id}"
        if plate_number:
            prev = self.best_plate_by_track.get(track_id)
            prev_conf = float(prev.get("confidence", 0.0)) if prev else 0.0
            new_conf = float(plate_confidence or 0.0)
            if (not prev) or (new_conf >= prev_conf):
                self.best_plate_by_track[track_id] = {"plate": plate_number, "confidence": new_conf}
        return vehicle_key
    
    def calculate_speed(self, vehicle_key, current_bbox, frame_number):
        """FIXED: Calculate speed using improved method"""
        if vehicle_key not in self.vehicles:
            return None
        
        vehicle_data = self.vehicles[vehicle_key]
        
        if 'position_history' not in vehicle_data:
            vehicle_data['position_history'] = []
        
        current_center = ((current_bbox[0] + current_bbox[2]) / 2, (current_bbox[1] + current_bbox[3]) / 2)
        vehicle_data['position_history'].append({
            'frame': frame_number,
            'center': current_center,
            'bbox': current_bbox
        })
        
        if len(vehicle_data['position_history']) > 30:  # Keep more history
            vehicle_data['position_history'] = vehicle_data['position_history'][-30:]
        
        # Calculate speed with at least 8 frames (so we get speed sooner)
        if len(vehicle_data['position_history']) >= 8:
            speeds = []
            
            # Calculate total distance
            total_distance = 0
            for i in range(1, len(vehicle_data['position_history'])):
                pos1 = vehicle_data['position_history'][i-1]['center']
                pos2 = vehicle_data['position_history'][i]['center']
                total_distance += np.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
            
            min_movement_pixels = 8  # Allow small movement so speed is computed
            if total_distance < min_movement_pixels:
                return None
            
            # Use multiple intervals for accuracy
            nhist = len(vehicle_data['position_history'])
            intervals = [7, 10, 15] if nhist >= 16 else [7, 10] if nhist >= 11 else [7] if nhist >= 8 else []
            
            for interval in intervals:
                if len(vehicle_data['position_history']) >= interval + 1:
                    old_pos = vehicle_data['position_history'][-interval-1]['center']
                    new_pos = vehicle_data['position_history'][-1]['center']
                    
                    distance_pixels = np.sqrt((new_pos[0] - old_pos[0])**2 + (new_pos[1] - old_pos[1])**2)
                    
                    # Allow fast vehicles: cap by pixels (e.g. 50 km/h ~ 200px in 10 frames at 15 ppm)
                    bbox_width = current_bbox[2] - current_bbox[0]
                    bbox_height = current_bbox[3] - current_bbox[1]
                    max_reasonable_distance = max(450, max(bbox_width, bbox_height) * 3)
                    if distance_pixels > max_reasonable_distance:
                        continue
                    
                    min_distance_per_interval = 3  # FIXED: Reduced from 5
                    if distance_pixels < min_distance_per_interval:
                        continue
                    
                    # FIXED: Better conversion to meters
                    distance_meters = distance_pixels / self.pixels_per_meter
                    
                    # Use effective_fps: position_history has one entry per *processed* frame
                    eff_fps = getattr(self, 'effective_fps', None) or (self.fps if self.fps > 0 else 30.0)
                    time_diff = interval / eff_fps
                    
                    if time_diff > 0:
                        speed_ms = distance_meters / time_diff
                        speed_kmh = speed_ms * 3.6
                        
                        # Realistic road speed range (5–200 km/h)
                        if 5 <= speed_kmh <= 200:
                            speeds.append(speed_kmh)
            
            if speeds:
                speed_kmh = np.median(speeds)
                
                if 'speed_history' not in vehicle_data:
                    vehicle_data['speed_history'] = []
                vehicle_data['speed_history'].append(speed_kmh)
                
                if len(vehicle_data['speed_history']) > 15:  # Keep more history
                    vehicle_data['speed_history'] = vehicle_data['speed_history'][-15:]
                
                # Update max speed ONLY for vehicles with plates (ignore parked/stationary vehicles)
                # This prevents false max speeds from parked vehicles or noise
                vehicle_key = vehicle_key  # Already in context
                if vehicle_key in self.vehicles:
                    vehicle_data_check = self.vehicles[vehicle_key]
                    plate_num = vehicle_data_check.get('plate_number')
                    # Only track max speed if vehicle has a plate (moving vehicle)
                    if plate_num and not str(plate_num).startswith('TRACK_'):
                        if speed_kmh > self.max_speed_detected:
                            self.max_speed_detected = speed_kmh
                
                return sum(vehicle_data['speed_history']) / len(vehicle_data['speed_history'])
        
        return None
    
    def add_violation(self, vehicle_id, violation_type, confidence, bbox, frame_number, speed=None, plate_number=None):
        """Add a violation for a vehicle - ONLY log if valid plate number exists"""
        # Always key by track to avoid duplicate vehicles/plates in summary
        vehicle_key = f"TRACK_{vehicle_id}"
        # Canonical plate for this track (best OCR confidence)
        best = self.best_plate_by_track.get(vehicle_id)
        canonical_plate = best["plate"] if best else plate_number
        
        # ✅ FIX 1: Skip violations without valid plate numbers
        if not canonical_plate or str(canonical_plate).startswith('TRACK_'):
            # For traffic_violation, be extra strict - skip completely
            if violation_type == 'traffic_violation':
                return False
            # For others, still skip but could log internally for debugging
            # print(f"⚠️ Skipping {violation_type} - no valid plate for {vehicle_key}")
            return False
        
        # ✅ FIX 2: Filter out low-confidence traffic violations
        if violation_type == 'traffic_violation' and confidence < 0.85:
            print(f"⚠️ Skipping low-confidence traffic violation (conf: {confidence:.2f})")
            return False
        
        if vehicle_key not in self.vehicles:
            self.vehicles[vehicle_key] = {
                'violations': set(),
                'first_seen': datetime.now(),
                'last_seen': datetime.now(),
                'total_detections': 0,
                'speed_history': [],
                'track_id': vehicle_id,
                'plate_number': canonical_plate
            }
        
        vehicle = self.vehicles[vehicle_key]
        vehicle['last_seen'] = datetime.now()
        vehicle['total_detections'] += 1
        if canonical_plate:
            vehicle['plate_number'] = canonical_plate
        
        if violation_type in vehicle['violations']:
            return False
        
        vehicle['violations'].add(violation_type)
        
        # ✅ FIX 3: Log with actual plate number
        violation_record = {
            'id': f'vid_{len(self.violation_log) + 1}',
            'plate_number': canonical_plate,  # Always actual plate, never TRACK_
            'type': violation_type,
            'type_name': self.violation_types.get(violation_type, violation_type),
            'confidence': float(confidence),
            'timestamp': datetime.now().isoformat(),
            'frame_number': int(frame_number),
            'bbox': [float(b) for b in bbox],
            'speed': f"{speed:.1f} km/h" if speed else 'N/A',
            'location': 'Traffic Camera',
            'severity': 'high' if violation_type in ['no_helmet', 'overspeed'] else 'medium',
            'description': f'{self.violation_types.get(violation_type, violation_type)} detected',
            'track_id': vehicle_key  # Keep for debugging
        }
        
        self.violation_log.append(violation_record)
        print(f"🚨 Frame {frame_number}: {violation_type} violation detected (conf: {confidence:.2f}) for plate {canonical_plate}")
        return True
    
    def get_summary(self):
        """Get summary of all detected vehicles and violations"""
        vehicles_summary = []
        unique_track_ids = set()  # Track unique vehicles by track_id
        
        for vehicle_key, data in self.vehicles.items():
            track_id = data.get('track_id')
            # Only count "real" vehicles: ignore very short/noisy tracks
            if track_id and data.get('total_detections', 0) >= 10:
                unique_track_ids.add(track_id)
            
            avg_speed = None
            max_speed = None
            if 'speed_history' in data and data['speed_history']:
                avg_speed = sum(data['speed_history']) / len(data['speed_history'])
                max_speed = max(data['speed_history'])
            
            vehicles_summary.append({
                'plate_number': data.get('plate_number', vehicle_key),
                'track_id': track_id,
                'violations': list(data['violations']),
                'violation_names': [self.violation_types.get(v, v) for v in data['violations']],
                'violation_count': len(data['violations']),
                'first_seen': data['first_seen'].isoformat(),
                'last_seen': data['last_seen'].isoformat(),
                'total_detections': data['total_detections'],
                'average_speed': avg_speed,
                'max_speed': max_speed
            })
        
        return {
            'total_vehicles': len(unique_track_ids),  # Count unique vehicles by track_id
            'total_violations': len(self.violation_log),
            'vehicles': vehicles_summary,
            'violation_types_count': {
                vtype: sum(1 for v in self.violation_log if v['type'] == vtype)
                for vtype in self.violation_types.keys()
            },
            'max_speed_detected': self.max_speed_detected
        }


def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return inter_area / union_area


def calculate_expanded_iou(violation_bbox, plate_bbox, expansion_factor=1.5):
    """Calculate IoU with expanded plate bbox"""
    plate_x1, plate_y1, plate_x2, plate_y2 = plate_bbox
    plate_width = plate_x2 - plate_x1
    plate_height = plate_y2 - plate_y1
    
    expanded_x1 = max(0, plate_x1 - plate_width * (expansion_factor - 1) / 2)
    expanded_y1 = max(0, plate_y1 - plate_height * expansion_factor)
    expanded_x2 = plate_x2 + plate_width * (expansion_factor - 1) / 2
    expanded_y2 = plate_y2 + plate_height * (expansion_factor - 1) / 2
    
    expanded_plate = [expanded_x1, expanded_y1, expanded_x2, expanded_y2]
    return calculate_iou(violation_bbox, expanded_plate)


def process_video_improved(video_path, output_dir='output', skip_frames=2, use_gpu=True):
    """FIXED: Process video with all corrections applied"""
    
    print(f"\n{'='*80}")
    print(f"🚦 TRAFFIC VIOLATION DETECTION SYSTEM - CORRECTED VERSION")
    print(f"{'='*80}")
    print(f"🎥 Video: {video_path}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")
    
    # 1) Pretrained YOLOv8 for vehicle detection (car, motorcycle, bus, truck) - tracking & speed
    print("\n🤖 Loading pretrained YOLOv8 for vehicle detection (tracking + speed)...")
    vehicle_model = None
    for vpath in ['yolov8n.pt', 'yolov8s.pt']:
        try:
            vehicle_model = YOLO(vpath)
            print(f"✅ Vehicle model: {vpath}")
            break
        except Exception as e:
            print(f"⚠️  Failed to load {vpath}: {e}")
            continue
    if vehicle_model is None:
        print("❌ Error: No vehicle model (yolov8n.pt) found. Install: pip install ultralytics")
        return

    # 2) Trained refined model for helmet / no_helmet / number_plate (violations + plate boxes)
    print("🤖 Loading trained YOLOv8 refined model (helmet, plate, violations)...")
    violation_model_paths = [
        'runs/final_refined_model/weights/best.pt',
        'runs/final_model_refined/weights/best.pt',
        'runs/final_model/weights/best.pt',
    ]
    violation_model = None
    for model_path in violation_model_paths:
        if os.path.exists(model_path):
            try:
                violation_model = YOLO(model_path)
                print(f"✅ Violation model: {model_path}")
                break
            except Exception as e:
                print(f"⚠️  Failed to load {model_path}: {e}")
                continue
    if violation_model is None:
        print("❌ Error: No valid violation model found")
        return
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"✅ Video loaded: {width}x{height} @ {fps} FPS, {total_frames} frames")
    print(f"⚡ Frame skipping: {skip_frames} (processing every {skip_frames+1} frames)")
    
    tracker = ImprovedViolationTracker()
    tracker.fps = fps
    # Effective FPS: we process every (skip_frames+1) frames, so time between processed frames is (skip_frames+1)/fps
    tracker.effective_fps = fps / (skip_frames + 1) if (skip_frames + 1) > 0 else fps
    # PPM: scale by frame size (portrait 1080x1920 vs landscape differ)
    # For portrait videos, use width-based calibration (road width ~4-5m)
    h, w = height, width
    # PPM calibration: adjust to match actual speeds (if 47 km/h is correct but getting 57.1, PPM is too low)
    # Higher PPM = lower calculated speed, so increase PPM to reduce speed
    # Ratio: 57.1/47 = 1.215, so multiply PPM by 1.215
    if h > w:  # Portrait
        base_ppm = w / 4.5  # Assume ~4.5m road width
        tracker.pixels_per_meter = max(15, min(60, base_ppm * 1.2))  # Adjusted for accuracy
    else:  # Landscape
        base_ppm = (w + h) / 2 / 8
        tracker.pixels_per_meter = max(15, min(60, base_ppm * 1.2))
    print(f"📏 Pixels per meter (PPM): {tracker.pixels_per_meter:.1f} (calibrated for {w}x{h})")
    
    # Portrait (mobile) video often stored upside down; correct for display
    portrait_rotate_180 = (h > w)
    if portrait_rotate_180:
        print("📱 Portrait video detected: output will be rotated right-side up")
    
    video_name = Path(video_path).stem
    output_video_path = os.path.join(output_dir, f'annotated_{video_name}.mp4')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # FIXED: No rotation - write video with original orientation
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    colors = {
        'number_plate': (0, 255, 0),
        'helmet': (0, 255, 0),
        'no_helmet': (0, 0, 255),  # FIXED: Red for better visibility
        'mobile_usage': (0, 165, 255),
        'phone_usage': (0, 165, 255),
        'triple_riding': (255, 0, 255),
        'traffic_violation': (255, 0, 255),
        'overspeed': (0, 255, 255)
    }
    
    frame_count = 0
    processed_frames = 0
    start_time = datetime.now()
    last_annotated_frame = None
    
    # FIXED: Track plates and their violations for terminal output
    plate_speed_map = {}  # {plate_number: {'speeds': [], 'violations': set()}}
    
    print(f"\n🎬 Processing video...")
    print(f"{'='*80}\n")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Portrait video from phone is often upside down; rotate so output is right-side up
            if portrait_rotate_180:
                frame = cv2.rotate(frame, cv2.ROTATE_180)
            
            should_process = (frame_count % (skip_frames + 1) == 0)
            
            if should_process:
                processed_frames += 1

                # --- Vehicle detection (pretrained COCO): for tracking and speed ---
                vehicle_results = vehicle_model.predict(
                    frame, conf=0.4, imgsz=512, verbose=False, device=device, half=False
                )
                vehicle_detections = []
                for result in vehicle_results:
                    if result.boxes is None:
                        continue
                    for box in result.boxes:
                        class_id = int(box.cls[0].cpu().numpy())
                        if class_id not in COCO_VEHICLE_IDS:
                            continue
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_name = COCO_VEHICLE_IDS[class_id]
                        bbox = [float(x1), float(y1), float(x2), float(y2)]
                        vehicle_detections.append({
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': bbox
                        })

                # --- Violation detection (trained model): no_helmet, mobile_usage, triple_riding, number_plate ---
                violation_results = violation_model.predict(
                    frame, conf=0.25, imgsz=512, verbose=False, device=device, half=False
                )
                number_plate_boxes = []
                violation_detections = []
                for result in violation_results:
                    if result.boxes is None:
                        continue
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = violation_model.names[class_id]
                        bbox = [float(x1), float(y1), float(x2), float(y2)]
                        violation_detections.append({
                            'class': class_name,
                            'confidence': confidence,
                            'bbox': bbox
                        })
                        if class_name == 'number_plate':
                            number_plate_boxes.append(bbox)
                
                # Run OCR more frequently
                detected_plates = []
                run_ocr = False
                
                if number_plate_boxes:
                    if processed_frames == 1 or processed_frames % tracker._ocr_interval == 0:
                        run_ocr = True
                    elif not hasattr(tracker, '_last_plate_count') or len(number_plate_boxes) != tracker._last_plate_count:
                        run_ocr = True
                    elif len(getattr(tracker, '_cached_plates', [])) == 0:
                        run_ocr = True
                    
                    if run_ocr:
                        verbose_ocr = (processed_frames <= 5)
                        detected_plates = tracker.detect_number_plates(
                            frame,
                            number_plate_boxes,
                            verbose=verbose_ocr
                        )
                        tracker._last_plate_count = len(number_plate_boxes)
                        tracker._cached_plates = detected_plates
                        
                        # FIXED: Print detected plates with frame number
                        if detected_plates:
                            print(f"🔤 Frame {frame_count}: Extracted {len(detected_plates)} number plate(s):")
                            for plate in detected_plates:
                                print(f"   ✅ {plate['text']} (confidence: {plate['confidence']:.2%})")
                    else:
                        detected_plates = getattr(tracker, '_cached_plates', [])

                # Track VEHICLES only (from pretrained COCO) for stable IDs and speed
                tracked_vehicles = tracker.vehicle_tracker.update(vehicle_detections, detected_plates)
                
                # Match plates to tracks (plate often inside or below vehicle bbox, e.g. bike front)
                if detected_plates:
                    for track_id, track_data in tracked_vehicles.items():
                        if not track_data.get('plate'):
                            best_plate = None
                            best_score = 0
                            t_bbox = track_data['bbox']
                            # Expand vehicle bbox down and sideways so plate below/near bike still matches
                            tw, th = t_bbox[2] - t_bbox[0], t_bbox[3] - t_bbox[1]
                            t_expand = [t_bbox[0] - tw * 0.2, t_bbox[1], t_bbox[2] + tw * 0.2, t_bbox[3] + th * 0.4]
                            for plate in detected_plates:
                                p_bbox = plate['bbox']
                                expanded_iou = calculate_expanded_iou(t_bbox, p_bbox, expansion_factor=2.5)
                                p_cx = (p_bbox[0] + p_bbox[2]) / 2
                                p_cy = (p_bbox[1] + p_bbox[3]) / 2
                                inside = (t_expand[0] <= p_cx <= t_expand[2] and t_expand[1] <= p_cy <= t_expand[3])
                                dist = np.sqrt((p_cx - (t_bbox[0]+t_bbox[2])/2)**2 + (p_cy - (t_bbox[1]+t_bbox[3])/2)**2)
                                near = dist < max(tw, th) * 1.2  # plate within 1.2x bbox size of vehicle center
                                score = expanded_iou if expanded_iou > 0.05 else (0.5 if inside else (0.35 if near else 0))
                                if score > best_score and (expanded_iou > 0.08 or inside or near):
                                    best_score = score
                                    best_plate = plate
                            if best_plate:
                                track_data['plate'] = best_plate['text']
                                track_data['plate_confidence'] = best_plate.get('confidence', 0.0)
                        # Update canonical best-plate mapping (even if plate already existed)
                        if track_data.get('plate'):
                            tracker.get_vehicle_key(
                                track_id,
                                track_data.get('plate'),
                                float(track_data.get('plate_confidence') or 0.0),
                            )

                annotated_frame = frame.copy()

                # Initialize vehicles and calculate speed for EVERY track (not only those with violations)
                for track_id, track_data in tracked_vehicles.items():
                    plate_number = track_data.get('plate')
                    plate_conf = float(track_data.get('plate_confidence') or 0.0)
                    vehicle_key = tracker.get_vehicle_key(track_id, plate_number, plate_conf)
                    if vehicle_key not in tracker.vehicles:
                        tracker.vehicles[vehicle_key] = {
                            'violations': set(),
                            'first_seen': datetime.now(),
                            'last_seen': datetime.now(),
                            'total_detections': 0,
                            'speed_history': [],
                            'track_id': track_id,
                            'plate_number': (tracker.best_plate_by_track.get(track_id) or {}).get("plate")
                        }
                    else:
                        best = tracker.best_plate_by_track.get(track_id)
                        if best and best.get("plate"):
                            tracker.vehicles[vehicle_key]['plate_number'] = best["plate"]
                    # Speed for every vehicle every frame so position_history accumulates
                    speed = tracker.calculate_speed(vehicle_key, track_data['bbox'], processed_frames)
                    canonical_plate = (tracker.best_plate_by_track.get(track_id) or {}).get("plate") or plate_number
                    if canonical_plate:
                        if canonical_plate not in plate_speed_map:
                            plate_speed_map[canonical_plate] = {'speeds': [], 'violations': set(), 'max_speed': 0}
                        if speed is not None:
                            plate_speed_map[canonical_plate]['speeds'].append(speed)
                            # Track max speed per plate
                            if speed > plate_speed_map[canonical_plate]['max_speed']:
                                plate_speed_map[canonical_plate]['max_speed'] = speed

                # Match violation detections (no_helmet, etc.) to vehicle tracks
                # Filter false positives: mobile_usage, triple_riding, traffic_violation need higher confidence
                track_violations = defaultdict(list)
                for detection in violation_detections:
                    class_name = detection['class']
                    confidence = detection['confidence']
                    if class_name not in tracker.violation_types:
                        continue
                    # Filter false positives: require higher confidence for these violation types
                    if class_name in ['mobile_usage', 'phone_usage', 'triple_riding', 'traffic_violation']:
                        if confidence < 0.5:  # Require at least 50% confidence for these
                            continue
                    best_track = None
                    best_score = 0
                    v_bbox = detection['bbox']
                    v_center = ((v_bbox[0] + v_bbox[2]) / 2, (v_bbox[1] + v_bbox[3]) / 2)
                    v_area = (v_bbox[2] - v_bbox[0]) * (v_bbox[3] - v_bbox[1])
                    for track_id, track_data in tracked_vehicles.items():
                        t_bbox = track_data['bbox']
                        t_center = ((t_bbox[0] + t_bbox[2]) / 2, (t_bbox[1] + t_bbox[3]) / 2)
                        iou = calculate_iou(v_bbox, t_bbox)
                        # Violation center inside vehicle bbox
                        inside = (t_bbox[0] <= v_center[0] <= t_bbox[2] and t_bbox[1] <= v_center[1] <= t_bbox[3])
                        # Distance between centers (normalized by vehicle size)
                        t_w, t_h = t_bbox[2] - t_bbox[0], t_bbox[3] - t_bbox[1]
                        dist = np.sqrt((v_center[0] - t_center[0])**2 + (v_center[1] - t_center[1])**2)
                        max_dim = max(t_w, t_h)
                        normalized_dist = dist / max_dim if max_dim > 0 else 999
                        # Score: IoU > inside > near (within 1.5x vehicle size)
                        if iou > 0.05:
                            score = iou
                        elif inside:
                            score = 0.4
                        elif normalized_dist < 1.5:  # Violation within 1.5x vehicle size
                            score = 0.3 * (1.0 - normalized_dist / 1.5)
                        else:
                            score = 0
                        if score > best_score:
                            best_score = score
                            best_track = track_id
                    # Very low threshold - match if violation is anywhere near a vehicle
                    if best_track and best_score > 0.01:
                        track_violations[best_track].append((detection, best_score))
                
                # Process violations for each track
                for track_id, violations in track_violations.items():
                    track_data = tracked_vehicles[track_id]
                    plate_number = track_data.get('plate')
                    plate_conf = float(track_data.get('plate_confidence') or 0.0)
                    
                    vehicle_key = tracker.get_vehicle_key(track_id, plate_number, plate_conf)
                    
                    if vehicle_key not in tracker.vehicles:
                        tracker.vehicles[vehicle_key] = {
                            'violations': set(),
                            'first_seen': datetime.now(),
                            'last_seen': datetime.now(),
                            'total_detections': 0,
                            'speed_history': [],
                            'track_id': track_id,
                            'plate_number': (tracker.best_plate_by_track.get(track_id) or {}).get("plate") or plate_number
                        }
                    else:
                        best = tracker.best_plate_by_track.get(track_id)
                        if best and best.get("plate"):
                            tracker.vehicles[vehicle_key]['plate_number'] = best["plate"]
                    
                    # Calculate speed
                    speed = tracker.calculate_speed(vehicle_key, track_data['bbox'], processed_frames)
                    
                    # FIXED: Track speed for this plate
                    canonical_plate = (tracker.best_plate_by_track.get(track_id) or {}).get("plate") or plate_number
                    if canonical_plate and speed:
                        if canonical_plate not in plate_speed_map:
                            plate_speed_map[canonical_plate] = {'speeds': [], 'violations': set(), 'max_speed': 0}
                        plate_speed_map[canonical_plate]['speeds'].append(speed)
                        if speed > plate_speed_map[canonical_plate]['max_speed']:
                            plate_speed_map[canonical_plate]['max_speed'] = speed
                    
                    # Process each violation
                    for detection, iou in violations:
                        class_name = detection['class']
                        confidence = detection['confidence']
                        bbox = detection['bbox']
                        
                        # Add violation
                        added = tracker.add_violation(
                            track_id, class_name, confidence, bbox, processed_frames, speed, plate_number
                        )
                        
                        # FIXED: Track violations for this plate
                        canonical_plate = (tracker.best_plate_by_track.get(track_id) or {}).get("plate") or plate_number
                        if added and canonical_plate:
                            if canonical_plate not in plate_speed_map:
                                plate_speed_map[canonical_plate] = {'speeds': [], 'violations': set(), 'max_speed': 0}
                            plate_speed_map[canonical_plate]['violations'].add(class_name)
                            # Print violation detection for debugging
                            print(f"🚨 Frame {frame_count}: {class_name} violation detected (conf: {confidence:.2f}) for plate {canonical_plate}")
                        
                        # FIXED: Check for overspeed using current speed (not max)
                        # Will do final overspeed check at end using max speed
                        if speed and speed > tracker.speed_limit:
                            overspeed_added = tracker.add_violation(
                                track_id, 'overspeed', confidence, bbox, processed_frames, speed, plate_number
                            )
                            canonical_plate = (tracker.best_plate_by_track.get(track_id) or {}).get("plate") or plate_number
                            if overspeed_added and canonical_plate:
                                if canonical_plate not in plate_speed_map:
                                    plate_speed_map[canonical_plate] = {'speeds': [], 'violations': set(), 'max_speed': 0}
                                plate_speed_map[canonical_plate]['violations'].add('overspeed')
                
                # Draw vehicle tracks (with speed and plate if available)
                for track_id, track_data in tracked_vehicles.items():
                    bbox = track_data['bbox']
                    x1, y1, x2, y2 = [int(b) for b in bbox]
                    plate_number = track_data.get('plate')
                    plate_conf = float(track_data.get('plate_confidence') or 0.0)
                    vehicle_key = tracker.get_vehicle_key(track_id, plate_number, plate_conf)
                    speed = None
                    if vehicle_key in tracker.vehicles:
                        speed = tracker.calculate_speed(vehicle_key, bbox, processed_frames)
                    label = f"#{track_id} {track_data.get('class', 'vehicle')}"
                    best = tracker.best_plate_by_track.get(track_id)
                    canonical_plate = (best or {}).get("plate") or plate_number
                    if canonical_plate:
                        label += f" | {canonical_plate}"
                    if speed is not None:
                        label += f" | {speed:.1f} km/h"
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    # Larger, bold text for better visibility
                    font_scale = 0.9
                    thickness = 2
                    (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    cv2.putText(annotated_frame, label, (x1, max(text_height + 5, y1 - 5)),
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

                # Draw violation detections (no_helmet, etc.)
                for detection in violation_detections:
                    class_name = detection['class']
                    if class_name not in tracker.violation_types:
                        continue
                    confidence = detection['confidence']
                    bbox = detection['bbox']
                    x1, y1, x2, y2 = [int(b) for b in bbox]
                    color = colors.get(class_name, (0, 0, 255))
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                    # Larger, bold text for violations
                    font_scale = 0.9
                    thickness = 2
                    violation_label = f"{class_name}: {confidence:.2f}"
                    (text_width, text_height), baseline = cv2.getTextSize(violation_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    cv2.putText(annotated_frame, violation_label, (x1, max(text_height + 5, y1 - 5)),
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
                
                # Draw detected plates
                for plate in detected_plates:
                    x1, y1, x2, y2 = [int(b) for b in plate['bbox']]
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    plate_label = f"Plate: {plate['text']}"
                    font_scale = 0.9
                    thickness = 2
                    (text_width, text_height), baseline = cv2.getTextSize(plate_label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    cv2.putText(annotated_frame, plate_label, (x1, max(text_height + 5, y1 - 30)),
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)
                
                last_annotated_frame = annotated_frame.copy()
            
            # Write frame
            if last_annotated_frame is not None:
                out.write(last_annotated_frame)
            else:
                out.write(frame)
            
            # Progress update
            if processed_frames > 0 and processed_frames % 50 == 0:
                progress = (frame_count / total_frames) * 100
                summary = tracker.get_summary()
                elapsed = (datetime.now() - start_time).total_seconds()
                fps_actual = processed_frames / elapsed if elapsed > 0 else 0
                
                plates_count = len(plate_speed_map) if plate_speed_map else 0
                
                print(f"📈 Progress: {progress:.1f}% ({frame_count}/{total_frames}) - "
                      f"{summary['total_vehicles']} vehicles, {plates_count} plates, "
                      f"{summary['total_violations']} violations, max speed: {tracker.max_speed_detected:.1f} km/h - "
                      f"Speed: {fps_actual:.2f} FPS")
    
    except KeyboardInterrupt:
        print("\n⏹️ Processing stopped by user")
    
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        cap.release()
        out.release()
        
        # FIXED: Final overspeed check based on max speed in video
        print(f"\n{'='*80}")
        print(f"🎯 FINAL OVERSPEED CHECK (Max Speed Threshold: {tracker.speed_limit} km/h)")
        print(f"{'='*80}")
        # Use per-plate max speeds (canonical) to avoid noisy global spikes
        global_max_speed = 0.0
        for pdata in (plate_speed_map or {}).values():
            global_max_speed = max(global_max_speed, float(pdata.get("max_speed", 0.0) or 0.0))
        print(f"Maximum speed detected in video: {global_max_speed:.1f} km/h")
        
        if global_max_speed >= tracker.speed_limit:
            print(f"⚠️  OVERSPEED DETECTED! Max speed {global_max_speed:.1f} km/h exceeds limit of {tracker.speed_limit} km/h")
            
            # ✅ FIX: Ensure overspeed violations are logged with actual plate numbers
            for plate_num, data in plate_speed_map.items():
                max_plate_speed = float(data.get("max_speed", 0.0) or 0.0)
                if max_plate_speed >= tracker.speed_limit:
                    print(f"   🚨 Vehicle {plate_num}: Max speed {max_plate_speed:.1f} km/h - OVERSPEED")
                    data.setdefault("violations", set()).add("overspeed")
                    
                    # Add overspeed violation to tracker vehicles (by track)
                    for vehicle_key, vehicle_data in tracker.vehicles.items():
                        if vehicle_data.get('plate_number') == plate_num:
                            vehicle_data['violations'].add('overspeed')
                            
                            # ✅ FIX: Add overspeed to violation_log with actual plate number
                            # Check if overspeed already logged for this plate
                            already_logged = any(
                                v.get('type') == 'overspeed' and v.get('plate_number') == plate_num 
                                for v in tracker.violation_log
                            )
                            
                            if not already_logged:
                                overspeed_violation = {
                                    'id': f'vid_{len(tracker.violation_log) + 1}',
                                    'plate_number': plate_num,  # ✅ Actual plate number
                                    'type': 'overspeed',
                                    'type_name': 'Overspeed',
                                    'confidence': 0.95,  # High confidence for calculated speed
                                    'timestamp': datetime.now().isoformat(),
                                    'frame_number': 0,  # Final check
                                    'bbox': [0, 0, 0, 0],  # Not frame-specific
                                    'speed': f"{max_plate_speed:.1f} km/h",
                                    'location': 'Traffic Camera',
                                    'severity': 'high',
                                    'description': f'Overspeed detected - {max_plate_speed:.1f} km/h in {tracker.speed_limit} km/h zone',
                                    'track_id': vehicle_key
                                }
                                tracker.violation_log.append(overspeed_violation)
                                print(f"   ✅ Logged overspeed violation for {plate_num}")
        else:
            print(f"✅ No overspeed detected - Max speed {global_max_speed:.1f} km/h is within limit")
        
        summary = tracker.get_summary()
        
        # Save violations
        violations_file = os.path.join(output_dir, "violations_log.json")
        with open(violations_file, 'w', encoding='utf-8') as f:
            json.dump(tracker.violation_log, f, indent=2, ensure_ascii=False)
        
        summary_file = os.path.join(output_dir, "vehicle_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # FIXED: Terminal output showing plates, speeds, and violations
        elapsed_time = datetime.now() - start_time
        elapsed_seconds = elapsed_time.total_seconds()
        
        # Count unique plates from plate_speed_map (canonical, deduplicated)
        plates_extracted = len(plate_speed_map) if plate_speed_map else 0
        violations_with_plates = sum(1 for v in tracker.violation_log if v.get('plate_number') and not str(v['plate_number']).startswith('TRACK_'))
        
        print(f"\n{'='*80}")
        if plates_extracted > 0 and summary['total_violations'] > 0:
            print(f"✅ PROCESSING SUCCESSFUL!")
        elif summary['total_violations'] > 0:
            print(f"⚠️  PROCESSING COMPLETED WITH WARNINGS")
        else:
            print(f"⚠️  PROCESSING COMPLETED (No violations detected)")
        print(f"{'='*80}")
        
        print(f"\n📊 Processing Statistics:")
        print(f"   - Frames processed: {processed_frames} (of {frame_count} total)")
        print(f"   - Total vehicles detected: {summary['total_vehicles']}")
        print(f"   - Number plates extracted: {plates_extracted}")
        print(f"   - Total violations logged: {summary['total_violations']}")
        print(f"   - Violations with plate numbers: {violations_with_plates}")
        print(f"   - Maximum speed detected: {global_max_speed:.1f} km/h")
        print(f"   - Processing time: {elapsed_time}")
        print(f"   - Average FPS: {processed_frames / elapsed_seconds:.2f}" if elapsed_seconds > 0 else "   - Average FPS: N/A")
        
        # FIXED: Detailed terminal output
        print(f"\n{'='*80}")
        print(f"🔤 DETECTED NUMBER PLATES WITH SPEEDS AND VIOLATIONS:")
        print(f"{'='*80}\n")
        
        # Build plate -> violations from tracker.vehicles (ground truth; includes no_helmet before plate was first seen)
        plate_violations = {}
        for vkey, vdata in tracker.vehicles.items():
            plate = vdata.get('plate_number')
            if plate and not str(plate).startswith('TRACK_'):
                plate_violations.setdefault(plate, set()).update(vdata.get('violations', set()))
        
        if plate_speed_map:
            for plate_num, data in sorted(plate_speed_map.items()):
                print(f"🚗 Plate: {plate_num}")
                print(f"   {'─'*70}")
                
                # Use tracked max_speed (more accurate than recalculating from speeds list)
                max_speed = data.get('max_speed', max(data['speeds']) if data['speeds'] else 0)
                if max_speed > 0:
                    print(f"   🚦 Maximum Speed: {max_speed:.1f} km/h")
                    if max_speed >= tracker.speed_limit:
                        print(f"   ⚠️  OVERSPEED DETECTED! (Limit: {tracker.speed_limit} km/h)")
                else:
                    print(f"   🚦 Speed: Not calculated")
                
                vset = plate_violations.get(plate_num, set()) or set()
                if vset:
                    print(f"   🚨 Violations Detected ({len(vset)}):")
                    for violation in sorted(vset):
                        violation_name = tracker.violation_types.get(violation, violation)
                        print(f"      • {violation_name}")
                else:
                    print(f"   ✅ No violations detected")
                
                print()
        else:
            print("   ⚠️  No number plates detected with tracking data")
        
        # Show violations summary
        print(f"{'='*80}")
        print(f"📋 VIOLATIONS SUMMARY:")
        print(f"{'='*80}")
        for violation_type, count in summary['violation_types_count'].items():
            if count > 0:
                violation_name = tracker.violation_types.get(violation_type, violation_type)
                print(f"   • {violation_name}: {count}")
        
        print(f"\n{'='*80}")
        print(f"📁 Output Files:")
        print(f"   🎥 Annotated video: {output_video_path}")
        print(f"   📄 Violations log: {violations_file}")
        print(f"   📄 Vehicle summary: {summary_file}")
        
        print(f"\n{'='*80}")
        if plates_extracted > 0 and violations_with_plates > 0:
            print(f"✅ SUCCESS: {plates_extracted} plate(s) extracted, {violations_with_plates} violation(s) logged")
        elif summary['total_violations'] > 0:
            print(f"⚠️  PARTIAL: {summary['total_violations']} violation(s) detected")
        else:
            print(f"ℹ️  COMPLETED: No violations detected")
        print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Traffic Violation Detection - CORRECTED VERSION'
    )
    parser.add_argument('video_path', help='Path to input video file')
    parser.add_argument('--output', '-o', default='output', help='Output directory')
    parser.add_argument('--skip-frames', '-s', type=int, default=2, help='Skip N frames (default: 2)')
    parser.add_argument('--cpu', action='store_true', help='Force CPU usage')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video_path):
        print(f"❌ Error: Video file not found: {args.video_path}")
        return
    
    process_video_improved(
        video_path=args.video_path,
        output_dir=args.output,
        skip_frames=args.skip_frames,
        use_gpu=not args.cpu
    )


if __name__ == "__main__":
    main()
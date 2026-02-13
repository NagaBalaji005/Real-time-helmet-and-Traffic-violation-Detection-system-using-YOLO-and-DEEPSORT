"""
DeepSORT Tracker for Traffic Violation Detection
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
from deep_sort_realtime.deepsort_tracker import DeepSort
from config import TRACKING_CONFIG

class Tracker:
    def __init__(self, max_age: int = None, min_hits: int = None):
        """
        Initialize DeepSORT tracker
        Args:
            max_age: Maximum age of track before deletion
            min_hits: Minimum detections before tracking starts
        """
        self.max_age = max_age or TRACKING_CONFIG['max_age']
        self.min_hits = min_hits or TRACKING_CONFIG['min_hits']
        self.iou_threshold = TRACKING_CONFIG['iou_threshold']
        
        # Initialize DeepSORT tracker
        self.tracker = DeepSort(
            max_age=self.max_age,
            n_init=self.min_hits,
            nms_max_overlap=1.0,
            max_cosine_distance=0.2,
            nn_budget=None
        )
        
        # Track history
        self.track_history = {}
        self.frame_count = 0
        
        print(f"Tracker initialized with max_age={self.max_age}, min_hits={self.min_hits}")
    
    def update(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """
        Update tracker with new detections
        Args:
            detections: List of detection dictionaries
            frame: Current frame
        Returns:
            List of tracked objects
        """
        self.frame_count += 1
        
        # Convert detections to DeepSORT format
        detection_list = []
        for det in detections:
            bbox = det['bbox']
            detection_list.append((bbox, det['confidence'], det['class_name']))
        
        # Update tracker
        tracks = self.tracker.update_tracks(detection_list, frame=frame)
        
        # Convert tracks to our format
        tracked_objects = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            bbox = track.to_tlbr()  # top left bottom right
            
            # Update track history
            if track_id not in self.track_history:
                self.track_history[track_id] = []
            
            self.track_history[track_id].append({
                'frame': self.frame_count,
                'bbox': bbox.tolist(),
                'class_name': track.det_class
            })
            
            # Keep only recent history
            if len(self.track_history[track_id]) > 30:
                self.track_history[track_id] = self.track_history[track_id][-30:]
            
            tracked_object = {
                'track_id': track_id,
                'bbox': bbox.tolist(),
                'class_name': track.det_class,
                'confidence': track.get_det_conf(),
                'age': track.time_since_update,
                'hits': track.hits,
                'history': self.track_history[track_id]
            }
            tracked_objects.append(tracked_object)
        
        return tracked_objects
    
    def get_track_by_id(self, track_id: int) -> Dict:
        """
        Get track information by ID
        Args:
            track_id: Track ID
        Returns:
            Track information or None
        """
        return self.track_history.get(track_id)
    
    def get_tracks_by_class(self, tracked_objects: List[Dict], class_name: str) -> List[Dict]:
        """
        Get tracks for specific class
        Args:
            tracked_objects: List of tracked objects
            class_name: Class name to filter by
        Returns:
            Filtered tracks
        """
        return [track for track in tracked_objects if track['class_name'] == class_name]
    
    def get_vehicle_tracks(self, tracked_objects: List[Dict]) -> List[Dict]:
        """
        Get vehicle tracks (for speed estimation)
        Args:
            tracked_objects: List of tracked objects
        Returns:
            Vehicle tracks
        """
        vehicle_classes = ['vehicle', 'rider']
        return [track for track in tracked_objects if track['class_name'] in vehicle_classes]
    
    def calculate_centroid(self, bbox: List[float]) -> Tuple[float, float]:
        """
        Calculate centroid of bounding box
        Args:
            bbox: [x1, y1, x2, y2]
        Returns:
            Centroid coordinates (x, y)
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def get_track_velocity(self, track_id: int, frames: int = 5) -> Tuple[float, float]:
        """
        Calculate track velocity based on recent positions
        Args:
            track_id: Track ID
            frames: Number of frames to consider
        Returns:
            Velocity vector (dx, dy) in pixels per frame
        """
        if track_id not in self.track_history:
            return (0, 0)
        
        history = self.track_history[track_id]
        if len(history) < 2:
            return (0, 0)
        
        # Get recent positions
        recent_positions = history[-frames:] if len(history) >= frames else history
        
        if len(recent_positions) < 2:
            return (0, 0)
        
        # Calculate velocity from first to last position
        first_pos = self.calculate_centroid(recent_positions[0]['bbox'])
        last_pos = self.calculate_centroid(recent_positions[-1]['bbox'])
        
        dx = last_pos[0] - first_pos[0]
        dy = last_pos[1] - first_pos[1]
        
        # Normalize by number of frames
        num_frames = len(recent_positions) - 1
        if num_frames > 0:
            dx /= num_frames
            dy /= num_frames
        
        return (dx, dy)
    
    def get_track_direction(self, track_id: int) -> str:
        """
        Get track movement direction
        Args:
            track_id: Track ID
        Returns:
            Direction string ('left', 'right', 'up', 'down', 'stationary')
        """
        dx, dy = self.get_track_velocity(track_id)
        
        # Threshold for movement
        threshold = 2.0
        
        if abs(dx) < threshold and abs(dy) < threshold:
            return 'stationary'
        elif abs(dx) > abs(dy):
            return 'right' if dx > 0 else 'left'
        else:
            return 'down' if dy > 0 else 'up'
    
    def cleanup_old_tracks(self):
        """
        Clean up old tracks from history
        """
        current_frame = self.frame_count
        max_age_frames = self.max_age * 2  # Keep some extra history
        
        tracks_to_remove = []
        for track_id, history in self.track_history.items():
            if history and current_frame - history[-1]['frame'] > max_age_frames:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.track_history[track_id]
    
    def get_tracker_stats(self) -> Dict:
        """
        Get tracker statistics
        Returns:
            Tracker statistics
        """
        return {
            'frame_count': self.frame_count,
            'active_tracks': len(self.track_history),
            'max_age': self.max_age,
            'min_hits': self.min_hits
        }

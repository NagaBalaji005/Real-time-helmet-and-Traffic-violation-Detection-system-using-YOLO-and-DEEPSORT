"""
Speed Estimation Module for Traffic Violation Detection
"""

import math
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
from config import SPEED_CONFIG, SPEED_LIMITS

class SpeedEstimator:
    def __init__(self, pixels_per_meter: float = None, fps: float = 30):
        """
        Initialize speed estimator
        Args:
            pixels_per_meter: Calibration factor (pixels per meter)
            fps: Frames per second of video
        """
        self.pixels_per_meter = pixels_per_meter or SPEED_CONFIG['pixels_per_meter']
        self.fps = fps
        self.smoothing_frames = SPEED_CONFIG['smoothing_frames']
        self.min_track_frames = SPEED_CONFIG['min_track_frames']
        
        # Track speed history
        self.speed_history = {}
        self.calibration_points = []
        
        print(f"Speed estimator initialized: {self.pixels_per_meter} pixels/meter, {self.fps} FPS")
    
    def calibrate_camera(self, point1: Tuple[float, float], point2: Tuple[float, float], 
                        real_distance: float) -> float:
        """
        Calibrate camera using known distance
        Args:
            point1: First point (x, y) in pixels
            point2: Second point (x, y) in pixels
            real_distance: Real distance in meters
        Returns:
            Calibrated pixels per meter
        """
        pixel_distance = math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
        self.pixels_per_meter = pixel_distance / real_distance
        
        print(f"Camera calibrated: {self.pixels_per_meter:.2f} pixels/meter")
        return self.pixels_per_meter
    
    def add_calibration_point(self, point: Tuple[float, float], real_distance: float):
        """
        Add calibration point for more accurate calibration
        Args:
            point: Point coordinates (x, y)
            real_distance: Distance from camera in meters
        """
        self.calibration_points.append((point, real_distance))
    
    def calculate_speed(self, track_history: List[Dict], track_id: int) -> Optional[float]:
        """
        Calculate speed for a track
        Args:
            track_history: Track history from tracker
            track_id: Track ID
        Returns:
            Speed in km/h or None if insufficient data
        """
        if len(track_history) < self.min_track_frames:
            return None
        
        # Get recent positions
        recent_positions = track_history[-self.smoothing_frames:] if len(track_history) >= self.smoothing_frames else track_history
        
        if len(recent_positions) < 2:
            return None
        
        # Calculate total distance traveled
        total_distance_pixels = 0
        for i in range(1, len(recent_positions)):
            pos1 = self._get_centroid(recent_positions[i-1]['bbox'])
            pos2 = self._get_centroid(recent_positions[i]['bbox'])
            
            distance = math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
            total_distance_pixels += distance
        
        # Convert to meters
        total_distance_meters = total_distance_pixels / self.pixels_per_meter
        
        # Calculate time (frames to seconds)
        time_seconds = (len(recent_positions) - 1) / self.fps
        
        if time_seconds <= 0:
            return None
        
        # Calculate speed in m/s
        speed_mps = total_distance_meters / time_seconds
        
        # Convert to km/h
        speed_kmh = speed_mps * 3.6
        
        # Store in history for smoothing
        if track_id not in self.speed_history:
            self.speed_history[track_id] = deque(maxlen=10)
        
        self.speed_history[track_id].append(speed_kmh)
        
        # Return smoothed speed
        return self._smooth_speed(track_id)
    
    def _get_centroid(self, bbox: List[float]) -> Tuple[float, float]:
        """
        Get centroid of bounding box
        Args:
            bbox: [x1, y1, x2, y2]
        Returns:
            Centroid (x, y)
        """
        x1, y1, x2, y2 = bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)
    
    def _smooth_speed(self, track_id: int) -> float:
        """
        Smooth speed using moving average
        Args:
            track_id: Track ID
        Returns:
            Smoothed speed
        """
        if track_id not in self.speed_history or not self.speed_history[track_id]:
            return 0.0
        
        speeds = list(self.speed_history[track_id])
        return sum(speeds) / len(speeds)
    
    def detect_speed_violation(self, speed: float, zone_type: str = 'default') -> Dict:
        """
        Detect speed violation
        Args:
            speed: Current speed in km/h
            zone_type: Type of zone ('urban', 'highway', 'school_zone', 'default')
        Returns:
            Violation information
        """
        speed_limit = SPEED_LIMITS.get(zone_type, SPEED_LIMITS['default'])
        
        violation = {
            'is_violation': speed > speed_limit,
            'current_speed': speed,
            'speed_limit': speed_limit,
            'excess_speed': max(0, speed - speed_limit),
            'zone_type': zone_type
        }
        
        return violation
    
    def get_speed_statistics(self, track_id: int) -> Dict:
        """
        Get speed statistics for a track
        Args:
            track_id: Track ID
        Returns:
            Speed statistics
        """
        if track_id not in self.speed_history or not self.speed_history[track_id]:
            return {
                'current_speed': 0.0,
                'max_speed': 0.0,
                'avg_speed': 0.0,
                'speed_samples': 0
            }
        
        speeds = list(self.speed_history[track_id])
        
        return {
            'current_speed': speeds[-1] if speeds else 0.0,
            'max_speed': max(speeds) if speeds else 0.0,
            'avg_speed': sum(speeds) / len(speeds) if speeds else 0.0,
            'speed_samples': len(speeds)
        }
    
    def estimate_speed_from_velocity(self, velocity: Tuple[float, float]) -> float:
        """
        Estimate speed from velocity vector
        Args:
            velocity: Velocity vector (dx, dy) in pixels per frame
        Returns:
            Speed in km/h
        """
        dx, dy = velocity
        pixel_speed = math.sqrt(dx**2 + dy**2)
        
        # Convert to meters per second
        speed_mps = (pixel_speed * self.fps) / self.pixels_per_meter
        
        # Convert to km/h
        return speed_mps * 3.6
    
    def cleanup_old_tracks(self, active_track_ids: List[int]):
        """
        Clean up speed history for inactive tracks
        Args:
            active_track_ids: List of currently active track IDs
        """
        tracks_to_remove = []
        for track_id in self.speed_history:
            if track_id not in active_track_ids:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.speed_history[track_id]
    
    def get_estimator_info(self) -> Dict:
        """
        Get speed estimator information
        Returns:
            Estimator information
        """
        return {
            'pixels_per_meter': self.pixels_per_meter,
            'fps': self.fps,
            'smoothing_frames': self.smoothing_frames,
            'min_track_frames': self.min_track_frames,
            'active_tracks': len(self.speed_history)
        }

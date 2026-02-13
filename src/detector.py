"""
YOLOv8 Detector for Traffic Violation Detection
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple
import torch

# Fix for PyTorch 2.6 compatibility - patch torch.load to allow loading YOLOv8 weights
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
                # Use weights_only=False for PyTorch 2.6+ compatibility
                return torch.load(file, map_location='cpu', weights_only=False), file
        except Exception as e:
            # Fallback to original function if patch fails
            return original_torch_safe_load(weight)
    
    # Apply the patch
    import ultralytics.nn.tasks
    ultralytics.nn.tasks.torch_safe_load = patched_torch_safe_load
except Exception as e:
    pass  # Silently fail if patch can't be applied

from ultralytics import YOLO

try:
    from .config import DETECTION_CONFIG, CLASSES
except ImportError:
    from config import DETECTION_CONFIG, CLASSES

class Detector:
    def __init__(self, model_path: str = None, device: str = 'auto'):
        """
        Initialize YOLOv8 detector
        Args:
            model_path: Path to trained YOLO model
            device: Device to run inference on ('cpu', 'cuda', 'auto')
        """
        self.device = device
        self.model_size = DETECTION_CONFIG['model_size']
        self.conf_threshold = DETECTION_CONFIG['confidence_threshold']
        self.nms_threshold = DETECTION_CONFIG['nms_threshold']
        
        # Load model
        if model_path and model_path.endswith('.pt'):
            self.model = YOLO(model_path)
        else:
            # Use trained model if available, otherwise pretrained
            try:
                self.model = YOLO('runs/train/final_model_refined/weights/best.pt')
                print("✅ Loaded trained traffic violation model")
            except:
                try:
                    self.model = YOLO('models/traffic_violations_best.pt')
                    print("✅ Loaded trained traffic violation model (fallback)")
                except:
                    self.model = YOLO('yolov8s.pt')
                    print("⚠️ Using pretrained model (trained model not found)")
        
        # Set device
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        print(f"Detector initialized on device: {self.device}")
    
    def predict(self, image: np.ndarray) -> List[Dict]:
        """
        Run detection on image
        Args:
            image: Input image (BGR format)
        Returns:
            List of detection dictionaries
        """
        # Run YOLO inference
        results = self.model.predict(
            image,
            imgsz=self.model_size,
            conf=self.conf_threshold,
            iou=self.nms_threshold,
            device=self.device,
            verbose=False
        )
        
        detections = []
        
        if len(results) > 0:
            result = results[0]
            
            # Extract detections
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for i in range(len(boxes)):
                    detection = {
                        'bbox': boxes[i].tolist(),
                        'confidence': float(confidences[i]),
                        'class_id': int(class_ids[i]),
                        'class_name': CLASSES.get(int(class_ids[i]), f'class_{int(class_ids[i])}')
                    }
                    detections.append(detection)
        
        return detections
    
    def filter_detections(self, detections: List[Dict], class_names: List[str] = None) -> List[Dict]:
        """
        Filter detections by class names
        Args:
            detections: List of detections
            class_names: List of class names to keep
        Returns:
            Filtered detections
        """
        if class_names is None:
            return detections
        
        filtered = []
        for det in detections:
            if det['class_name'] in class_names:
                filtered.append(det)
        
        return filtered
    
    def get_detections_by_class(self, detections: List[Dict], class_name: str) -> List[Dict]:
        """
        Get detections for specific class
        Args:
            detections: List of detections
            class_name: Class name to filter by
        Returns:
            Filtered detections
        """
        return [det for det in detections if det['class_name'] == class_name]
    
    def get_best_detection(self, detections: List[Dict], class_name: str) -> Dict:
        """
        Get detection with highest confidence for specific class
        Args:
            detections: List of detections
            class_name: Class name to filter by
        Returns:
            Best detection or None
        """
        class_detections = self.get_detections_by_class(detections, class_name)
        
        if not class_detections:
            return None
        
        return max(class_detections, key=lambda x: x['confidence'])
    
    def detect_helmet_violation(self, detections: List[Dict]) -> List[Dict]:
        """
        Detect riders without helmets (using no_helmet class directly)
        Args:
            detections: List of detections
        Returns:
            List of violations
        """
        violations = []
        no_helmet_detections = self.get_detections_by_class(detections, 'no_helmet')
        
        for detection in no_helmet_detections:
            violation = {
                'type': 'no_helmet',
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'timestamp': None  # Will be set by pipeline
            }
            violations.append(violation)
        
        return violations
    
    def detect_phone_violation(self, detections: List[Dict]) -> List[Dict]:
        """
        Detect mobile phone usage violations (using mobile_usage class directly)
        Args:
            detections: List of detections
        Returns:
            List of violations
        """
        violations = []
        mobile_detections = self.get_detections_by_class(detections, 'mobile_usage')
        
        for detection in mobile_detections:
            violation = {
                'type': 'mobile_usage',
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'timestamp': None
            }
            violations.append(violation)
        
        return violations
    
    def detect_traffic_violation(self, detections: List[Dict]) -> List[Dict]:
        """
        Detect traffic signal violations (using traffic_violation class directly)
        Args:
            detections: List of detections
        Returns:
            List of violations
        """
        violations = []
        traffic_violations = self.get_detections_by_class(detections, 'traffic_violation')
        
        for detection in traffic_violations:
            violation = {
                'type': 'traffic_violation',
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'timestamp': None
            }
            violations.append(violation)
        
        return violations
    
    def detect_overspeed_violation(self, detections: List[Dict]) -> List[Dict]:
        """
        Detect overspeed violations (using overspeed class directly)
        Args:
            detections: List of detections
        Returns:
            List of violations
        """
        violations = []
        overspeed_detections = self.get_detections_by_class(detections, 'overspeed')
        
        for detection in overspeed_detections:
            violation = {
                'type': 'overspeed',
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'timestamp': None
            }
            violations.append(violation)
        
        return violations
    
    def get_number_plates(self, detections: List[Dict]) -> List[Dict]:
        """
        Get number plate detections
        Args:
            detections: List of detections
        Returns:
            List of number plate detections
        """
        return self.get_detections_by_class(detections, 'number_plate')
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """
        Calculate IoU between two bounding boxes
        Args:
            box1: [x1, y1, x2, y2]
            box2: [x1, y1, x2, y2]
        Returns:
            IoU value
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
    
    def get_model_info(self) -> Dict:
        """
        Get model information
        Returns:
            Model info dictionary
        """
        return {
            'model_path': self.model.ckpt_path if hasattr(self.model, 'ckpt_path') else 'pretrained',
            'device': self.device,
            'model_size': self.model_size,
            'conf_threshold': self.conf_threshold,
            'nms_threshold': self.nms_threshold
        }

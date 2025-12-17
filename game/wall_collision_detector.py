"""Ball detector using YOLO after Aruco transformation"""
import cv2
import numpy as np
from typing import Optional, Dict, Tuple
from game.config import Config
from game.logger import get_logger
from game.aruco_transform import ArucoTransform

logger = get_logger()

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("YOLO not available. Install with: pip install ultralytics")


class WallCollisionDetector:
    """
    Detects ball using YOLO after Aruco transformation.
    
    Detects: sports ball, apple, orange
    """
    
    def __init__(self, aruco_transform: Optional[ArucoTransform] = None):
        """
        Initialize ball detector
        
        Args:
            aruco_transform: ArucoTransform instance for perspective correction
        """
        self.aruco_transform = aruco_transform
        
        # YOLO model (lazy loading)
        self.yolo_model = None
        self.yolo_available = YOLO_AVAILABLE
        
        # Ball tracking state
        self.ball_position = None  # (x, y) in game coordinates
        self.last_ball_position = None
        
        # YOLO parameters
        self.yolo_confidence_threshold = 0.25
        self.target_classes = ['sports ball', 'ball', 'apple', 'orange']
        self.target_class_ids = [32, 47, 49]  # sports ball, apple, orange in COCO dataset
    
    def get_yolo_model(self):
        """Get or load YOLO model (lazy loading)"""
        if not self.yolo_available:
            return None
        
        if self.yolo_model is None:
            try:
                self.yolo_model = YOLO('yolov8n.pt')
                logger.info("YOLOv8 nano model loaded for ball detection")
            except Exception as e:
                logger.warning(f"Failed to load YOLO model: {e}")
                self.yolo_available = False
                return None
        
        return self.yolo_model
    
    def detect_ball_yolo(self, transformed_frame: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """
        Detect ball using YOLO on transformed frame.
        
        Args:
            transformed_frame: Frame after Aruco transformation (game space)
        
        Returns:
            (x, y, radius) of detected ball or None
        """
        model = self.get_yolo_model()
        if model is None:
            return None
        
        try:
            results = model(transformed_frame, verbose=False, conf=self.yolo_confidence_threshold)
            
            best_detection = None
            best_confidence = 0.0
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    confidence = float(box.conf[0])
                    
                    # Check if it's one of our target classes
                    if (class_name in self.target_classes or 
                        cls in self.target_class_ids):
                        
                        if confidence > best_confidence:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                            
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            width = x2 - x1
                            height = y2 - y1
                            radius = int((width + height) / 4)
                            
                            best_detection = (center_x, center_y, radius)
                            best_confidence = confidence
            
            return best_detection
        except Exception as e:
            logger.debug(f"YOLO detection error: {e}")
            return None
    
    def detect_collision(
        self, 
        frame: np.ndarray,
        game_objects: Optional[Dict] = None
    ) -> Dict:
        """
        Detect ball using YOLO after Aruco transformation.
        
        Args:
            frame: Camera frame (BGR)
            game_objects: Not used (kept for compatibility)
        
        Returns:
            Dictionary with detection results:
                - ball_detected: bool
                - ball_position: (x, y) or None
                - transformed_frame: Transformed frame (if Aruco available)
                - original_frame: Original camera frame
        """
        original_frame = frame.copy()
        transformed_frame = frame.copy()
        
        # Apply Aruco transformation if available
        if self.aruco_transform and self.aruco_transform.calibrated:
            # Use stored calibration positions for transform
            if self.aruco_transform.calibration_marker_positions is not None:
                calib = self.aruco_transform.calibration_marker_positions
                game_width = calib['game_screen_width']
                game_height = calib['game_screen_height']
                
                top_left = calib['top_left']
                top_right = calib['top_right']
                bottom_right = calib['bottom_right']
                bottom_left = calib['bottom_left']
                
                # Destination points (output rectangle - game screen size)
                dst_points = np.array([
                    [0, 0],           # Top-left
                    [game_width, 0],           # Top-right
                    [game_width, game_height],           # Bottom-right
                    [0, game_height]            # Bottom-left
                ], dtype=np.float32)
                
                # Source points (Aruco marker corners from camera view)
                src_points = np.array([
                    top_left,         # Top-left
                    top_right,        # Top-right
                    bottom_right,     # Bottom-right
                    bottom_left       # Bottom-left
                ], dtype=np.float32)
                
                # Calculate perspective transform matrix
                matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                
                # Apply transformation to original camera frame
                transformed_frame = cv2.warpPerspective(frame, matrix, (game_width, game_height))
        
        # Detect ball using YOLO on transformed frame
        ball_detection = self.detect_ball_yolo(transformed_frame)
        
        result = {
            'ball_detected': False,
            'ball_position': None,
            'transformed_frame': transformed_frame,
            'original_frame': original_frame
        }
        
        if ball_detection is None:
            # No ball detected, reset tracking
            self.last_ball_position = self.ball_position
            self.ball_position = None
            return result
        
        x, y, radius = ball_detection
        result['ball_detected'] = True
        result['ball_position'] = (x, y)
        
        # Update ball tracking
        self.last_ball_position = self.ball_position
        self.ball_position = (x, y)
        
        return result
    
    def reset(self):
        """Reset detector state"""
        self.ball_position = None
        self.last_ball_position = None

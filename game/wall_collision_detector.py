"""Ball detector using custom YOLO model after Aruco transformation"""
import cv2
import numpy as np
import threading
import queue
from typing import Optional, Dict, Tuple
from pathlib import Path
from game.config import Config
from game.logger import get_logger
from game.aruco_transform import ArucoTransform

logger = get_logger()

# Try to import YOLO
try:
    from ultralytics import YOLO
    import torch
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    torch = None
    logger.warning("ultralytics not available. YOLO detection will not work.")


class WallCollisionDetector:
    """
    Detects ball using custom YOLO model after Aruco transformation.
    
    Uses trained YOLO model for robust ball detection with optimizations.
    """
    
    def __init__(self, aruco_transform: Optional[ArucoTransform] = None):
        """
        Initialize ball detector
        
        Args:
            aruco_transform: ArucoTransform instance for perspective correction
        """
        self.aruco_transform = aruco_transform
        
        # Ball tracking state
        self.ball_position = None  # (x, y) in game coordinates
        self.last_ball_position = None
        
        # Detection persistence: store detections that disappeared but should still exist for 0.5 seconds
        self.persistent_detections = []  # List of {'position_game': (x, y), 'confidence': float, 'time_left': float}
        self.detection_persistence_time = 0.5  # seconds
        
        # YOLO model (lazy loading)
        self.yolo_model = None
        self.yolo_model_path = Path("models/best.pt")
        
        # YOLO optimization settings
        self.yolo_device = 'cuda' if torch and torch.cuda.is_available() else 'cpu'
        self.yolo_half = False  # FP16 - enable if GPU available and model supports it
        self.yolo_imgsz = Config.YOLO_IMAGE_SIZE  # Input image size from config (320, 416, 640)
        self.yolo_conf = Config.YOLO_CONFIDENCE_THRESHOLD  # Confidence threshold from config
        self.yolo_max_det = Config.YOLO_MAX_DETECTIONS  # Maximum detections per frame
        self.yolo_async = Config.YOLO_ASYNC_PROCESSING  # Use async processing
        
        # Async processing state
        self.async_queue = queue.Queue(maxsize=2)  # Queue for frames to process (max 2 frames)
        self.result_queue = queue.Queue(maxsize=1)  # Queue for results (only latest)
        self.processing_thread = None
        self.processing_active = False
        self.latest_result = None  # Latest detection result (for sync fallback)
        
        if not YOLO_AVAILABLE:
            logger.error("YOLO not available. Ball detection will not work.")
        elif not self.yolo_model_path.exists():
            logger.error(f"YOLO model not found at {self.yolo_model_path}. Ball detection will not work.")
        else:
            logger.info(f"YOLO will use device: {self.yolo_device}, imgsz: {self.yolo_imgsz}, async: {self.yolo_async}")
            
            # Start async processing thread if enabled
            if self.yolo_async:
                self.processing_active = True
                self.processing_thread = threading.Thread(target=self._async_processing_worker, daemon=True)
                self.processing_thread.start()
                logger.info("YOLO async processing thread started")
    
    def get_yolo_model(self):
        """Get or load YOLO model (lazy loading)"""
        if not YOLO_AVAILABLE:
            return None
        
        if self.yolo_model is None:
            try:
                if not self.yolo_model_path.exists():
                    logger.error(f"YOLO model not found at {self.yolo_model_path}")
                    return None
                
                self.yolo_model = YOLO(str(self.yolo_model_path))
                
                # Try to enable FP16 if GPU is available
                if self.yolo_device == 'cuda' and torch and torch.cuda.is_available():
                    try:
                        # Test if model supports half precision
                        self.yolo_half = True
                        logger.info("FP16 (half precision) enabled for YOLO")
                    except Exception as e:
                        logger.debug(f"FP16 not available: {e}")
                        self.yolo_half = False
                
                logger.info(f"Custom YOLO model loaded from {self.yolo_model_path}")
                logger.info(f"YOLO device: {self.yolo_device}, half: {self.yolo_half}, imgsz: {self.yolo_imgsz}")
            except Exception as e:
                logger.error(f"Failed to load YOLO model: {e}")
                return None
        
        return self.yolo_model
    
    def _mask_ui_panel(self, frame: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        Create a mask to exclude UI panel area from detection.
        UI panel is on the right side of the screen.
        
        Args:
            frame: Camera frame (BGR)
        
        Returns:
            Tuple of (masked_frame, ui_panel_start_x):
            - masked_frame: Frame with UI panel area blacked out
            - ui_panel_start_x: X coordinate where UI panel starts (for filtering detections)
        """
        masked_frame = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        
        # Calculate UI panel width in camera coordinates
        # UI panel is a percentage of screen width, but we need to map it to camera frame
        # Assuming camera sees the full game screen, UI panel is on the right
        ui_panel_width_percent = Config.UI_PANEL_WIDTH_PERCENT / 100.0
        ui_panel_start_x = int(frame_width * (1 - ui_panel_width_percent))
        
        # Black out the UI panel area
        masked_frame[:, ui_panel_start_x:] = 0
        
        return masked_frame, ui_panel_start_x
    
    def _is_in_ui_panel(self, x: int, frame_width: int) -> bool:
        """Check if x coordinate is in UI panel area"""
        ui_panel_width_percent = Config.UI_PANEL_WIDTH_PERCENT / 100.0
        ui_panel_start_x = int(frame_width * (1 - ui_panel_width_percent))
        return x >= ui_panel_start_x
    
    def detect_ball_yolo(self, frame: np.ndarray, mask_ui_panel: bool = True) -> Tuple[Optional[Tuple[int, int, float]], list]:
        """
        Detect ball using YOLO model on frame (before Aruco transformation).
        
        Args:
            frame: Camera frame (BGR) - original frame before transformation
            mask_ui_panel: Whether to mask UI panel area (default: True)
        
        Returns:
            Tuple of (best_detection, all_detections):
            - best_detection: (x, y, confidence) of detected ball center or None (in original frame coordinates)
            - all_detections: List of all detections for visualization (in original frame coordinates)
        """
        model = self.get_yolo_model()
        if model is None:
            return None, []
        
        try:
            # Mask UI panel area if requested
            if mask_ui_panel:
                detection_frame, ui_panel_start_x = self._mask_ui_panel(frame)
                frame_width = frame.shape[1]
            else:
                detection_frame = frame
                ui_panel_start_x = frame.shape[1]  # No masking
                frame_width = frame.shape[1]
            
            # Run YOLO inference with optimizations
            # Only call once and reuse results
            results = model(
                detection_frame,
                imgsz=self.yolo_imgsz,
                conf=self.yolo_conf,
                device=self.yolo_device,
                half=self.yolo_half,
                max_det=self.yolo_max_det,  # Limit number of detections for speed
                verbose=False
            )
            
            if not results or len(results) == 0:
                return None, []
            
            # Get detections from first result
            result = results[0]
            
            if result.boxes is None or len(result.boxes) == 0:
                return None, []
            
            # Process all detections, filtering out UI panel area
            all_detections = []
            best_detection = None
            best_confidence = 0.0
            
            for box in result.boxes:
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                
                # Skip detections in UI panel area
                if mask_ui_panel and self._is_in_ui_panel(center_x, frame_width):
                    continue
                
                # Add to all detections for visualization
                all_detections.append({
                    'bbox': (int(x1), int(y1), int(x2), int(y2)),
                    'confidence': confidence,
                    'center': (center_x, center_y)
                })
                
                # Track best detection (excluding UI panel)
                if confidence > best_confidence:
                    best_detection = (center_x, center_y, confidence)
                    best_confidence = confidence
            
            return best_detection, all_detections
            
        except Exception as e:
            logger.debug(f"Error in YOLO detection: {e}")
            return None, []
    
    def _async_processing_worker(self):
        """Background thread worker for async YOLO processing"""
        model = self.get_yolo_model()
        if model is None:
            return
        
        while self.processing_active:
            try:
                # Get frame from queue (with timeout to allow checking processing_active)
                try:
                    frame_data = self.async_queue.get(timeout=0.1)
                    if frame_data is None:  # Shutdown signal
                        break
                    
                    frame, mask_ui_panel = frame_data
                    
                    # Run YOLO detection
                    best_detection, all_detections = self.detect_ball_yolo(frame, mask_ui_panel)
                    
                    # Put result in result queue (replace old result if queue is full)
                    try:
                        self.result_queue.put_nowait((best_detection, all_detections))
                    except queue.Full:
                        # Remove old result and add new one
                        try:
                            self.result_queue.get_nowait()
                        except queue.Empty:
                            pass
                        self.result_queue.put_nowait((best_detection, all_detections))
                    
                except queue.Empty:
                    continue  # No frame to process, check again
                    
            except Exception as e:
                logger.debug(f"Error in async processing worker: {e}")
                continue
    
    def detect_collision(
        self, 
        frame: np.ndarray,
        game_objects: Optional[Dict] = None,
        dt: float = 0.0
    ) -> Dict:
        """
        Detect ball using YOLO model before Aruco transformation.
        
        Args:
            frame: Camera frame (BGR)
            game_objects: Not used (kept for compatibility)
        
        Returns:
            Dictionary with detection results:
                - ball_detected: bool
                - ball_position: (x, y) in original camera coordinates or None
                - transformed_frame: Transformed frame (if Aruco available)
                - original_frame: Original camera frame
                - yolo_detections: List of all YOLO detections for visualization (in original coordinates)
        """
        original_frame = frame.copy()
        transformed_frame = frame.copy()
        transform_matrix = None
        
        # Prepare Aruco transformation if available (for visualization)
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
                transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
                
                # Apply transformation to original camera frame (for visualization only)
                transformed_frame = cv2.warpPerspective(frame, transform_matrix, (game_width, game_height))
        
        # Detect ball using YOLO on original frame (before transformation)
        # Use async processing if enabled, otherwise sync
        if self.yolo_async and self.processing_active:
            # Try to get latest result from async queue
            try:
                ball_detection, yolo_detections = self.result_queue.get_nowait()
                self.latest_result = (ball_detection, yolo_detections)
            except queue.Empty:
                # No result yet, use latest result if available
                if self.latest_result is not None:
                    ball_detection, yolo_detections = self.latest_result
                else:
                    ball_detection, yolo_detections = None, []
            
            # Add current frame to processing queue (skip if queue is full)
            try:
                self.async_queue.put_nowait((original_frame.copy(), True))
            except queue.Full:
                # Queue is full, skip this frame (processing is slower than frame rate)
                pass
        else:
            # Synchronous processing
            ball_detection, yolo_detections = self.detect_ball_yolo(original_frame, mask_ui_panel=True)
        
        # Get game dimensions for UI panel visualization
        game_width = None
        game_height = None
        if self.aruco_transform and self.aruco_transform.calibrated:
            if self.aruco_transform.calibration_marker_positions is not None:
                calib = self.aruco_transform.calibration_marker_positions
                game_width = calib['game_screen_width']
                game_height = calib['game_screen_height']
        
        # Update persistent detections (decrease time left)
        for det in self.persistent_detections:
            det['time_left'] -= dt
        
        # Remove expired persistent detections
        self.persistent_detections = [d for d in self.persistent_detections if d['time_left'] > 0]
        
        # Transform all detections to game coordinates and filter out UI panel area
        all_detections_game = []
        ui_panel_start_x = None
        if transform_matrix is not None and yolo_detections and game_width:
            ui_panel_width_percent = Config.UI_PANEL_WIDTH_PERCENT / 100.0
            ui_panel_start_x = int(game_width * (1 - ui_panel_width_percent))
            
            for det in yolo_detections:
                center_x, center_y = det['center']
                # Transform point from camera coordinates to game coordinates
                point_camera = np.array([[[center_x, center_y]]], dtype=np.float32)
                point_game = cv2.perspectiveTransform(point_camera, transform_matrix)[0][0]
                game_x = int(point_game[0])
                game_y = int(point_game[1])
                
                # Filter out detections in UI panel area (in game coordinates)
                if game_x < ui_panel_start_x:  # Only add if not in UI panel
                    all_detections_game.append({
                        'position_game': (game_x, game_y),
                        'confidence': det['confidence']
                    })
        
        # Add persistent detections that are still valid
        if ui_panel_start_x is not None:
            for det in self.persistent_detections:
                game_x, game_y = det['position_game']
                # Only add if not in UI panel
                if game_x < ui_panel_start_x:
                    all_detections_game.append({
                        'position_game': (game_x, game_y),
                        'confidence': det['confidence']
                    })
        
        result = {
            'ball_detected': False,
            'ball_position': None,  # Position in original camera coordinates (best detection)
            'ball_position_game': None,  # Position in game coordinates (after transformation, best detection)
            'all_detections_game': all_detections_game,  # All detections in game coordinates
            'transformed_frame': transformed_frame,
            'original_frame': original_frame,
            'yolo_detections': yolo_detections,
            'transform_matrix': transform_matrix,
            'game_width': game_width,
            'game_height': game_height
        }
        
        # Check if best detection is in UI panel and filter it out
        best_detection_game = None
        if ball_detection is not None and transform_matrix is not None and game_width:
            x, y, confidence = ball_detection
            # Transform point from camera coordinates to game coordinates
            point_camera = np.array([[[x, y]]], dtype=np.float32)
            point_game = cv2.perspectiveTransform(point_camera, transform_matrix)[0][0]
            game_x = int(point_game[0])
            game_y = int(point_game[1])
            
            ui_panel_width_percent = Config.UI_PANEL_WIDTH_PERCENT / 100.0
            ui_panel_start_x = int(game_width * (1 - ui_panel_width_percent))
            
            # Only use best detection if it's not in UI panel
            if game_x < ui_panel_start_x:
                best_detection_game = (game_x, game_y)
                result['ball_detected'] = True
                result['ball_position'] = (x, y)  # Position in original camera coordinates
                result['ball_position_game'] = best_detection_game
                result['confidence'] = confidence
                
                # Update persistent detections - add current detection
                # Remove old persistent detections and add new one
                self.persistent_detections = [{
                    'position_game': best_detection_game,
                    'confidence': confidence,
                    'time_left': self.detection_persistence_time
                }]
            else:
                # Best detection is in UI panel, use persistent detections if available
                if self.persistent_detections:
                    best_persistent = max(self.persistent_detections, key=lambda d: d['confidence'])
                    result['ball_detected'] = True
                    result['ball_position_game'] = best_persistent['position_game']
                    result['confidence'] = best_persistent['confidence']
                else:
                    # No valid detection, reset tracking
                    self.last_ball_position = self.ball_position
                    self.ball_position = None
        elif ball_detection is None:
            # No new detection - use persistent detections if available
            if self.persistent_detections:
                best_persistent = max(self.persistent_detections, key=lambda d: d['confidence'])
                result['ball_detected'] = True
                result['ball_position_game'] = best_persistent['position_game']
                result['confidence'] = best_persistent['confidence']
            else:
                # No ball detected, reset tracking
                self.last_ball_position = self.ball_position
                self.ball_position = None
        
        # Update ball tracking (in original camera coordinates)
        if result.get('ball_position'):
            self.last_ball_position = self.ball_position
            self.ball_position = result['ball_position']
        
        return result
    
    def reset(self):
        """Reset detector state"""
        self.ball_position = None
        self.last_ball_position = None
    
    def cleanup(self):
        """Cleanup resources and stop async processing"""
        if self.yolo_async and self.processing_active:
            self.processing_active = False
            # Signal shutdown
            try:
                self.async_queue.put_nowait(None)
            except queue.Full:
                pass
            # Wait for thread to finish (with timeout)
            if self.processing_thread and self.processing_thread.is_alive():
                self.processing_thread.join(timeout=1.0)
            logger.info("YOLO async processing stopped")

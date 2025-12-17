"""Hybrid snowball detector using Hough Circles + YOLO for tracking"""
import sys
import os

# Add parent directory to path to import game modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from game.config import Config
from game.logger import get_logger

logger = get_logger()

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("YOLO not available. Install with: pip install ultralytics")


class HybridSnowballDetector:
    """Hybrid detector using Hough Circles (primary) + YOLO (fallback tracking)"""
    
    def __init__(self):
        # YOLO model (lazy loading)
        self.yolo_model = None
        self.yolo_available = YOLO_AVAILABLE
        
        # Tracking state
        self.last_detection = None  # (x, y, radius, confidence, frame_count)
        self.lost_frames = 0
        self.max_lost_frames = 10  # Max frames to keep prediction after loss
        self.prediction_confidence = 0.0
        
        # Kalman filter for smoothing and prediction
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                   [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
        self.kalman.measurementNoiseCov = 0.1 * np.eye(2, dtype=np.float32)
        self.kalman.statePre = np.array([0, 0, 0, 0], dtype=np.float32)
        self.kalman.statePost = np.array([0, 0, 0, 0], dtype=np.float32)
        self.kalman_initialized = False
        
        # Hough Circles parameters (tuned for noise reduction)
        self.hough_params = {
            'dp': 1.2,           # Inverse ratio of accumulator resolution
            'minDist': 80,       # Minimum distance between circle centers (increased to reduce noise)
            'param1': 50,       # Upper threshold for edge detection
            'param2': 30,        # Accumulator threshold (higher = fewer false positives)
            'minRadius': 15,     # Minimum circle radius (increased to filter small noise)
            'maxRadius': 100     # Maximum circle radius
        }
        
        # YOLO parameters
        self.yolo_confidence_threshold = 0.2
        self.yolo_check_interval = 3  # Check YOLO every N frames when tracking is lost
    
    def get_yolo_model(self):
        """Get or load YOLO model (lazy loading)"""
        if not self.yolo_available:
            return None
        
        if self.yolo_model is None:
            try:
                self.yolo_model = YOLO('yolov8n.pt')
                logger.info("YOLOv8 nano model loaded for hybrid detection")
            except Exception as e:
                logger.warning(f"Failed to load YOLO model: {e}")
                self.yolo_available = False
                return None
        
        return self.yolo_model
    
    def detect_hough_circles(self, frame):
        """Primary detection method: Hough Circles with noise filtering"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (9, 9), 2)
        
        # Apply median blur for additional noise reduction
        blurred = cv2.medianBlur(blurred, 5)
        
        # Detect circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            **self.hough_params
        )
        
        detections = []
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            for (x, y, r) in circles[0, :]:
                # Additional filtering: check if the circle region is actually round
                # by examining the area around the circle
                y1 = max(0, y - r)
                y2 = min(frame.shape[0], y + r)
                x1 = max(0, x - r)
                x2 = min(frame.shape[1], x + r)
                
                if y2 > y1 and x2 > x1:
                    # Extract circle region
                    circle_region = gray[y1:y2, x1:x2]
                    
                    # Create mask for circle
                    mask = np.zeros(circle_region.shape, dtype=np.uint8)
                    center_mask = (circle_region.shape[1] // 2, circle_region.shape[0] // 2)
                    cv2.circle(mask, center_mask, min(r, min(circle_region.shape) // 2), 255, -1)
                    
                    # Check if region is reasonably uniform (ball should be relatively uniform)
                    if mask.sum() > 0:
                        masked_region = cv2.bitwise_and(circle_region, mask)
                        non_zero = masked_region[masked_region > 0]
                        
                        if len(non_zero) > 0:
                            # Calculate standard deviation - balls should have moderate variation
                            std_dev = np.std(non_zero)
                            mean_val = np.mean(non_zero)
                            
                            # Filter out regions that are too uniform (likely noise) or too varied
                            # Good balls have moderate variation
                            if 10 < std_dev < 80 and 50 < mean_val < 250:
                                detections.append({
                                    'center': (int(x), int(y)),
                                    'radius': int(r),
                                    'confidence': 1.0,
                                    'method': 'hough',
                                    'std_dev': std_dev,
                                    'mean_val': mean_val
                                })
        
        return detections
    
    def detect_yolo(self, frame):
        """Fallback detection method: YOLO for object detection"""
        model = self.get_yolo_model()
        if model is None:
            return []
        
        try:
            results = model(frame, verbose=False, conf=self.yolo_confidence_threshold)
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    confidence = float(box.conf[0])
                    
                    # Filter for ball-like objects
                    if (class_name in ['sports ball', 'ball'] or 
                        (cls == 32 and confidence > self.yolo_confidence_threshold)):
                        
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        width = x2 - x1
                        height = y2 - y1
                        radius = int((width + height) / 4)
                        
                        # Check if region is white (optional verification)
                        y1_int = max(0, y1)
                        y2_int = min(frame.shape[0], y2)
                        x1_int = max(0, x1)
                        x2_int = min(frame.shape[1], x2)
                        
                        if y2_int > y1_int and x2_int > x1_int:
                            ball_region = frame[y1_int:y2_int, x1_int:x2_int]
                            if ball_region.size > 0:
                                gray_region = cv2.cvtColor(ball_region, cv2.COLOR_BGR2GRAY)
                                white_pixels = np.sum(gray_region > 200)
                                total_pixels = gray_region.size
                                white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
                                
                                if white_ratio > 0.25:
                                    detections.append({
                                        'center': (center_x, center_y),
                                        'radius': radius,
                                        'confidence': confidence,
                                        'method': 'yolo',
                                        'white_ratio': white_ratio
                                    })
            
            return detections
        except Exception as e:
            logger.debug(f"YOLO detection error: {e}")
            return []
    
    def update_kalman(self, measurement):
        """Update Kalman filter with new measurement"""
        if not self.kalman_initialized:
            self.kalman.statePre = np.array([measurement[0], measurement[1], 0, 0], dtype=np.float32)
            self.kalman.statePost = np.array([measurement[0], measurement[1], 0, 0], dtype=np.float32)
            self.kalman_initialized = True
        
        # Predict
        prediction = self.kalman.predict()
        
        # Update with measurement
        measurement_array = np.array([[measurement[0]], [measurement[1]]], dtype=np.float32)
        self.kalman.correct(measurement_array)
        
        return prediction
    
    def predict_position(self):
        """Predict position using Kalman filter"""
        if not self.kalman_initialized:
            return None
        
        prediction = self.kalman.predict()
        return (int(prediction[0]), int(prediction[1]))
    
    def detect(self, frame, frame_count=0):
        """
        Main detection method combining Hough Circles and YOLO
        
        Returns:
            dict with 'detection', 'method', 'confidence', 'prediction'
        """
        # Primary: Try Hough Circles
        hough_detections = self.detect_hough_circles(frame)
        
        if hough_detections:
            # Use the detection with highest confidence or best quality
            best_detection = max(hough_detections, key=lambda d: d.get('confidence', 0))
            
            center = best_detection['center']
            radius = best_detection['radius']
            confidence = best_detection['confidence']
            
            # Update Kalman filter
            self.update_kalman(center)
            
            # Update tracking state
            self.last_detection = (center[0], center[1], radius, confidence, frame_count)
            self.lost_frames = 0
            self.prediction_confidence = confidence
            
            return {
                'detection': best_detection,
                'method': 'hough',
                'confidence': confidence,
                'prediction': None,
                'lost': False
            }
        
        # If Hough Circles failed, check if we should use YOLO
        # Use YOLO if:
        # 1. We recently had a detection (within max_lost_frames)
        # 2. It's time to check YOLO (based on interval)
        should_check_yolo = (
            self.last_detection is not None and
            self.lost_frames < self.max_lost_frames and
            frame_count % self.yolo_check_interval == 0
        )
        
        if should_check_yolo and self.yolo_available:
            yolo_detections = self.detect_yolo(frame)
            
            if yolo_detections:
                # Use YOLO detection
                best_detection = max(yolo_detections, key=lambda d: d.get('confidence', 0))
                
                center = best_detection['center']
                radius = best_detection['radius']
                confidence = best_detection['confidence'] * 0.7  # Lower confidence for YOLO
                
                # Update Kalman filter
                self.update_kalman(center)
                
                # Update tracking state
                self.last_detection = (center[0], center[1], radius, confidence, frame_count)
                self.lost_frames = 0
                self.prediction_confidence = confidence
                
                return {
                    'detection': best_detection,
                    'method': 'yolo',
                    'confidence': confidence,
                    'prediction': None,
                    'lost': False
                }
        
        # No detection found - use prediction if we recently had a detection
        if self.last_detection is not None and self.lost_frames < self.max_lost_frames:
            self.lost_frames += 1
            
            # Predict position using Kalman filter
            predicted_center = self.predict_position()
            last_x, last_y, last_radius, last_conf, _ = self.last_detection
            
            if predicted_center:
                # Use predicted position
                predicted_radius = last_radius
                predicted_confidence = max(0.1, self.prediction_confidence * (1 - self.lost_frames / self.max_lost_frames))
                
                return {
                    'detection': {
                        'center': predicted_center,
                        'radius': predicted_radius,
                        'confidence': predicted_confidence,
                        'method': 'prediction'
                    },
                    'method': 'prediction',
                    'confidence': predicted_confidence,
                    'prediction': predicted_center,
                    'lost': True
                }
            else:
                # Fallback to last known position
                return {
                    'detection': {
                        'center': (last_x, last_y),
                        'radius': last_radius,
                        'confidence': max(0.1, last_conf * (1 - self.lost_frames / self.max_lost_frames)),
                        'method': 'last_known'
                    },
                    'method': 'last_known',
                    'confidence': max(0.1, last_conf * (1 - self.lost_frames / self.max_lost_frames)),
                    'prediction': None,
                    'lost': True
                }
        
        # Completely lost
        self.lost_frames += 1
        if self.lost_frames > self.max_lost_frames:
            self.last_detection = None
            self.kalman_initialized = False
        
        return {
            'detection': None,
            'method': 'none',
            'confidence': 0.0,
            'prediction': None,
            'lost': True
        }
    
    def draw_detection(self, frame, result):
        """Draw detection result on frame"""
        if result['detection'] is None:
            return frame
        
        detection = result['detection']
        center = detection['center']
        radius = detection['radius']
        method = result['method']
        confidence = result['confidence']
        is_lost = result['lost']
        
        # Color based on method and state
        if method == 'hough':
            color = (0, 255, 0)  # Green for Hough
        elif method == 'yolo':
            color = (255, 0, 255)  # Magenta for YOLO
        elif method == 'prediction':
            color = (0, 255, 255)  # Yellow for prediction
        else:
            color = (128, 128, 128)  # Gray for last known
        
        # Draw circle (dashed if lost)
        if is_lost:
            # Draw dashed circle
            for angle in range(0, 360, 20):
                x1 = int(center[0] + radius * np.cos(np.radians(angle)))
                y1 = int(center[1] + radius * np.sin(np.radians(angle)))
                x2 = int(center[0] + radius * np.cos(np.radians(angle + 10)))
                y2 = int(center[1] + radius * np.sin(np.radians(angle + 10)))
                cv2.line(frame, (x1, y1), (x2, y2), color, 2)
        else:
            cv2.circle(frame, center, radius, color, 2)
        
        # Draw center point
        cv2.circle(frame, center, 5, color, -1)
        
        # Draw info text
        method_text = f"[{method.upper()}]"
        conf_text = f"Conf: {confidence:.2f}"
        if is_lost:
            method_text += " (LOST)"
        
        y_offset = center[1] - radius - 20
        if y_offset < 20:
            y_offset = center[1] + radius + 20
        
        cv2.putText(frame, method_text, (center[0] - 40, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, conf_text, (center[0] - 40, y_offset + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame


def main():
    """Main function"""
    # Initialize camera
    camera = cv2.VideoCapture(Config.SNOWBALL_CAMERA_INDEX)
    
    if not camera.isOpened():
        logger.error(f"Could not open camera {Config.SNOWBALL_CAMERA_INDEX}")
        return
    
    # Set camera format to MJPEG
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    camera.set(cv2.CAP_PROP_FOURCC, fourcc)
    
    # Set camera resolution
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, Config.SNOWBALL_CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.SNOWBALL_CAMERA_HEIGHT)
    
    # Apply camera quality settings from config
    camera.set(cv2.CAP_PROP_EXPOSURE, Config.SNOWBALL_CAMERA_EXPOSURE)
    camera.set(cv2.CAP_PROP_BRIGHTNESS, Config.SNOWBALL_CAMERA_BRIGHTNESS)
    camera.set(cv2.CAP_PROP_CONTRAST, Config.SNOWBALL_CAMERA_CONTRAST)
    camera.set(cv2.CAP_PROP_SATURATION, Config.SNOWBALL_CAMERA_SATURATION)
    camera.set(cv2.CAP_PROP_SHARPNESS, Config.SNOWBALL_CAMERA_SHARPNESS)
    
    logger.info("Camera initialized. Press 'q' to quit.")
    logger.info("Hybrid detector: Hough Circles (primary) + YOLO (fallback)")
    
    detector = HybridSnowballDetector()
    frame_count = 0
    
    while True:
        ret, frame = camera.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            break
        
        # Detect snowball
        result = detector.detect(frame, frame_count)
        
        # Draw detection
        frame_with_detection = detector.draw_detection(frame.copy(), result)
        
        # Draw status
        status_text = f"Method: {result['method']} | "
        status_text += f"Confidence: {result['confidence']:.2f} | "
        status_text += f"Lost frames: {detector.lost_frames}"
        
        cv2.putText(frame_with_detection, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Resize for display
        h, w = frame_with_detection.shape[:2]
        small_frame = cv2.resize(frame_with_detection, (w // 2, h // 2))
        cv2.imshow("Hybrid Snowball Detector", small_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset detector
            detector.last_detection = None
            detector.lost_frames = 0
            detector.kalman_initialized = False
            logger.info("Detector reset")
        
        frame_count += 1
    
    # Cleanup
    camera.release()
    cv2.destroyAllWindows()
    logger.info("Camera released, windows closed")


if __name__ == "__main__":
    main()


"""Combined Aruco marker detection + Hybrid snowball detector"""
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

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Aruco marker settings
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50
ARUCO_MARKER_IDS = [0, 1, 2, 3]  # Expected marker IDs

# Camera settings (only MJPEG format is used)

# Hough Circles parameters (tuned for noise reduction)
HOUGH_DP = 1.2                    # Inverse ratio of accumulator resolution
HOUGH_MIN_DIST = 80               # Minimum distance between circle centers
HOUGH_PARAM1 = 50                 # Upper threshold for edge detection
HOUGH_PARAM2 = 30                 # Accumulator threshold (higher = fewer false positives)
HOUGH_MIN_RADIUS = 15             # Minimum circle radius
HOUGH_MAX_RADIUS = 100            # Maximum circle radius

# Image preprocessing for Hough Circles
GAUSSIAN_BLUR_SIZE = (9, 9)
GAUSSIAN_BLUR_SIGMA = 2
MEDIAN_BLUR_SIZE = 5

# Circle region filtering (for noise reduction)
MIN_STD_DEV = 10                  # Minimum standard deviation for valid circle
MAX_STD_DEV = 80                  # Maximum standard deviation for valid circle
MIN_MEAN_VAL = 50                 # Minimum mean value for valid circle
MAX_MEAN_VAL = 250                # Maximum mean value for valid circle

# YOLO parameters
YOLO_CONFIDENCE_THRESHOLD = 0.2
YOLO_CHECK_INTERVAL = 3           # Check YOLO every N frames when tracking is lost
YOLO_WHITE_RATIO_THRESHOLD = 0.25 # Minimum white pixel ratio for YOLO detection

# Tracking parameters
MAX_LOST_FRAMES = 10              # Max frames to keep prediction after loss
KALMAN_PROCESS_NOISE = 0.03       # Kalman filter process noise covariance
KALMAN_MEASUREMENT_NOISE = 0.1   # Kalman filter measurement noise covariance

# Display settings
DISPLAY_SCALE_FACTOR = 2          # Resize factor for display (1/2 size)

# ============================================================================
# CLASSES
# ============================================================================

class ArucoSnowballDetector:
    """Combined detector: Aruco markers for perspective transform + Hybrid snowball detection"""
    
    def __init__(self):
        # Aruco detection
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.last_marker_positions = {0: None, 1: None, 2: None, 3: None}
        
        # YOLO model (lazy loading)
        self.yolo_model = None
        self.yolo_available = YOLO_AVAILABLE
        
        # Tracking state
        self.last_detection = None  # (x, y, radius, confidence, frame_count)
        self.lost_frames = 0
        self.prediction_confidence = 0.0
        
        # Kalman filter for smoothing and prediction
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                   [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = KALMAN_PROCESS_NOISE * np.eye(4, dtype=np.float32)
        self.kalman.measurementNoiseCov = KALMAN_MEASUREMENT_NOISE * np.eye(2, dtype=np.float32)
        self.kalman.statePre = np.array([0, 0, 0, 0], dtype=np.float32)
        self.kalman.statePost = np.array([0, 0, 0, 0], dtype=np.float32)
        self.kalman_initialized = False
        
        # Perspective transform matrix (cached)
        self.transform_matrix = None
        self.inverse_transform_matrix = None
        self.transform_valid = False
    
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
    
    def get_closest_corner_to_center(self, corner_points, screen_center):
        """Get the corner of a marker that is closest to the screen center"""
        distances = []
        for corner in corner_points:
            dist = np.sqrt((corner[0] - screen_center[0])**2 + (corner[1] - screen_center[1])**2)
            distances.append(dist)
        
        closest_idx = np.argmin(distances)
        return corner_points[closest_idx]
    
    def detect_aruco_markers(self, frame):
        """Detect Aruco markers in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect markers (new API for OpenCV 4.7+)
        try:
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            corners, ids, rejected = detector.detectMarkers(gray)
        except AttributeError:
            # Fallback to old API (OpenCV < 4.7)
            corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        return corners, ids
    
    def determine_corners(self, corners, ids, screen_center):
        """Determine which marker belongs to which corner"""
        # Get corner points (closest to screen center) of each marker from current frame
        marker_corners = {}
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in ARUCO_MARKER_IDS:
                    # Get corner closest to screen center
                    corner_points = corners[i][0]
                    closest_corner = self.get_closest_corner_to_center(corner_points, screen_center)
                    marker_corners[marker_id] = closest_corner
                    # Update last known position
                    self.last_marker_positions[marker_id] = closest_corner
        
        # Use last known positions for missing markers
        for marker_id in ARUCO_MARKER_IDS:
            if marker_id not in marker_corners and self.last_marker_positions[marker_id] is not None:
                marker_corners[marker_id] = self.last_marker_positions[marker_id]
        
        if len(marker_corners) != 4:
            return None, None, None, None
        
        # Determine corners based on position
        corners_list = [(id, corner) for id, corner in marker_corners.items()]
        
        # Find top-left (minimum x + y)
        top_left_marker = min(corners_list, key=lambda x: x[1][0] + x[1][1])
        
        # Find top-right (maximum x, minimum y)
        top_right_marker = max(corners_list, key=lambda x: x[1][0] - x[1][1])
        
        # Find bottom-right (maximum x + y)
        bottom_right_marker = max(corners_list, key=lambda x: x[1][0] + x[1][1])
        
        # Find bottom-left (minimum x, maximum y)
        bottom_left_marker = min(corners_list, key=lambda x: x[1][0] - x[1][1])
        
        return (top_left_marker[1], top_right_marker[1], 
                bottom_right_marker[1], bottom_left_marker[1])
    
    def update_perspective_transform(self, frame, corners, ids):
        """Update perspective transform matrix based on Aruco markers"""
        h, w = frame.shape[:2]
        screen_center = (w // 2, h // 2)
        
        # Determine corners
        top_left, top_right, bottom_right, bottom_left = self.determine_corners(corners, ids, screen_center)
        
        if any(p is None for p in [top_left, top_right, bottom_right, bottom_left]):
            self.transform_valid = False
            return False
        
        # Destination points (output rectangle)
        dst_points = np.array([
            [0, 0],           # Top-left
            [w, 0],           # Top-right
            [w, h],           # Bottom-right
            [0, h]            # Bottom-left
        ], dtype=np.float32)
        
        # Source points
        src = np.array([
            top_left,         # Top-left
            top_right,        # Top-right
            bottom_right,     # Bottom-right
            bottom_left       # Bottom-left
        ], dtype=np.float32)
        
        # Calculate perspective transform matrix
        self.transform_matrix = cv2.getPerspectiveTransform(src, dst_points)
        self.inverse_transform_matrix = cv2.getPerspectiveTransform(dst_points, src)
        self.transform_valid = True
        
        return True
    
    def transform_to_game_space(self, frame):
        """Transform frame to game space (rectified view)"""
        if not self.transform_valid or self.transform_matrix is None:
            return None
        
        h, w = frame.shape[:2]
        transformed = cv2.warpPerspective(frame, self.transform_matrix, (w, h))
        return transformed
    
    def transform_from_game_space(self, point):
        """Transform point from game space back to camera space"""
        if not self.transform_valid or self.inverse_transform_matrix is None:
            return point
        
        point_array = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point_array, self.inverse_transform_matrix)
        return tuple(transformed[0][0].astype(int))
    
    def detect_hough_circles(self, frame):
        """Primary detection method: Hough Circles with noise filtering"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, GAUSSIAN_BLUR_SIZE, GAUSSIAN_BLUR_SIGMA)
        
        # Apply median blur for additional noise reduction
        blurred = cv2.medianBlur(blurred, MEDIAN_BLUR_SIZE)
        
        # Detect circles
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=HOUGH_DP,
            minDist=HOUGH_MIN_DIST,
            param1=HOUGH_PARAM1,
            param2=HOUGH_PARAM2,
            minRadius=HOUGH_MIN_RADIUS,
            maxRadius=HOUGH_MAX_RADIUS
        )
        
        detections = []
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            
            for (x, y, r) in circles[0, :]:
                # Additional filtering: check if the circle region is actually round
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
                    
                    # Check if region is reasonably uniform
                    if mask.sum() > 0:
                        masked_region = cv2.bitwise_and(circle_region, mask)
                        non_zero = masked_region[masked_region > 0]
                        
                        if len(non_zero) > 0:
                            # Calculate standard deviation and mean
                            std_dev = np.std(non_zero)
                            mean_val = np.mean(non_zero)
                            
                            # Filter out regions that are too uniform or too varied
                            if (MIN_STD_DEV < std_dev < MAX_STD_DEV and 
                                MIN_MEAN_VAL < mean_val < MAX_MEAN_VAL):
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
            results = model(frame, verbose=False, conf=YOLO_CONFIDENCE_THRESHOLD)
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    confidence = float(box.conf[0])
                    
                    # Filter for ball-like objects
                    if (class_name in ['sports ball', 'ball'] or 
                        (cls == 32 and confidence > YOLO_CONFIDENCE_THRESHOLD)):
                        
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        center_x = int((x1 + x2) / 2)
                        center_y = int((y1 + y2) / 2)
                        width = x2 - x1
                        height = y2 - y1
                        radius = int((width + height) / 4)
                        
                        # Check if region is white
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
                                
                                if white_ratio > YOLO_WHITE_RATIO_THRESHOLD:
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
        Main detection method: detect snowballs only in transformed game space
        
        Returns:
            dict with 'detection', 'method', 'confidence', 'prediction', 'lost'
        """
        # First, detect Aruco markers and update transform
        corners, ids = self.detect_aruco_markers(frame)
        self.update_perspective_transform(frame, corners, ids)
        
        # Transform frame to game space
        transformed_frame = self.transform_to_game_space(frame)
        
        if transformed_frame is None:
            # No valid transform - can't detect
            return {
                'detection': None,
                'method': 'no_transform',
                'confidence': 0.0,
                'prediction': None,
                'lost': True
            }
        
        # Detect in transformed space
        hough_detections = self.detect_hough_circles(transformed_frame)
        
        if hough_detections:
            # Use the detection with highest confidence
            best_detection = max(hough_detections, key=lambda d: d.get('confidence', 0))
            
            center = best_detection['center']
            radius = best_detection['radius']
            confidence = best_detection['confidence']
            
            # Transform center back to camera space for display
            camera_center = self.transform_from_game_space(center)
            
            # Update Kalman filter (in game space)
            self.update_kalman(center)
            
            # Update tracking state
            self.last_detection = (center[0], center[1], radius, confidence, frame_count)
            self.lost_frames = 0
            self.prediction_confidence = confidence
            
            return {
                'detection': {
                    **best_detection,
                    'camera_center': camera_center  # Add camera space center for drawing
                },
                'method': 'hough',
                'confidence': confidence,
                'prediction': None,
                'lost': False
            }
        
        # If Hough Circles failed, check if we should use YOLO
        should_check_yolo = (
            self.last_detection is not None and
            self.lost_frames < MAX_LOST_FRAMES and
            frame_count % YOLO_CHECK_INTERVAL == 0
        )
        
        if should_check_yolo and self.yolo_available:
            yolo_detections = self.detect_yolo(transformed_frame)
            
            if yolo_detections:
                best_detection = max(yolo_detections, key=lambda d: d.get('confidence', 0))
                
                center = best_detection['center']
                radius = best_detection['radius']
                confidence = best_detection['confidence'] * 0.7  # Lower confidence for YOLO
                
                # Transform center back to camera space
                camera_center = self.transform_from_game_space(center)
                
                # Update Kalman filter
                self.update_kalman(center)
                
                # Update tracking state
                self.last_detection = (center[0], center[1], radius, confidence, frame_count)
                self.lost_frames = 0
                self.prediction_confidence = confidence
                
                return {
                    'detection': {
                        **best_detection,
                        'camera_center': camera_center
                    },
                    'method': 'yolo',
                    'confidence': confidence,
                    'prediction': None,
                    'lost': False
                }
        
        # No detection found - use prediction if we recently had a detection
        if self.last_detection is not None and self.lost_frames < MAX_LOST_FRAMES:
            self.lost_frames += 1
            
            # Predict position using Kalman filter (in game space)
            predicted_center = self.predict_position()
            last_x, last_y, last_radius, last_conf, _ = self.last_detection
            
            if predicted_center:
                # Transform predicted center to camera space
                camera_predicted = self.transform_from_game_space(predicted_center)
                
                predicted_radius = last_radius
                predicted_confidence = max(0.1, self.prediction_confidence * (1 - self.lost_frames / MAX_LOST_FRAMES))
                
                return {
                    'detection': {
                        'center': predicted_center,
                        'camera_center': camera_predicted,
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
                camera_last = self.transform_from_game_space((last_x, last_y))
                
                return {
                    'detection': {
                        'center': (last_x, last_y),
                        'camera_center': camera_last,
                        'radius': last_radius,
                        'confidence': max(0.1, last_conf * (1 - self.lost_frames / MAX_LOST_FRAMES)),
                        'method': 'last_known'
                    },
                    'method': 'last_known',
                    'confidence': max(0.1, last_conf * (1 - self.lost_frames / MAX_LOST_FRAMES)),
                    'prediction': None,
                    'lost': True
                }
        
        # Completely lost
        self.lost_frames += 1
        if self.lost_frames > MAX_LOST_FRAMES:
            self.last_detection = None
            self.kalman_initialized = False
        
        return {
            'detection': None,
            'method': 'none',
            'confidence': 0.0,
            'prediction': None,
            'lost': True
        }
    
    def draw_detection(self, frame, result, transformed_frame=None):
        """Draw detection result on frame"""
        # Draw Aruco markers
        corners, ids = self.detect_aruco_markers(frame)
        if ids is not None:
            try:
                detector = cv2.aruco.ArucoDetector(self.aruco_dict)
                for i, corner in enumerate(corners):
                    corner = corner.astype(int)
                    cv2.polylines(frame, [corner], True, (0, 255, 0), 2)
            except AttributeError:
                cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Draw IDs
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in ARUCO_MARKER_IDS:
                    corner_points = corners[i][0]
                    center = np.mean(corner_points, axis=0).astype(int)
                    cv2.putText(frame, f"ID:{marker_id}", tuple(center),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw detection if available
        if result['detection'] is None:
            return frame
        
        detection = result['detection']
        method = result['method']
        confidence = result['confidence']
        is_lost = result['lost']
        
        # Use camera space center for drawing on original frame
        if 'camera_center' in detection:
            center = detection['camera_center']
        else:
            center = detection['center']
        
        radius = detection['radius']
        
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
    
    logger.info("Camera initialized. Press 'q' to quit, 'r' to reset.")
    logger.info("Hybrid detector: Hough Circles (primary) + YOLO (fallback)")
    logger.info(f"Make sure 4 Aruco markers (ID: {ARUCO_MARKER_IDS}) are visible.")
    
    detector = ArucoSnowballDetector()
    frame_count = 0
    
    while True:
        ret, frame = camera.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            break
        
        # Detect snowball (only in transformed game space)
        result = detector.detect(frame, frame_count)
        
        # Get transformed frame for display
        transformed_frame = detector.transform_to_game_space(frame)
        
        # Draw detection on original frame
        frame_with_detection = detector.draw_detection(frame.copy(), result, transformed_frame)
        
        # Draw status
        status_text = f"Method: {result['method']} | "
        status_text += f"Confidence: {result['confidence']:.2f} | "
        status_text += f"Lost frames: {detector.lost_frames} | "
        status_text += f"Transform: {'OK' if detector.transform_valid else 'NO'}"
        
        cv2.putText(frame_with_detection, status_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Resize for display
        h, w = frame_with_detection.shape[:2]
        small_frame = cv2.resize(frame_with_detection, (w // DISPLAY_SCALE_FACTOR, h // DISPLAY_SCALE_FACTOR))
        cv2.imshow("Aruco + Snowball Detector", small_frame)
        
        # Show transformed view if available
        if transformed_frame is not None:
            h_t, w_t = transformed_frame.shape[:2]
            small_transformed = cv2.resize(transformed_frame, 
                                          (w_t // DISPLAY_SCALE_FACTOR, h_t // DISPLAY_SCALE_FACTOR))
            cv2.imshow("Game Space (Transformed)", small_transformed)
        
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


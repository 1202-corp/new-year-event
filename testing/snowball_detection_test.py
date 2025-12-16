"""Test script for snowball detection using YOLO neural network"""
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
    logger.error("YOLO not available. Install with: pip install ultralytics")
    sys.exit(1)

# Global YOLO model (lazy loading)
_yolo_model = None


def get_yolo_model():
    """Get or load YOLO model (lazy loading)"""
    global _yolo_model
    if _yolo_model is None:
        try:
            # Use YOLOv8 nano (lightweight model)
            # Model will be downloaded automatically on first use
            _yolo_model = YOLO('yolov8n.pt')  # nano version - smallest and fastest
            logger.info("YOLOv8 nano model loaded")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return None
    
    return _yolo_model


def detect_snowballs(frame, debug_mode=False, min_confidence=0.2):
    """
    Detect white snowballs using YOLO model.
    Returns list of detected balls with bounding boxes.
    
    Args:
        frame: Input frame
        debug_mode: If True, show all YOLO detections, not just balls
        min_confidence: Minimum confidence threshold
    """
    model = get_yolo_model()
    if model is None:
        return []
    
    try:
        # Run inference
        results = model(frame, verbose=False, conf=min_confidence)
        
        detected_balls = []
        all_detections = []  # For debug mode
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get class name
                cls = int(box.cls[0])
                class_name = model.names[cls]
                confidence = float(box.conf[0])
                
                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                # Check if region is white
                y1_int = max(0, y1)
                y2_int = min(frame.shape[0], y2)
                x1_int = max(0, x1)
                x2_int = min(frame.shape[1], x2)
                
                white_ratio = 0.0
                is_round = False
                
                if y2_int > y1_int and x2_int > x1_int:
                    ball_region = frame[y1_int:y2_int, x1_int:x2_int]
                    if ball_region.size > 0:
                        gray_region = cv2.cvtColor(ball_region, cv2.COLOR_BGR2GRAY)
                        white_pixels = np.sum(gray_region > 200)
                        total_pixels = gray_region.size
                        white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
                        
                        # Check if object is round (aspect ratio close to 1)
                        aspect_ratio = width / height if height > 0 else 0
                        is_round = 0.7 <= aspect_ratio <= 1.3  # Allow some variation
                
                # Method 1: Direct sports ball detection
                is_sports_ball = (class_name in ['sports ball', 'ball'] or 
                                 (cls == 32 and confidence > 0.2))
                
                # Method 2: Generic round white object detection
                # Accept if: round shape, high white ratio, reasonable size
                is_white_ball = (is_round and 
                                white_ratio > 0.25 and 
                                area > 100 and  # Minimum size
                                area < 50000 and  # Maximum size
                                confidence > 0.2)
                
                if is_sports_ball or is_white_ball:
                    detected_balls.append({
                        'bbox': (x1, y1, x2, y2),
                        'center': (center_x, center_y),
                        'width': width,
                        'height': height,
                        'area': area,
                        'confidence': confidence,
                        'white_ratio': white_ratio,
                        'class_name': class_name,
                        'is_sports_ball': is_sports_ball,
                        'is_white_ball': is_white_ball
                    })
                
                # Store all detections for debug mode
                if debug_mode:
                    all_detections.append({
                        'bbox': (x1, y1, x2, y2),
                        'center': (center_x, center_y),
                        'class_name': class_name,
                        'confidence': confidence,
                        'white_ratio': white_ratio,
                        'is_round': is_round
                    })
        
        # Return all detections in debug mode, otherwise just balls
        if debug_mode:
            return detected_balls, all_detections
        else:
            return detected_balls
    except Exception as e:
        logger.debug(f"YOLO detection error: {e}")
        return [] if not debug_mode else ([], [])


def draw_detections(frame, balls, all_detections=None):
    """Draw detected snowballs with bounding boxes on frame"""
    # Draw all detections in debug mode (yellow)
    if all_detections is not None:
        for det in all_detections:
            x1, y1, x2, y2 = det['bbox']
            color = (0, 255, 255)  # Yellow for all detections
            thickness = 1
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            
            # Draw class name and confidence
            text = f"{det['class_name']}: {det['confidence']:.2f}"
            cv2.putText(frame, text, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Draw detected balls (green)
    for ball in balls:
        x1, y1, x2, y2 = ball['bbox']
        center = ball['center']
        confidence = ball['confidence']
        
        # Color for bounding box (green)
        color = (0, 255, 0)
        thickness = 2
        
        # Draw bounding box rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
        # Draw center point
        cv2.circle(frame, center, 5, color, -1)
        
        # Draw info text above the bounding box
        conf_text = f"Conf: {confidence:.2f}"
        area_text = f"Area: {int(ball['area'])}"
        white_text = f"White: {ball['white_ratio']:.1%}"
        method_text = "Sports" if ball.get('is_sports_ball') else "White"
        
        y_offset = y1 - 10
        if y_offset < 20:
            y_offset = y2 + 20  # Draw below if not enough space above
        
        cv2.putText(frame, f"[{method_text}]", (x1, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, conf_text, (x1, y_offset + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, area_text, (x1, y_offset + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(frame, white_text, (x1, y_offset + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return frame


def main():
    """Main function"""
    if not YOLO_AVAILABLE:
        logger.error("YOLO is required but not available. Exiting.")
        return
    
    # Initialize camera
    camera = cv2.VideoCapture(Config.SNOWBALL_CAMERA_INDEX)
    
    if not camera.isOpened():
        logger.error(f"Could not open camera {Config.SNOWBALL_CAMERA_INDEX}")
        return
    
    # Set camera format to MJPEG (compressed) instead of RAW (uncompressed)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    camera.set(cv2.CAP_PROP_FOURCC, fourcc)
    
    # Set camera resolution
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, Config.SNOWBALL_CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.SNOWBALL_CAMERA_HEIGHT)
    
    # Set lower exposure and brightness
    camera.set(cv2.CAP_PROP_EXPOSURE, -6)
    camera.set(cv2.CAP_PROP_BRIGHTNESS, 50)
    
    try:
        camera.set(cv2.CAP_PROP_ISO_SPEED, 100)
    except:
        pass
    
    logger.info("Camera initialized. Press 'q' to quit.")
    logger.info("Press 'd' to toggle debug mode (show all YOLO detections).")
    logger.info("Throw white foam balls to test detection.")
    
    debug_mode = False
    
    while True:
        ret, frame = camera.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            break
        
        # Detect snowballs using YOLO
        if debug_mode:
            balls, all_detections = detect_snowballs(frame, debug_mode=True)
        else:
            balls = detect_snowballs(frame, debug_mode=False)
            all_detections = None
        
        # Draw detections on original frame
        frame_with_detections = frame.copy()
        frame_with_detections = draw_detections(frame_with_detections, balls, all_detections)
        
        # Draw count and mode
        count_text = f"Balls detected: {len(balls)}"
        mode_text = "[DEBUG MODE]" if debug_mode else "[NORMAL MODE]"
        cv2.putText(frame_with_detections, count_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_with_detections, mode_text, (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        if debug_mode and all_detections:
            all_text = f"All YOLO detections: {len(all_detections)}"
            cv2.putText(frame_with_detections, all_text, (10, 110),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display frame (resized to half size)
        h, w = frame_with_detections.shape[:2]
        small_frame = cv2.resize(frame_with_detections, (w // 2, h // 2))
        cv2.imshow("Snowball Detection (YOLO)", small_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('d'):
            debug_mode = not debug_mode
            logger.info(f"Debug mode: {'enabled' if debug_mode else 'disabled'}")
    
    # Cleanup
    camera.release()
    cv2.destroyAllWindows()
    logger.info("Camera released, windows closed")


if __name__ == "__main__":
    main()

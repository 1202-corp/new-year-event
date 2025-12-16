"""Test script for snowball detection using color segmentation and circle detection"""
import sys
import os

# Add parent directory to path to import game modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from game.config import Config
from game.logger import get_logger

logger = get_logger()


def detect_snowballs(frame):
    """
    Detect white snowballs in the frame using color segmentation and circle detection.
    Uses HSV color space for better white detection under different lighting.
    """
    # Convert to HSV color space (better for color detection under varying light)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for white color in HSV
    # White has low saturation and high value
    # Lower bound: low saturation, high value (bright white)
    # Upper bound: slightly higher saturation, very high value
    lower_white = np.array([0, 0, 200])  # Low hue, low saturation, high brightness
    upper_white = np.array([180, 30, 255])  # High hue, low saturation, max brightness
    
    # Create mask for white regions
    mask = cv2.inRange(hsv, lower_white, upper_white)
    
    # Apply morphological operations to remove noise
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove small noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill small holes
    
    # Apply Gaussian blur to smooth edges
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    
    # Detect circles using HoughCircles
    # Parameters:
    # - method: HOUGH_GRADIENT
    # - dp: inverse ratio of accumulator resolution
    # - minDist: minimum distance between circle centers
    # - param1: upper threshold for edge detection
    # - param2: accumulator threshold for center detection
    # - minRadius: minimum circle radius
    # - maxRadius: maximum circle radius
    circles = cv2.HoughCircles(
        mask,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=50,  # Minimum distance between circles
        param1=50,   # Upper threshold for edge detection
        param2=20,   # Accumulator threshold (lower = more false positives)
        minRadius=10,  # Minimum radius in pixels
        maxRadius=100  # Maximum radius in pixels
    )
    
    detected_balls = []
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center_x = int(i[0])
            center_y = int(i[1])
            radius = int(i[2])
            
            # Verify it's actually white by checking the area
            # Extract circle region
            y1 = max(0, center_y - radius)
            y2 = min(frame.shape[0], center_y + radius)
            x1 = max(0, center_x - radius)
            x2 = min(frame.shape[1], center_x + radius)
            
            if y2 > y1 and x2 > x1:
                circle_region = frame[y1:y2, x1:x2]
                if circle_region.size > 0:
                    # Check if region is mostly white
                    gray_region = cv2.cvtColor(circle_region, cv2.COLOR_BGR2GRAY)
                    white_pixels = np.sum(gray_region > 200)
                    total_pixels = gray_region.size
                    white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
                    
                    # Only accept if at least 40% of circle is white
                    if white_ratio > 0.4:
                        detected_balls.append({
                            'center': (center_x, center_y),
                            'radius': radius,
                            'area': np.pi * radius * radius
                        })
    
    return detected_balls, mask


def draw_detections(frame, balls):
    """Draw detected snowballs on frame"""
    for ball in balls:
        center = ball['center']
        radius = ball['radius']
        
        # Draw circle
        cv2.circle(frame, center, radius, (0, 255, 0), 2)
        cv2.circle(frame, center, 3, (0, 255, 0), -1)  # Center point
        
        # Draw area text
        area_text = f"Area: {int(ball['area'])}"
        cv2.putText(frame, area_text, (center[0] - 30, center[1] - radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return frame


def main():
    """Main function"""
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
    logger.info("Throw white foam balls to test detection.")
    
    while True:
        ret, frame = camera.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            break
        
        # Detect snowballs
        balls, mask = detect_snowballs(frame)
        
        # Draw detections on original frame
        frame_with_detections = frame.copy()
        frame_with_detections = draw_detections(frame_with_detections, balls)
        
        # Draw count
        count_text = f"Balls detected: {len(balls)}"
        cv2.putText(frame_with_detections, count_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display frames (resized to half size)
        h, w = frame_with_detections.shape[:2]
        small_frame = cv2.resize(frame_with_detections, (w // 2, h // 2))
        cv2.imshow("Snowball Detection", small_frame)
        
        # Show mask for debugging
        small_mask = cv2.resize(mask, (w // 2, h // 2))
        cv2.imshow("White Mask", small_mask)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    camera.release()
    cv2.destroyAllWindows()
    logger.info("Camera released, windows closed")


if __name__ == "__main__":
    main()


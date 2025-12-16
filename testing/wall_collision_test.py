"""Test script for wall collision detection using motion detection"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from game.config import Config
from game.logger import get_logger
from game.aruco_transform import ArucoTransform
from game.wall_collision_detector import WallCollisionDetector

logger = get_logger()


def draw_detection(frame, result):
    """Draw detection results on frame"""
    frame = frame.copy()
    
    # Draw motion mask overlay (semi-transparent)
    if 'motion_mask' in result:
        motion_colored = cv2.applyColorMap(result['motion_mask'], cv2.COLORMAP_JET)
        frame = cv2.addWeighted(frame, 0.7, motion_colored, 0.3, 0)
    
    # Draw game objects mask (green overlay)
    if 'game_objects_mask' in result:
        mask_colored = np.zeros_like(frame)
        mask_colored[result['game_objects_mask'] > 0] = [0, 255, 0]
        frame = cv2.addWeighted(frame, 0.8, mask_colored, 0.2, 0)
    
    # Draw ball position
    if result['ball_detected'] and result['ball_position']:
        x, y = result['ball_position']
        cv2.circle(frame, (x, y), 20, (0, 255, 255), 3)
        cv2.putText(frame, "BALL", (x + 25, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Draw collision
    if result['collision_detected']:
        if result['collision_position']:
            x, y = result['collision_position']
            cv2.circle(frame, (x, y), 30, (0, 0, 255), 5)
        
        wall = result.get('collision_wall', 'unknown')
        cv2.putText(frame, f"COLLISION: {wall.upper()}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    # Draw status
    status = "Ball: " + ("DETECTED" if result['ball_detected'] else "NOT DETECTED")
    cv2.putText(frame, status, (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
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
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, Config.SNOWBALL_CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.SNOWBALL_CAMERA_HEIGHT)
    
    logger.info("Camera initialized. Press 'q' to quit, 'r' to reset detector.")
    logger.info("Throw white foam balls to test collision detection.")
    
    # Initialize Aruco transform (if calibration is enabled)
    aruco_transform = None
    if Config.CALIBRATION_ENABLED:
        aruco_transform = ArucoTransform(
            camera_index=Config.SNOWBALL_CAMERA_INDEX,
            camera_width=Config.SNOWBALL_CAMERA_WIDTH,
            camera_height=Config.SNOWBALL_CAMERA_HEIGHT
        )
        # Note: Calibration should be done separately before running this test
    
    # Initialize collision detector
    detector = WallCollisionDetector(aruco_transform=aruco_transform)
    
    frame_count = 0
    
    # Example game objects (in real game, these would come from Game class)
    game_objects = {
        'enemies': [],  # Will be populated from game state
        'lines': [],    # Lane line Y positions
        'ui_panel': None  # UI panel position
    }
    
    while True:
        ret, frame = camera.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            break
        
        # Detect collision
        result = detector.detect_collision(frame, game_objects)
        
        # Draw results
        frame_with_detection = draw_detection(frame, result)
        
        # Resize for display
        h, w = frame_with_detection.shape[:2]
        small_frame = cv2.resize(frame_with_detection, (w // 2, h // 2))
        cv2.imshow("Wall Collision Detection", small_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.reset()
            logger.info("Detector reset")
        
        frame_count += 1
    
    # Cleanup
    camera.release()
    cv2.destroyAllWindows()
    logger.info("Camera released, windows closed")


if __name__ == "__main__":
    main()


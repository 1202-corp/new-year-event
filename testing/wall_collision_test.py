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


def create_combined_view(result, original_frame):
    """Create combined view with 3 images vertically: original camera, transformed camera, motion mask"""
    if 'transformed_frame' not in result or 'motion_mask' not in result:
        return None
    
    original_camera = original_frame.copy()
    transformed_frame = result['transformed_frame'].copy()
    motion_mask = result['motion_mask']
    
    # Convert motion mask to color
    motion_colored = cv2.applyColorMap(motion_mask, cv2.COLORMAP_JET)
    
    # Draw ball position on transformed frame
    if result['ball_detected'] and result['ball_position']:
        x, y = result['ball_position']
        cv2.circle(transformed_frame, (x, y), 20, (0, 255, 255), 3)
        cv2.putText(transformed_frame, "BALL", (x + 25, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # Draw game objects mask overlay on transformed frame (green)
    if 'game_objects_mask' in result:
        mask_colored_overlay = np.zeros_like(transformed_frame)
        mask_colored_overlay[result['game_objects_mask'] > 0] = [0, 255, 0]
        transformed_frame = cv2.addWeighted(transformed_frame, 0.8, mask_colored_overlay, 0.2, 0)
    
    # Resize all to same width (use transformed frame width as reference)
    target_w = transformed_frame.shape[1]
    target_h = transformed_frame.shape[0]
    
    # Resize original camera to match transformed frame size
    original_h, original_w = original_camera.shape[:2]
    original_ratio = original_h / original_w
    new_original_h = int(target_w * original_ratio)
    original_camera_resized = cv2.resize(original_camera, (target_w, new_original_h))
    
    # Resize motion mask to match transformed frame size
    motion_h, motion_w = motion_mask.shape[:2]
    motion_ratio = motion_h / motion_w
    new_motion_h = int(target_w * motion_ratio)
    motion_colored_resized = cv2.resize(motion_colored, (target_w, new_motion_h))
    
    # Add labels
    cv2.putText(original_camera_resized, "Original Camera", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(transformed_frame, "Transformed Camera (Aruco)", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    cv2.putText(motion_colored_resized, "Motion Mask", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Combine vertically: original camera, transformed camera, motion mask
    total_height = new_original_h + target_h + new_motion_h
    combined = np.zeros((total_height, target_w, 3), dtype=np.uint8)
    
    y_offset = 0
    combined[y_offset:y_offset + new_original_h, :] = original_camera_resized
    y_offset += new_original_h
    combined[y_offset:y_offset + target_h, :] = transformed_frame
    y_offset += target_h
    combined[y_offset:y_offset + new_motion_h, :] = motion_colored_resized
    
    return combined


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
        
        # Create combined view with 3 images
        combined = create_combined_view(result, frame)
        if combined is not None:
            # Resize for display
            h, w = combined.shape[:2]
            small_combined = cv2.resize(combined, (w // 2, h // 2))
            cv2.imshow("Motion Detection: Original | Transformed | Motion Mask", small_combined)
        else:
            # Fallback if no transformed frame
            cv2.imshow("Motion Detection", frame)
        
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


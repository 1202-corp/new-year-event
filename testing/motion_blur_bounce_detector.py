"""Detect ball bounce from wall using motion blur detection"""
import sys
import os

# Add parent directory to path to import game modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from game.config import Config
from game.logger import get_logger
from game.aruco_transform import ArucoTransform

logger = get_logger()

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

# Motion blur detection parameters
MIN_BLUR_LENGTH = 20      # Minimum length of motion blur (pixels)
MAX_BLUR_LENGTH = 500      # Maximum length of motion blur (pixels)
MIN_BLUR_WIDTH = 5         # Minimum width of motion blur (pixels)
MAX_BLUR_WIDTH = 100       # Maximum width of motion blur (pixels)
MIN_ASPECT_RATIO = 2.0     # Minimum aspect ratio (length/width) for motion blur
MAX_ASPECT_RATIO = 20.0    # Maximum aspect ratio

# Ball size estimation
KNOWN_BALL_RADIUS_CM = 3.0  # Real ball radius in cm (adjust based on your ball)
FOCAL_LENGTH_ESTIMATE = 1000  # Estimated focal length (will be calibrated)

# Background subtraction
BG_HISTORY = 500
BG_VAR_THRESHOLD = 50

# ============================================================================
# FUNCTIONS
# ============================================================================

def detect_motion_blur_regions(frame, background_subtractor):
    """
    Detect motion blur regions (oval shapes) in the frame.
    
    Returns:
        List of detected blur regions: [(center_x, center_y, length, width, angle), ...]
    """
    # Apply background subtraction
    fg_mask = background_subtractor.apply(frame)
    
    # Remove shadows (marked as 127 in MOG2)
    fg_mask[fg_mask == 127] = 0
    
    # Apply morphological operations to reduce noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    blur_regions = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:  # Filter small noise
            continue
        
        # Fit ellipse to contour (motion blur should be roughly elliptical)
        if len(contour) >= 5:
            try:
                ellipse = cv2.fitEllipse(contour)
                center, axes, angle = ellipse
                
                # axes is (width, height) - but we need length and width
                width, height = axes
                length = max(width, height)
                width = min(width, height)
                
                # Check if it looks like motion blur (elongated)
                aspect_ratio = length / width if width > 0 else 0
                
                if (MIN_BLUR_LENGTH <= length <= MAX_BLUR_LENGTH and
                    MIN_BLUR_WIDTH <= width <= MAX_BLUR_WIDTH and
                    MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
                    
                    blur_regions.append({
                        'center': center,
                        'length': length,
                        'width': width,
                        'angle': angle,
                        'aspect_ratio': aspect_ratio,
                        'area': area,
                        'contour': contour
                    })
            except:
                continue
    
    return blur_regions, fg_mask


def estimate_ball_distance(blur_width_pixels, known_radius_cm, focal_length):
    """
    Estimate distance to ball using its apparent size in the image.
    
    Formula: distance = (focal_length * real_size) / apparent_size
    
    Args:
        blur_width_pixels: Width of motion blur (approximate ball diameter in pixels)
        known_radius_cm: Real ball radius in cm
        focal_length: Camera focal length (pixels)
    
    Returns:
        Distance in cm
    """
    if blur_width_pixels <= 0:
        return None
    
    # blur_width is approximately the ball diameter
    ball_diameter_cm = known_radius_cm * 2
    distance_cm = (focal_length * ball_diameter_cm) / blur_width_pixels
    
    return distance_cm


def estimate_bounce_x_coordinate(blur_region, frame_width, wall_position_ratio=0.0):
    """
    Estimate X coordinate where ball hit the wall.
    
    For side-view camera:
    - Motion blur extends from wall outward
    - The end of blur closest to wall is where ball bounced
    - X coordinate is at the wall position
    
    Args:
        blur_region: Detected blur region dict
        frame_width: Width of frame
        wall_position_ratio: Ratio (0-1) where wall is in frame (0 = left, 1 = right)
    
    Returns:
        X coordinate of bounce (in game coordinates)
    """
    center_x, center_y = blur_region['center']
    length = blur_region['length']
    angle = blur_region['angle']
    
    # Calculate direction of motion blur
    # Angle is in degrees, 0 = horizontal right
    angle_rad = np.radians(angle)
    
    # Find the end of blur closest to wall
    # For side view: wall is typically on one side
    # Motion blur extends from bounce point outward
    
    # Simple approach: use center as bounce point
    # More accurate: use end of blur closest to wall
    wall_x = frame_width * wall_position_ratio
    
    # Calculate both ends of blur
    half_length = length / 2
    end1_x = center_x + half_length * np.cos(angle_rad)
    end2_x = center_x - half_length * np.cos(angle_rad)
    
    # Use end closer to wall as bounce point
    if abs(end1_x - wall_x) < abs(end2_x - wall_x):
        bounce_x = end1_x
    else:
        bounce_x = end2_x
    
    return bounce_x


def estimate_bounce_y_coordinate(blur_region, aruco_transform, game_screen_height):
    """
    Estimate Y coordinate where ball hit the wall.
    
    Uses Aruco transformation to correct for perspective distortion.
    
    Args:
        blur_region: Detected blur region dict
        aruco_transform: ArucoTransform instance
        game_screen_height: Height of game screen
    
    Returns:
        Y coordinate of bounce (in game coordinates)
    """
    if not aruco_transform or not aruco_transform.calibrated:
        # Fallback: use raw Y coordinate
        return blur_region['center'][1]
    
    # Transform blur center from camera space to game space
    camera_point = blur_region['center']
    
    # Use inverse transform to get game coordinates
    if aruco_transform.inverse_transform_matrix is not None:
        point_array = np.array([[camera_point]], dtype=np.float32)
        try:
            game_point = cv2.perspectiveTransform(
                point_array, 
                aruco_transform.inverse_transform_matrix
            )
            game_y = game_point[0][0][1]
            
            # Clamp to screen bounds
            game_y = max(0, min(game_screen_height, game_y))
            return game_y
        except:
            pass
    
    # Fallback
    return blur_region['center'][1]


def draw_detection(frame, blur_regions, bounce_points, transformed_frame=None):
    """Draw detected motion blur and bounce points"""
    result_frame = frame.copy()
    
    # Draw motion blur regions
    for blur in blur_regions:
        center = blur['center']
        length = blur['length']
        width = blur['width']
        angle = blur['angle']
        
        # Draw ellipse
        cv2.ellipse(result_frame, 
                   (int(center[0]), int(center[1])),
                   (int(length/2), int(width/2)),
                   angle,
                   0, 360,
                   (0, 255, 0), 2)
        
        # Draw center
        cv2.circle(result_frame, (int(center[0]), int(center[1])), 5, (0, 255, 0), -1)
        
        # Draw direction arrow
        angle_rad = np.radians(angle)
        end_x = int(center[0] + length/2 * np.cos(angle_rad))
        end_y = int(center[1] + length/2 * np.sin(angle_rad))
        cv2.arrowedLine(result_frame, 
                       (int(center[0]), int(center[1])),
                       (end_x, end_y),
                       (255, 0, 0), 2)
        
        # Draw info
        info = f"L:{length:.0f} W:{width:.0f} A:{blur['aspect_ratio']:.1f}"
        cv2.putText(result_frame, info,
                   (int(center[0]) + 10, int(center[1])),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Draw bounce points
    for bounce in bounce_points:
        x, y = bounce['bounce_point']
        cv2.circle(result_frame, (int(x), int(y)), 10, (0, 0, 255), -1)
        cv2.putText(result_frame, f"BOUNCE ({bounce['game_x']:.0f}, {bounce['game_y']:.0f})",
                   (int(x) + 15, int(y)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    return result_frame


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
    camera.set(cv2.CAP_PROP_BRIGHTNESS, Config.SNOWBALL_CAMERA_BRIGHTNESS)
    camera.set(cv2.CAP_PROP_CONTRAST, Config.SNOWBALL_CAMERA_CONTRAST)
    camera.set(cv2.CAP_PROP_SATURATION, Config.SNOWBALL_CAMERA_SATURATION)
    camera.set(cv2.CAP_PROP_SHARPNESS, Config.SNOWBALL_CAMERA_SHARPNESS)
    
    # Initialize Aruco transform for perspective correction
    aruco_transform = ArucoTransform(
        camera_index=Config.SNOWBALL_CAMERA_INDEX,
        camera_width=Config.SNOWBALL_CAMERA_WIDTH,
        camera_height=Config.SNOWBALL_CAMERA_HEIGHT
    )
    
    # Background subtractor
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=BG_HISTORY,
        varThreshold=BG_VAR_THRESHOLD,
        detectShadows=True
    )
    
    logger.info("Motion Blur Bounce Detector initialized")
    logger.info("Press 'q' to quit")
    logger.info("Press 'c' to calibrate Aruco markers")
    logger.info("Press 'r' to reset background")
    
    frame_count = 0
    calibrated = False
    
    while True:
        ret, frame = camera.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            break
        
        # Update Aruco calibration if needed
        if not calibrated:
            screen_width = frame.shape[1]
            screen_height = frame.shape[0]
            key = aruco_transform.update(screen_width, screen_height)
            
            if key == ord('c') or key == 32:  # 'c' or SPACE
                if aruco_transform.calibrate(screen_width, screen_height):
                    calibrated = True
                    logger.info("Aruco calibration successful!")
                else:
                    logger.warning("Calibration failed. Make sure all 4 markers are visible.")
        
        # Detect motion blur regions
        blur_regions, motion_mask = detect_motion_blur_regions(frame, bg_subtractor)
        
        # Estimate bounce points
        bounce_points = []
        for blur in blur_regions:
            # Estimate distance
            distance = estimate_ball_distance(
                blur['width'], 
                KNOWN_BALL_RADIUS_CM, 
                FOCAL_LENGTH_ESTIMATE
            )
            
            # Estimate bounce X coordinate
            bounce_x_camera = estimate_bounce_x_coordinate(blur, frame.shape[1])
            
            # Estimate bounce Y coordinate (with perspective correction)
            bounce_y_camera = blur['center'][1]
            if calibrated and aruco_transform:
                bounce_y_game = estimate_bounce_y_coordinate(
                    blur, 
                    aruco_transform, 
                    Config.SCREEN_HEIGHT
                )
            else:
                bounce_y_game = bounce_y_camera
            
            # Convert X to game coordinates (if calibrated)
            if calibrated and aruco_transform and aruco_transform.inverse_transform_matrix is not None:
                try:
                    point_array = np.array([[bounce_x_camera, bounce_y_camera]], dtype=np.float32)
                    game_point = cv2.perspectiveTransform(
                        point_array,
                        aruco_transform.inverse_transform_matrix
                    )
                    bounce_x_game = game_point[0][0][0]
                except:
                    bounce_x_game = bounce_x_camera
            else:
                bounce_x_game = bounce_x_camera
            
            bounce_points.append({
                'bounce_point': (bounce_x_camera, bounce_y_camera),
                'game_x': bounce_x_game,
                'game_y': bounce_y_game,
                'distance_cm': distance,
                'blur': blur
            })
        
        # Draw detection
        result_frame = draw_detection(frame, blur_regions, bounce_points)
        
        # Draw status
        status = f"Blur regions: {len(blur_regions)} | Bounces: {len(bounce_points)}"
        if calibrated:
            status += " | CALIBRATED"
        else:
            status += " | Press 'c' to calibrate"
        
        cv2.putText(result_frame, status, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Show bounce coordinates
        if bounce_points:
            y_offset = 60
            for i, bounce in enumerate(bounce_points[:3]):  # Show up to 3
                info = f"Bounce {i+1}: X={bounce['game_x']:.0f} Y={bounce['game_y']:.0f}"
                if bounce['distance_cm']:
                    info += f" Dist={bounce['distance_cm']:.1f}cm"
                cv2.putText(result_frame, info, (10, y_offset + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Display frames
        cv2.imshow("Motion Blur Bounce Detector", result_frame)
        cv2.imshow("Motion Mask", motion_mask)
        
        if calibrated and aruco_transform:
            transformed = aruco_transform.transform_to_game_space(frame)
            if transformed is not None:
                cv2.imshow("Game Space (Transformed)", transformed)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            if aruco_transform.calibrate(frame.shape[1], frame.shape[0]):
                calibrated = True
                logger.info("Aruco calibration successful!")
        elif key == ord('r'):
            bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=BG_HISTORY,
                varThreshold=BG_VAR_THRESHOLD,
                detectShadows=True
            )
            logger.info("Background reset")
        
        frame_count += 1
    
    # Cleanup
    camera.release()
    cv2.destroyAllWindows()
    logger.info("Camera released, windows closed")


if __name__ == "__main__":
    main()


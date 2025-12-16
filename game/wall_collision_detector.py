"""Wall collision detector using motion detection with known object masking"""
import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple
from game.config import Config
from game.logger import get_logger
from game.aruco_transform import ArucoTransform

logger = get_logger()


class WallCollisionDetector:
    """
    Detects ball collisions with walls using motion detection.
    
    Approach:
    1. Use background subtraction to detect moving objects
    2. Create mask of known game objects (enemies, lines, UI)
    3. Remove known objects from motion detection
    4. Remaining motion = potential ball
    5. Track ball movement to detect wall collisions
    """
    
    def __init__(self, aruco_transform: Optional[ArucoTransform] = None):
        """
        Initialize wall collision detector
        
        Args:
            aruco_transform: ArucoTransform instance for perspective correction
        """
        self.aruco_transform = aruco_transform
        
        # Background subtractor (MOG2 is better for varying lighting)
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,           # Number of frames for background model
            varThreshold=50,       # Threshold for variance
            detectShadows=True     # Detect and mark shadows
        )
        
        # Ball tracking state
        self.ball_position = None  # (x, y) in game coordinates
        self.ball_velocity = None  # (vx, vy) velocity vector
        self.last_ball_position = None
        
        # Motion detection parameters
        self.min_motion_area = 100  # Minimum area for motion blob
        self.max_motion_area = 10000  # Maximum area for motion blob
        
        # Collision detection parameters
        self.velocity_threshold = 5.0  # Minimum velocity change to detect collision
        self.wall_margin = 20  # Margin from edges to detect wall collision
        
    def create_game_object_mask(
        self, 
        frame_shape: Tuple[int, int],
        game_objects: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Create mask of known game objects that should be excluded from motion detection.
        
        Args:
            frame_shape: (height, width) of the frame
            game_objects: Dictionary with game object positions:
                - enemies: List of (x, y, width, height) tuples
                - lines: List of y positions for horizontal lines
                - ui_panel: (x, y, width, height) tuple for UI panel
        
        Returns:
            Binary mask where 1 = known object (should be masked), 0 = unknown (check for motion)
        """
        mask = np.zeros(frame_shape[:2], dtype=np.uint8)
        
        if game_objects is None:
            return mask
        
        h, w = frame_shape[:2]
        
        # Mask enemies
        if 'enemies' in game_objects:
            for enemy in game_objects['enemies']:
                if len(enemy) >= 4:
                    x, y, width, height = enemy[:4]
                    # Convert game coordinates to frame coordinates if needed
                    # For now, assume coordinates are already in frame space
                    x1 = max(0, int(x))
                    y1 = max(0, int(y))
                    x2 = min(w, int(x + width))
                    y2 = min(h, int(y + height))
                    if x2 > x1 and y2 > y1:
                        mask[y1:y2, x1:x2] = 255
        
        # Mask lane lines (horizontal lines)
        if 'lines' in game_objects:
            line_thickness = 8
            for line_y in game_objects['lines']:
                y = int(line_y)
                if 0 <= y < h:
                    y1 = max(0, y - line_thickness // 2)
                    y2 = min(h, y + line_thickness // 2)
                    mask[y1:y2, :] = 255
        
        # Mask UI panel
        if 'ui_panel' in game_objects and game_objects['ui_panel'] is not None:
            panel = game_objects['ui_panel']
            if len(panel) >= 4:
                x, y, width, height = panel[:4]
                x1 = max(0, int(x))
                y1 = max(0, int(y))
                x2 = min(w, int(x + width))
                y2 = min(h, int(y + height))
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = 255
        
        return mask
    
    def detect_motion(self, frame: np.ndarray, game_objects_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Detect motion in frame using background subtraction.
        
        Args:
            frame: Input frame (BGR)
            game_objects_mask: Binary mask of known objects to exclude
        
        Returns:
            Binary mask of detected motion
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows (shadows are marked as 127 in MOG2)
        fg_mask[fg_mask == 127] = 0
        
        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Remove known game objects from motion detection
        if game_objects_mask is not None:
            fg_mask[game_objects_mask > 0] = 0
        
        return fg_mask
    
    def find_ball_candidate(self, motion_mask: np.ndarray) -> Optional[Tuple[int, int, int]]:
        """
        Find ball candidate from motion mask.
        
        Args:
            motion_mask: Binary mask of motion
        
        Returns:
            (x, y, radius) of ball candidate or None
        """
        # Find contours
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Filter by area and find most likely ball
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_motion_area <= area <= self.max_motion_area:
                # Get bounding circle
                (x, y), radius = cv2.minEnclosingCircle(contour)
                x, y, radius = int(x), int(y), int(radius)
                
                # Check if roughly circular
                circularity = 4 * np.pi * area / (cv2.arcLength(contour, True) ** 2) if cv2.arcLength(contour, True) > 0 else 0
                if circularity > 0.5:  # Roughly circular
                    candidates.append((x, y, radius, area))
        
        if not candidates:
            return None
        
        # Return the largest candidate (most likely to be ball)
        best = max(candidates, key=lambda c: c[3])  # Sort by area
        return (best[0], best[1], best[2])
    
    def detect_collision(
        self, 
        frame: np.ndarray,
        game_objects: Optional[Dict] = None
    ) -> Dict:
        """
        Detect ball and check for wall collisions.
        
        Args:
            frame: Camera frame (BGR)
            game_objects: Dictionary with known game object positions
        
        Returns:
            Dictionary with detection results:
                - ball_detected: bool
                - ball_position: (x, y) or None
                - transformed_frame: Transformed frame (if Aruco available)
                - original_frame: Original camera frame
        """
        # Transform frame to game space using Aruco markers (like in aruco_detection_test.py)
        original_frame = frame.copy()
        transformed_frame = frame.copy()
        
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
        
        # Create mask of known game objects
        game_objects_mask = self.create_game_object_mask(
            transformed_frame.shape,
            game_objects
        )
        
        # Detect motion
        motion_mask = self.detect_motion(transformed_frame, game_objects_mask)
        
        # Find ball candidate
        ball_candidate = self.find_ball_candidate(motion_mask)
        
        result = {
            'ball_detected': False,
            'ball_position': None,
            'motion_mask': motion_mask,
            'game_objects_mask': game_objects_mask,
            'transformed_frame': transformed_frame,
            'original_frame': original_frame
        }
        
        if ball_candidate is None:
            # No ball detected, reset tracking
            self.last_ball_position = self.ball_position
            self.ball_position = None
            self.ball_velocity = None
            return result
        
        x, y, radius = ball_candidate
        result['ball_detected'] = True
        result['ball_position'] = (x, y)
        
        # Update ball tracking
        if self.ball_position is not None:
            # Calculate velocity
            dx = x - self.ball_position[0]
            dy = y - self.ball_position[1]
            self.ball_velocity = (dx, dy)
        else:
            # First detection, no velocity yet
            self.ball_velocity = None
        
        # Update position
        self.last_ball_position = self.ball_position
        self.ball_position = (x, y)
        
        return result
    
    def reset(self):
        """Reset detector state"""
        self.ball_position = None
        self.ball_velocity = None
        self.last_ball_position = None
        # Reset background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=True
        )


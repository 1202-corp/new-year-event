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
        self.collision_detected = False
        self.collision_position = None
        
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
        if 'ui_panel' in game_objects:
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
                - collision_detected: bool
                - collision_position: (x, y) or None
                - collision_wall: 'left', 'right', 'top', 'bottom' or None
        """
        # Transform frame to game space if Aruco transform is available
        if self.aruco_transform and self.aruco_transform.transform_valid:
            # Get game screen dimensions
            game_width = self.aruco_transform._output_width
            game_height = self.aruco_transform._output_height
            
            # Transform frame to game space
            transformed_frame = self.aruco_transform.apply_transform(frame)
            if transformed_frame is None:
                transformed_frame = frame
        else:
            transformed_frame = frame
            game_width = frame.shape[1]
            game_height = frame.shape[0]
        
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
            'collision_detected': False,
            'collision_position': None,
            'collision_wall': None,
            'motion_mask': motion_mask,
            'game_objects_mask': game_objects_mask
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
            
            # Check for wall collision
            collision = self._check_wall_collision(
                self.ball_position,
                (x, y),
                self.ball_velocity,
                game_width,
                game_height
            )
            
            if collision:
                result['collision_detected'] = True
                result['collision_position'] = (x, y)
                result['collision_wall'] = collision['wall']
                self.collision_detected = True
                self.collision_position = (x, y)
        else:
            # First detection, no velocity yet
            self.ball_velocity = None
        
        # Update position
        self.last_ball_position = self.ball_position
        self.ball_position = (x, y)
        
        return result
    
    def _check_wall_collision(
        self,
        old_pos: Tuple[int, int],
        new_pos: Tuple[int, int],
        velocity: Tuple[float, float],
        game_width: int,
        game_height: int
    ) -> Optional[Dict]:
        """
        Check if ball collided with a wall.
        
        Args:
            old_pos: Previous position (x, y)
            new_pos: Current position (x, y)
            velocity: Velocity vector (vx, vy)
            game_width: Game area width
            game_height: Game area height
        
        Returns:
            Dict with 'wall' key if collision detected, None otherwise
        """
        if velocity is None:
            return None
        
        vx, vy = velocity
        
        # Check if near wall and velocity indicates collision
        x, y = new_pos
        margin = self.wall_margin
        
        # Left wall
        if x <= margin and vx < 0:
            return {'wall': 'left'}
        
        # Right wall
        if x >= game_width - margin and vx > 0:
            return {'wall': 'right'}
        
        # Top wall
        if y <= margin and vy < 0:
            return {'wall': 'top'}
        
        # Bottom wall
        if y >= game_height - margin and vy > 0:
            return {'wall': 'bottom'}
        
        # Check for sudden velocity change (bounce)
        if self.last_ball_position is not None:
            old_vx = old_pos[0] - self.last_ball_position[0] if len(self.last_ball_position) >= 1 else 0
            old_vy = old_pos[1] - self.last_ball_position[1] if len(self.last_ball_position) >= 1 else 0
            
            # Check for significant velocity reversal
            if abs(old_vx) > self.velocity_threshold and abs(vx) > self.velocity_threshold:
                if (old_vx > 0 and vx < 0) or (old_vx < 0 and vx > 0):
                    # Horizontal bounce
                    if x <= margin:
                        return {'wall': 'left'}
                    elif x >= game_width - margin:
                        return {'wall': 'right'}
            
            if abs(old_vy) > self.velocity_threshold and abs(vy) > self.velocity_threshold:
                if (old_vy > 0 and vy < 0) or (old_vy < 0 and vy > 0):
                    # Vertical bounce
                    if y <= margin:
                        return {'wall': 'top'}
                    elif y >= game_height - margin:
                        return {'wall': 'bottom'}
        
        return None
    
    def reset(self):
        """Reset detector state"""
        self.ball_position = None
        self.ball_velocity = None
        self.last_ball_position = None
        self.collision_detected = False
        self.collision_position = None
        # Reset background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500,
            varThreshold=50,
            detectShadows=True
        )


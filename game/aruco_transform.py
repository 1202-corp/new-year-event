"""Aruco marker detection and perspective transformation for game display"""
import cv2
import numpy as np
from typing import Optional, Tuple
from game.logger import get_logger

logger = get_logger()

# Aruco marker settings
ARUCO_DICT_TYPE = cv2.aruco.DICT_4X4_50
ARUCO_MARKER_IDS = [0, 1, 2, 3]  # Expected marker IDs

# Global storage for last known marker positions
last_marker_positions = {0: None, 1: None, 2: None, 3: None}


class ArucoTransform:
    """Handles Aruco marker detection and perspective transformation"""
    
    def __init__(self, camera_index: int, camera_width: int, camera_height: int):
        self.camera_index = camera_index
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera = None
        
        # Aruco detection
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
        # Transform matrices
        self.transform_matrix = None
        self.inverse_transform_matrix = None
        self.transform_valid = False
        
        # Initialize camera
        self._init_camera()
    
    def _init_camera(self):
        """Initialize camera for Aruco detection"""
        try:
            self.camera = cv2.VideoCapture(self.camera_index)
            if self.camera.isOpened():
                # Set camera format to MJPEG for better performance
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                self.camera.set(cv2.CAP_PROP_FOURCC, fourcc)
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
                
                # Set lower exposure and brightness for better marker detection
                self.camera.set(cv2.CAP_PROP_EXPOSURE, -6)
                self.camera.set(cv2.CAP_PROP_BRIGHTNESS, 50)
                
                try:
                    self.camera.set(cv2.CAP_PROP_ISO_SPEED, 100)
                except:
                    pass
                
                logger.info(f"Aruco camera {self.camera_index} initialized")
            else:
                logger.warning(f"Could not open Aruco camera {self.camera_index}")
        except Exception as e:
            logger.warning(f"Failed to initialize Aruco camera: {e}")
    
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
        global last_marker_positions
        
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
        """Determine which marker belongs to which corner (using inner corners)"""
        global last_marker_positions
        
        # Get corner points (closest to screen center = inner corner) of each marker
        marker_corners = {}
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in ARUCO_MARKER_IDS:
                    # Get corner closest to screen center (inner corner)
                    corner_points = corners[i][0]
                    closest_corner = self.get_closest_corner_to_center(corner_points, screen_center)
                    marker_corners[marker_id] = closest_corner
                    # Update last known position
                    last_marker_positions[marker_id] = closest_corner
        
        # Use last known positions for missing markers
        for marker_id in ARUCO_MARKER_IDS:
            if marker_id not in marker_corners and last_marker_positions[marker_id] is not None:
                marker_corners[marker_id] = last_marker_positions[marker_id]
        
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
    
    def update_transform(self, frame):
        """Update perspective transform matrix based on Aruco markers"""
        if frame is None:
            self.transform_valid = False
            return False
        
        h, w = frame.shape[:2]
        screen_center = (w // 2, h // 2)
        
        # Detect markers
        corners, ids = self.detect_aruco_markers(frame)
        
        # Determine corners
        top_left, top_right, bottom_right, bottom_left = self.determine_corners(corners, ids, screen_center)
        
        if any(p is None for p in [top_left, top_right, bottom_right, bottom_left]):
            self.transform_valid = False
            return False
        
        # Destination points (output rectangle - game screen dimensions)
        # These will be set based on actual game screen size
        # For now, use frame dimensions
        dst_points = np.array([
            [0, 0],           # Top-left
            [w, 0],           # Top-right
            [w, h],           # Bottom-right
            [0, h]            # Bottom-left
        ], dtype=np.float32)
        
        # Source points (Aruco marker inner corners)
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
    
    def read_camera_frame(self):
        """Read frame from camera for Aruco detection"""
        if self.camera is None or not self.camera.isOpened():
            return None
        
        ret, frame = self.camera.read()
        if not ret:
            return None
        
        return frame
    
    def update(self, game_screen_width: int, game_screen_height: int):
        """
        Update transform based on current camera frame
        
        Args:
            game_screen_width: Width of the game screen
            game_screen_height: Height of the game screen
        """
        frame = self.read_camera_frame()
        if frame is None:
            self.transform_valid = False
            return False
        
        h, w = frame.shape[:2]
        screen_center = (w // 2, h // 2)
        
        # Detect markers
        corners, ids = self.detect_aruco_markers(frame)
        
        # Determine corners
        top_left, top_right, bottom_right, bottom_left = self.determine_corners(corners, ids, screen_center)
        
        if any(p is None for p in [top_left, top_right, bottom_right, bottom_left]):
            self.transform_valid = False
            return False
        
        # Destination points (game screen rectangle)
        dst_points = np.array([
            [0, 0],                           # Top-left
            [game_screen_width, 0],           # Top-right
            [game_screen_width, game_screen_height],  # Bottom-right
            [0, game_screen_height]           # Bottom-left
        ], dtype=np.float32)
        
        # Source points (Aruco marker inner corners from camera)
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
    
    def apply_transform(self, game_surface: np.ndarray) -> Optional[np.ndarray]:
        """
        Apply perspective transformation to game surface
        
        Args:
            game_surface: Game screen as numpy array (BGR format)
        
        Returns:
            Transformed surface or None if transform is invalid
        """
        if not self.transform_valid or self.transform_matrix is None:
            return game_surface
        
        h, w = game_surface.shape[:2]
        transformed = cv2.warpPerspective(game_surface, self.transform_matrix, (w, h))
        return transformed
    
    def is_valid(self) -> bool:
        """Check if transform is valid"""
        return self.transform_valid
    
    def release(self):
        """Release camera resources"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None


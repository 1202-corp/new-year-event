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
        
        # Calibration state
        self.calibrated = False
        self.calibration_marker_positions = None  # Store marker positions after calibration
        
        # Debug display - enable by default, will be disabled if windows can't be shown
        self.debug_enabled = True  # Enable by default, disable if cv2.imshow fails
        self._debug_initialized = False  # Track if debug windows were successfully initialized
        
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
        """Determine which marker belongs to which corner (using corners closest to screen center)"""
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
    
    def calibrate(self, game_screen_width: int, game_screen_height: int) -> bool:
        """
        Perform calibration: detect and store marker positions
        
        Args:
            game_screen_width: Width of the game screen
            game_screen_height: Height of the game screen
        
        Returns:
            True if calibration successful, False otherwise
        """
        frame = self.read_camera_frame()
        if frame is None:
            if self.debug_enabled:
                self._draw_debug_no_frame()
            return False
        
        h, w = frame.shape[:2]
        screen_center = (w // 2, h // 2)
        
        # Detect markers
        corners, ids = self.detect_aruco_markers(frame)
        
        # Determine corners
        top_left, top_right, bottom_right, bottom_left = self.determine_corners(corners, ids, screen_center)
        
        # Always show debug during calibration
        if self.debug_enabled:
            self._draw_debug(frame, corners, ids, top_left, top_right, bottom_right, bottom_left)
        
        if any(p is None for p in [top_left, top_right, bottom_right, bottom_left]):
            return False
        
        # Store calibration positions
        self.calibration_marker_positions = {
            'top_left': top_left.copy(),
            'top_right': top_right.copy(),
            'bottom_right': bottom_right.copy(),
            'bottom_left': bottom_left.copy(),
            'game_screen_width': game_screen_width,
            'game_screen_height': game_screen_height
        }
        
        # Calculate and store transform
        self._calculate_transform_from_positions()
        self.calibrated = True
        
        return True
    
    def _calculate_transform_from_positions(self):
        """Calculate transform matrix from stored calibration positions"""
        if self.calibration_marker_positions is None:
            return
        
        game_screen_width = self.calibration_marker_positions['game_screen_width']
        game_screen_height = self.calibration_marker_positions['game_screen_height']
        
        top_left = self.calibration_marker_positions['top_left']
        top_right = self.calibration_marker_positions['top_right']
        bottom_right = self.calibration_marker_positions['bottom_right']
        bottom_left = self.calibration_marker_positions['bottom_left']
        
        # Source points (game screen rectangle - what we have)
        src_points = np.array([
            [0, 0],                           # Top-left
            [game_screen_width, 0],           # Top-right
            [game_screen_width, game_screen_height],  # Bottom-right
            [0, game_screen_height]           # Bottom-left
        ], dtype=np.float32)
        
        # Destination points (Aruco marker corners from camera)
        dst_points_camera = np.array([
            top_left,         # Top-left marker corner
            top_right,        # Top-right marker corner
            bottom_right,     # Bottom-right marker corner
            bottom_left       # Bottom-left marker corner
        ], dtype=np.float32)
        
        # Scale marker positions from camera coordinates to game screen size
        x_coords = dst_points_camera[:, 0]
        y_coords = dst_points_camera[:, 1]
        min_x = float(np.min(x_coords))
        max_x = float(np.max(x_coords))
        min_y = float(np.min(y_coords))
        max_y = float(np.max(y_coords))
        
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y
        
        # Normalize marker positions to [0, 1] range based on bounding box
        if bbox_width > 0 and bbox_height > 0:
            dst_points_normalized = dst_points_camera.copy()
            dst_points_normalized[:, 0] = (dst_points_normalized[:, 0] - min_x) / bbox_width
            dst_points_normalized[:, 1] = (dst_points_normalized[:, 1] - min_y) / bbox_height
        else:
            dst_points_normalized = dst_points_camera.copy()
        
        # Scale normalized positions to game screen dimensions
        dst_points = dst_points_normalized.copy()
        dst_points[:, 0] *= game_screen_width
        dst_points[:, 1] *= game_screen_height
        
        # Store output dimensions
        self._output_width = game_screen_width
        self._output_height = game_screen_height
        
        # Calculate perspective transform matrix
        self.transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        self.inverse_transform_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
        self.transform_valid = True
    
    def update(self, game_screen_width: int, game_screen_height: int):
        """
        Update transform based on current camera frame (only if not calibrated)
        
        Args:
            game_screen_width: Width of the game screen
            game_screen_height: Height of the game screen
        """
        # If already calibrated, use stored positions
        if self.calibrated and self.calibration_marker_positions is not None:
            # Recalculate transform if screen size changed
            if (self.calibration_marker_positions['game_screen_width'] != game_screen_width or
                self.calibration_marker_positions['game_screen_height'] != game_screen_height):
                self.calibration_marker_positions['game_screen_width'] = game_screen_width
                self.calibration_marker_positions['game_screen_height'] = game_screen_height
                self._calculate_transform_from_positions()
            return self.transform_valid
        
        # Not calibrated yet - detect markers in real-time
        frame = self.read_camera_frame()
        if frame is None:
            self.transform_valid = False
            if self.debug_enabled:
                self._draw_debug_no_frame()
            return False
        
        h, w = frame.shape[:2]
        screen_center = (w // 2, h // 2)
        
        # Detect markers
        corners, ids = self.detect_aruco_markers(frame)
        
        # Determine corners
        top_left, top_right, bottom_right, bottom_left = self.determine_corners(corners, ids, screen_center)
        
        if any(p is None for p in [top_left, top_right, bottom_right, bottom_left]):
            self.transform_valid = False
            if self.debug_enabled:
                self._draw_debug(frame, corners, ids, None, None, None, None)
            return False
        
        # Source points (game screen rectangle - what we have)
        src_points = np.array([
            [0, 0],                           # Top-left
            [game_screen_width, 0],           # Top-right
            [game_screen_width, game_screen_height],  # Bottom-right
            [0, game_screen_height]           # Bottom-left
        ], dtype=np.float32)
        
        # Destination points (Aruco marker centers from camera - where we want to map to)
        # These define where the corners of the game window should appear for the viewer
        # Marker centers are at the corners of the projected image from viewer's perspective
        dst_points_camera = np.array([
            top_left,         # Top-left marker center
            top_right,        # Top-right marker center
            bottom_right,     # Bottom-right marker center
            bottom_left       # Bottom-left marker center
        ], dtype=np.float32)
        
        # Scale marker positions from camera coordinates to game screen size
        # Calculate bounding box of markers in camera coordinates
        x_coords = dst_points_camera[:, 0]
        y_coords = dst_points_camera[:, 1]
        min_x = float(np.min(x_coords))
        max_x = float(np.max(x_coords))
        min_y = float(np.min(y_coords))
        max_y = float(np.max(y_coords))
        
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y
        
        # Normalize marker positions to [0, 1] range based on bounding box
        if bbox_width > 0 and bbox_height > 0:
            dst_points_normalized = dst_points_camera.copy()
            dst_points_normalized[:, 0] = (dst_points_normalized[:, 0] - min_x) / bbox_width
            dst_points_normalized[:, 1] = (dst_points_normalized[:, 1] - min_y) / bbox_height
        else:
            dst_points_normalized = dst_points_camera.copy()
        
        # Scale normalized positions to game screen dimensions
        dst_points = dst_points_normalized.copy()
        dst_points[:, 0] *= game_screen_width
        dst_points[:, 1] *= game_screen_height
        
        # Store output dimensions (same as game screen)
        self._output_width = game_screen_width
        self._output_height = game_screen_height
        
        # Calculate perspective transform matrix
        # This transforms the rectangular game window to match the scaled Aruco marker corners
        self.transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        self.inverse_transform_matrix = cv2.getPerspectiveTransform(dst_points, src_points)
        self.transform_valid = True
        
        # Draw debug windows
        if self.debug_enabled:
            self._draw_debug(frame, corners, ids, top_left, top_right, bottom_right, bottom_left, 
                           src_points, dst_points, game_screen_width, game_screen_height)
        
        return True
    
    def _draw_debug_no_frame(self):
        """Draw debug window when no frame is available"""
        if not self.debug_enabled:
            return
        
        # Create empty frame with message
        debug_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(debug_frame, "No camera frame", (50, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        try:
            cv2.imshow("Aruco Debug: Camera View", debug_frame)
            # Process OpenCV window events
            cv2.waitKey(1)
        except Exception as e:
            logger.debug(f"Could not display debug window: {e}")
            # Disable debug if windows can't be shown
            self.debug_enabled = False
    
    def _draw_debug(self, frame, corners, ids, top_left, top_right, bottom_right, bottom_left,
                   src_points=None, dst_points=None, game_width=None, game_height=None):
        """Draw debug visualization of Aruco detection and transform"""
        if not self.debug_enabled:
            return
        
        # Create debug frame (copy of original)
        debug_frame = frame.copy()
        h, w = debug_frame.shape[:2]
        
        # Draw detected markers
        if ids is not None:
            try:
                detector = cv2.aruco.ArucoDetector(self.aruco_dict)
                for i, corner in enumerate(corners):
                    corner = corner.astype(int)
                    cv2.polylines(debug_frame, [corner], True, (0, 255, 0), 2)
            except AttributeError:
                cv2.aruco.drawDetectedMarkers(debug_frame, corners, ids)
            
            # Draw IDs and centers
            for i, marker_id in enumerate(ids.flatten()):
                if marker_id in ARUCO_MARKER_IDS:
                    corner_points = corners[i][0]
                    center = np.mean(corner_points, axis=0).astype(int)
                    # Draw center point
                    cv2.circle(debug_frame, tuple(center), 8, (255, 255, 255), -1)
                    cv2.putText(debug_frame, f"ID:{marker_id}", tuple(center + [15, -10]),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw corner points and labels
        if top_left is not None:
            cv2.circle(debug_frame, tuple(top_left.astype(int)), 10, (0, 255, 0), -1)
            cv2.putText(debug_frame, "TL", tuple(top_left.astype(int) + [10, -10]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if top_right is not None:
            cv2.circle(debug_frame, tuple(top_right.astype(int)), 10, (255, 0, 0), -1)
            cv2.putText(debug_frame, "TR", tuple(top_right.astype(int) + [10, -10]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        if bottom_right is not None:
            cv2.circle(debug_frame, tuple(bottom_right.astype(int)), 10, (0, 0, 255), -1)
            cv2.putText(debug_frame, "BR", tuple(bottom_right.astype(int) + [10, 10]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if bottom_left is not None:
            cv2.circle(debug_frame, tuple(bottom_left.astype(int)), 10, (255, 255, 0), -1)
            cv2.putText(debug_frame, "BL", tuple(bottom_left.astype(int) + [-30, 10]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Draw transform quadrilateral
        if all(p is not None for p in [top_left, top_right, bottom_right, bottom_left]):
            pts = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
            cv2.polylines(debug_frame, [pts], True, (255, 255, 255), 2)
            
            # Draw status
            status_text = "Transform: VALID" if self.transform_valid else "Transform: INVALID"
            status_color = (0, 255, 0) if self.transform_valid else (0, 0, 255)
            cv2.putText(debug_frame, status_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
            
            if game_width and game_height:
                info_text = f"Game: {game_width}x{game_height}"
                cv2.putText(debug_frame, info_text, (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        else:
            cv2.putText(debug_frame, "Waiting for 4 markers...", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Draw center point
        screen_center = (w // 2, h // 2)
        cv2.circle(debug_frame, screen_center, 5, (255, 255, 255), -1)
        cv2.putText(debug_frame, "CENTER", (screen_center[0] + 10, screen_center[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Resize for display (half size)
        small_frame = cv2.resize(debug_frame, (w // 2, h // 2))
        try:
            cv2.imshow("Aruco Debug: Camera View", small_frame)
            # Process OpenCV window events (required for window updates)
            cv2.waitKey(1)
            # Mark debug as successfully initialized on first successful call
            if not self._debug_initialized:
                self._debug_initialized = True
                logger.info("Debug windows enabled successfully")
        except Exception as e:
            # Only log and disable on first failure
            if not self._debug_initialized:
                logger.warning(f"Could not display debug window: {e}. Debug windows disabled.")
                self.debug_enabled = False
            return
        
        # Show transformed preview if transform is valid
        if self.transform_valid:
            # Create a test rectangle to show transformation
            test_rect = np.zeros((game_height or h, game_width or w, 3), dtype=np.uint8)
            # Draw grid on test rectangle
            grid_size = 50
            for x in range(0, test_rect.shape[1], grid_size):
                cv2.line(test_rect, (x, 0), (x, test_rect.shape[0]), (100, 100, 100), 1)
            for y in range(0, test_rect.shape[0], grid_size):
                cv2.line(test_rect, (0, y), (test_rect.shape[1], y), (100, 100, 100), 1)
            
            # Draw corners on test rectangle
            corner_size = 20
            cv2.circle(test_rect, (0, 0), corner_size, (0, 255, 0), -1)  # TL
            cv2.circle(test_rect, (test_rect.shape[1], 0), corner_size, (255, 0, 0), -1)  # TR
            cv2.circle(test_rect, (test_rect.shape[1], test_rect.shape[0]), corner_size, (0, 0, 255), -1)  # BR
            cv2.circle(test_rect, (0, test_rect.shape[0]), corner_size, (255, 255, 0), -1)  # BL
            
            # Apply inverse transform to show how it would look
            if self.inverse_transform_matrix is not None:
                transformed_preview = cv2.warpPerspective(test_rect, self.inverse_transform_matrix, (w, h))
                small_preview = cv2.resize(transformed_preview, (w // 2, h // 2))
                try:
                    cv2.imshow("Aruco Debug: Transform Preview", small_preview)
                    # Process OpenCV window events
                    cv2.waitKey(1)
                except Exception as e:
                    # Only log on failure, don't disable if already initialized
                    if not self._debug_initialized:
                        logger.warning(f"Could not display transform preview: {e}")
                        self.debug_enabled = False
    
    def apply_transform(self, game_surface: np.ndarray) -> Optional[np.ndarray]:
        """
        Apply perspective transformation to game surface
        
        Args:
            game_surface: Game screen as numpy array (BGR format)
        
        Returns:
            Transformed surface with black areas where there's no game content,
            or None if transform is invalid
        """
        if not self.transform_valid or self.transform_matrix is None:
            return game_surface
        
        # Get output dimensions from stored values
        if not hasattr(self, '_output_width') or not hasattr(self, '_output_height'):
            # Fallback: use input size
            h, w = game_surface.shape[:2]
            return cv2.warpPerspective(game_surface, self.transform_matrix, (w, h))
        
        output_width = self._output_width
        output_height = self._output_height
        
        # Create output image with black background (for empty areas)
        output_image = np.zeros((output_height, output_width, 3), dtype=np.uint8)
        
        # Warp the game surface to match the Aruco marker corners
        # This will create a transformed image that fits within the bounding box
        transformed = cv2.warpPerspective(
            game_surface, 
            self.transform_matrix, 
            (output_width, output_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)  # Black border for empty areas
        )
        
        return transformed
    
    def is_valid(self) -> bool:
        """Check if transform is valid"""
        return self.transform_valid
    
    def release(self):
        """Release camera resources"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None


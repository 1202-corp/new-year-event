"""Test script for Aruco marker detection and perspective transformation"""
import sys
import os

# Add parent directory to path to import game modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from game.config import Config
from game.logger import get_logger

logger = get_logger()

# Global storage for last known marker positions
last_marker_positions = {0: None, 1: None, 2: None, 3: None}


def detect_aruco_markers(frame):
    """Detect Aruco markers in the frame"""
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Define Aruco dictionary (4x4, 50 markers)
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_params = cv2.aruco.DetectorParameters()
    
    # Detect markers (new API for OpenCV 4.7+)
    try:
        # Try new API first (OpenCV 4.7+)
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        corners, ids, rejected = detector.detectMarkers(gray)
    except AttributeError:
        # Fallback to old API (OpenCV < 4.7)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
    
    return corners, ids


def get_closest_corner_to_center(corner_points, screen_center):
    """
    Get the corner of a marker that is closest to the screen center.
    corner_points: array of 4 corners of the marker
    screen_center: (x, y) center of the screen
    Returns: corner point closest to screen center
    """
    distances = []
    for corner in corner_points:
        dist = np.sqrt((corner[0] - screen_center[0])**2 + (corner[1] - screen_center[1])**2)
        distances.append(dist)
    
    closest_idx = np.argmin(distances)
    return corner_points[closest_idx]


def determine_corners(corners, ids, screen_center):
    """
    Determine which marker belongs to which corner.
    Expected IDs: 0, 1, 2, 3
    Returns: top_left, top_right, bottom_right, bottom_left
    Uses corner closest to screen center for each marker.
    Uses last known positions if markers are temporarily lost.
    """
    global last_marker_positions
    
    # Get corner points (closest to screen center) of each marker from current frame
    marker_corners = {}
    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in [0, 1, 2, 3]:
                # Get corner closest to screen center
                corner_points = corners[i][0]
                closest_corner = get_closest_corner_to_center(corner_points, screen_center)
                marker_corners[marker_id] = closest_corner
                # Update last known position
                last_marker_positions[marker_id] = closest_corner
    
    # Use last known positions for missing markers
    for marker_id in [0, 1, 2, 3]:
        if marker_id not in marker_corners and last_marker_positions[marker_id] is not None:
            marker_corners[marker_id] = last_marker_positions[marker_id]
    
    if len(marker_corners) != 4:
        return None, None, None, None
    
    # Determine corners based on position
    # Top-left: smallest x + y
    # Top-right: largest x, smallest y
    # Bottom-right: largest x + y
    # Bottom-left: smallest x, largest y
    
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


def apply_perspective_transform(frame, src_points):
    """
    Apply perspective transformation to the frame
    src_points: (top_left, top_right, bottom_right, bottom_left)
    """
    # Check if any point is None (need to check each element separately)
    if any(p is None for p in src_points):
        return None
    
    # Get frame dimensions
    h, w = frame.shape[:2]
    
    # Destination points (output rectangle)
    dst_points = np.array([
        [0, 0],           # Top-left
        [w, 0],           # Top-right
        [w, h],           # Bottom-right
        [0, h]            # Bottom-left
    ], dtype=np.float32)
    
    # Source points
    src = np.array([
        src_points[0],    # Top-left
        src_points[1],    # Top-right
        src_points[2],    # Bottom-right
        src_points[3]     # Bottom-left
    ], dtype=np.float32)
    
    # Calculate perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src, dst_points)
    
    # Apply transformation
    transformed = cv2.warpPerspective(frame, matrix, (w, h))
    
    return transformed


def draw_markers(frame, corners, ids):
    """Draw detected markers on frame"""
    if ids is not None:
        # Draw markers (new API for OpenCV 4.7+)
        try:
            # Try new API first (OpenCV 4.7+)
            detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
            # New API doesn't have drawDetectedMarkers, draw manually
            for i, corner in enumerate(corners):
                corner = corner.astype(int)
                cv2.polylines(frame, [corner], True, (0, 255, 0), 2)
        except AttributeError:
            # Fallback to old API (OpenCV < 4.7)
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        # Draw IDs
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id in [0, 1, 2, 3]:
                # Get center of marker
                corner_points = corners[i][0]
                center = np.mean(corner_points, axis=0).astype(int)
                # Draw ID text
                cv2.putText(frame, f"ID:{marker_id}", tuple(center),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return frame


def main():
    """Main function"""
    # Initialize camera
    camera = cv2.VideoCapture(Config.SNOWBALL_CAMERA_INDEX)
    
    if not camera.isOpened():
        logger.error(f"Could not open camera {Config.SNOWBALL_CAMERA_INDEX}")
        return
    
    # Set camera format to MJPEG (compressed) instead of RAW (uncompressed)
    # MJPEG is much faster and lighter than RAW format
    # FOURCC code for MJPEG: 'MJPG'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    camera.set(cv2.CAP_PROP_FOURCC, fourcc)
    
    # Set camera resolution
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, Config.SNOWBALL_CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.SNOWBALL_CAMERA_HEIGHT)
    
    # Verify format
    current_fourcc = int(camera.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((current_fourcc >> 8 * i) & 0xFF) for i in range(4)])
    
    logger.info("Camera initialized. Press 'q' to quit.")
    logger.info(f"Camera format: {fourcc_str} (should be MJPG for MJPEG)")
    logger.info("Make sure 4 Aruco markers (ID: 0, 1, 2, 3) are visible in the frame.")
    
    while True:
        ret, frame = camera.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            break
        
        # Detect Aruco markers
        corners, ids = detect_aruco_markers(frame)
        
        # Draw markers on original frame
        frame_with_markers = frame.copy()
        if ids is not None:
            frame_with_markers = draw_markers(frame_with_markers, corners, ids)
        
        # Get screen center for corner selection
        h, w = frame.shape[:2]
        screen_center = (w // 2, h // 2)
        
        # Determine corners (uses last known positions if markers are lost)
        top_left, top_right, bottom_right, bottom_left = determine_corners(corners, ids, screen_center)
        
        # Apply perspective transform if we have positions (from current or last known)
        transformed_frame = None
        if all(p is not None for p in [top_left, top_right, bottom_right, bottom_left]):
            transformed_frame = apply_perspective_transform(frame, 
                                                           (top_left, top_right, bottom_right, bottom_left))
            
            # Draw corner indicators on original frame
            cv2.circle(frame_with_markers, tuple(top_left.astype(int)), 10, (0, 255, 0), -1)
            cv2.circle(frame_with_markers, tuple(top_right.astype(int)), 10, (255, 0, 0), -1)
            cv2.circle(frame_with_markers, tuple(bottom_right.astype(int)), 10, (0, 0, 255), -1)
            cv2.circle(frame_with_markers, tuple(bottom_left.astype(int)), 10, (255, 255, 0), -1)
            
            # Draw labels
            cv2.putText(frame_with_markers, "TL", tuple(top_left.astype(int) + [10, -10]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_with_markers, "TR", tuple(top_right.astype(int) + [10, -10]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame_with_markers, "BR", tuple(bottom_right.astype(int) + [10, 10]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame_with_markers, "BL", tuple(bottom_left.astype(int) + [-30, 10]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            # Show message only if we don't have any stored positions
            if all(p is None for p in last_marker_positions.values()):
                cv2.putText(frame_with_markers, "Need 4 markers (ID: 0,1,2,3)", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                transformed_frame = np.zeros_like(frame)
                cv2.putText(transformed_frame, "Waiting for 4 markers...", (10, frame.shape[0] // 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display frames (resized to half size)
        h, w = frame_with_markers.shape[:2]
        small_frame = cv2.resize(frame_with_markers, (w // 2, h // 2))
        cv2.imshow("Camera View (with Aruco markers)", small_frame)
        
        if transformed_frame is not None:
            h_t, w_t = transformed_frame.shape[:2]
            small_transformed = cv2.resize(transformed_frame, (w_t // 2, h_t // 2))
            cv2.imshow("Transformed View", small_transformed)
        
        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    camera.release()
    cv2.destroyAllWindows()
    logger.info("Camera released, windows closed")


if __name__ == "__main__":
    main()


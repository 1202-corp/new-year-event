"""Test script comparing multiple snowball detection methods"""
import sys
import os

# Add parent directory to path to import game modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from game.config import Config
from game.logger import get_logger

logger = get_logger()


class MultiMethodDetector:
    """Detector that uses multiple methods to detect snowballs"""
    
    def __init__(self):
        # Background subtractors
        self.bg_subtractor_mog2 = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=True
        )
        self.bg_subtractor_gmg = cv2.createBackgroundSubtractorGMG(
            initializationFrames=120, decisionThreshold=0.8
        )
        
        # For optical flow
        self.prev_gray = None
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.tracks = []
        self.track_len = 10
        self.detect_interval = 5
        self.frame_idx = 0
        
        # For frame difference
        self.prev_frame = None
        
        # For Kalman filter (simple smoothing)
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                   [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)
        self.kalman.measurementNoiseCov = 0.1 * np.eye(2, dtype=np.float32)
        self.kalman.statePre = np.array([0, 0, 0, 0], dtype=np.float32)
        self.kalman.statePost = np.array([0, 0, 0, 0], dtype=np.float32)
        self.last_detection = None
    
    def detect_hough_circles(self, frame):
        """Method 1: Hough Circles detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.medianBlur(gray, 5)
        
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=100
        )
        
        result_frame = frame.copy()
        detections = []
        
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for (x, y, r) in circles[0, :]:
                cv2.circle(result_frame, (x, y), r, (0, 255, 0), 2)
                cv2.circle(result_frame, (x, y), 3, (0, 255, 0), -1)
                detections.append({'center': (x, y), 'radius': r, 'method': 'Hough'})
        
        return result_frame, detections
    
    def detect_contours(self, frame):
        """Method 2: Contour detection with Canny"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Morphological operations to close gaps
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result_frame = frame.copy()
        detections = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 300:  # Minimum area filter
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                perimeter = cv2.arcLength(cnt, True)
                
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                    
                    if radius > 10 and circularity > 0.7:
                        x, y, radius = int(x), int(y), int(radius)
                        cv2.circle(result_frame, (x, y), radius, (255, 0, 0), 2)
                        cv2.circle(result_frame, (x, y), 3, (255, 0, 0), -1)
                        detections.append({
                            'center': (x, y),
                            'radius': radius,
                            'method': 'Contours',
                            'circularity': circularity
                        })
        
        return result_frame, detections, edges
    
    def detect_background_subtraction_mog2(self, frame):
        """Method 3: Background subtraction using MOG2"""
        fg_mask = self.bg_subtractor_mog2.apply(frame)
        
        # Morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in the foreground mask
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result_frame = frame.copy()
        detections = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                if radius > 10:
                    x, y, radius = int(x), int(y), int(radius)
                    cv2.circle(result_frame, (x, y), radius, (0, 0, 255), 2)
                    cv2.circle(result_frame, (x, y), 3, (0, 0, 255), -1)
                    detections.append({
                        'center': (x, y),
                        'radius': radius,
                        'method': 'MOG2'
                    })
        
        # Convert mask to BGR for display
        fg_mask_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        
        return result_frame, detections, fg_mask_bgr
    
    def detect_background_subtraction_gmg(self, frame):
        """Method 4: Background subtraction using GMG"""
        fg_mask = self.bg_subtractor_gmg.apply(frame)
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result_frame = frame.copy()
        detections = []
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                if radius > 10:
                    x, y, radius = int(x), int(y), int(radius)
                    cv2.circle(result_frame, (x, y), radius, (255, 0, 255), 2)
                    cv2.circle(result_frame, (x, y), 3, (255, 0, 255), -1)
                    detections.append({
                        'center': (x, y),
                        'radius': radius,
                        'method': 'GMG'
                    })
        
        # Convert mask to BGR for display
        fg_mask_bgr = cv2.cvtColor(fg_mask, cv2.COLOR_GRAY2BGR)
        
        return result_frame, detections, fg_mask_bgr
    
    def detect_frame_difference(self, frame):
        """Method 5: Frame difference detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        result_frame = frame.copy()
        detections = []
        diff_frame = None
        
        if self.prev_frame is not None:
            # Calculate absolute difference
            diff = cv2.absdiff(self.prev_frame, gray)
            _, diff_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            
            # Morphological operations
            kernel = np.ones((5, 5), np.uint8)
            diff_thresh = cv2.morphologyEx(diff_thresh, cv2.MORPH_OPEN, kernel)
            diff_thresh = cv2.morphologyEx(diff_thresh, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(diff_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 300:
                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    if radius > 10:
                        x, y, radius = int(x), int(y), int(radius)
                        cv2.circle(result_frame, (x, y), radius, (0, 255, 255), 2)
                        cv2.circle(result_frame, (x, y), 3, (0, 255, 255), -1)
                        detections.append({
                            'center': (x, y),
                            'radius': radius,
                            'method': 'FrameDiff'
                        })
            
            diff_frame = cv2.cvtColor(diff_thresh, cv2.COLOR_GRAY2BGR)
        
        self.prev_frame = gray.copy()
        
        return result_frame, detections, diff_frame
    
    def detect_optical_flow_lk(self, frame):
        """Method 6: Lucas-Kanade optical flow"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        result_frame = frame.copy()
        detections = []
        
        if self.prev_gray is not None:
            # Detect new features or use existing tracks
            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, gray
                p0 = np.float32([tr[-1] for tr in self.tracks]).reshape(-1, 1, 2)
                p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **self.lk_params)
                p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **self.lk_params)
                d = abs(p0 - p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        tr.pop(0)
                    new_tracks.append(tr)
                    
                    # Draw track
                    cv2.circle(result_frame, (int(x), int(y)), 5, (255, 255, 0), -1)
                    
                self.tracks = new_tracks
            
            # Detect new features periodically
            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(gray, mask=mask, **self.feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])
            
            # Draw tracks
            for tr in self.tracks:
                if len(tr) > 1:
                    pts = np.int32(tr)
                    cv2.polylines(result_frame, [pts], False, (255, 255, 0), 2)
                    
                    # Use last point as detection
                    if len(tr) >= 3:
                        x, y = tr[-1]
                        detections.append({
                            'center': (int(x), int(y)),
                            'radius': 15,
                            'method': 'LK Flow'
                        })
        
        self.prev_gray = gray.copy()
        self.frame_idx += 1
        
        return result_frame, detections
    
    def detect_optical_flow_farneback(self, frame):
        """Method 7: Farneback dense optical flow"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        result_frame = frame.copy()
        detections = []
        flow_vis = None
        
        if self.prev_gray is not None:
            # Calculate dense optical flow
            flow = cv2.calcOpticalFlowFarneback(
                self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Calculate magnitude and angle
            magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Create visualization
            hsv = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
            hsv[..., 0] = angle * 180 / np.pi / 2
            hsv[..., 1] = 255
            hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
            flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Find regions with significant movement
            magnitude_thresh = cv2.threshold(magnitude, 2.0, 255, cv2.THRESH_BINARY)[1]
            magnitude_thresh = magnitude_thresh.astype(np.uint8)
            
            # Morphological operations
            kernel = np.ones((5, 5), np.uint8)
            magnitude_thresh = cv2.morphologyEx(magnitude_thresh, cv2.MORPH_OPEN, kernel)
            magnitude_thresh = cv2.morphologyEx(magnitude_thresh, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(magnitude_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 300:
                    (x, y), radius = cv2.minEnclosingCircle(cnt)
                    if radius > 10:
                        x, y, radius = int(x), int(y), int(radius)
                        cv2.circle(result_frame, (x, y), radius, (255, 255, 0), 2)
                        cv2.circle(result_frame, (x, y), 3, (255, 255, 0), -1)
                        detections.append({
                            'center': (x, y),
                            'radius': radius,
                            'method': 'Farneback'
                        })
        
        self.prev_gray = gray.copy()
        
        return result_frame, detections, flow_vis
    
    def detect_all(self, frame):
        """Run all detection methods and return results"""
        results = {}
        
        # Method 1: Hough Circles
        frame1, det1 = self.detect_hough_circles(frame)
        results['hough'] = {'frame': frame1, 'detections': det1}
        
        # Method 2: Contours
        frame2, det2, edges = self.detect_contours(frame)
        results['contours'] = {'frame': frame2, 'detections': det2, 'debug': edges}
        
        # Method 3: MOG2
        frame3, det3, mask3 = self.detect_background_subtraction_mog2(frame)
        results['mog2'] = {'frame': frame3, 'detections': det3, 'debug': mask3}
        
        # Method 4: GMG
        frame4, det4, mask4 = self.detect_background_subtraction_gmg(frame)
        results['gmg'] = {'frame': frame4, 'detections': det4, 'debug': mask4}
        
        # Method 5: Frame Difference
        frame5, det5, diff5 = self.detect_frame_difference(frame)
        results['framediff'] = {'frame': frame5, 'detections': det5, 'debug': diff5}
        
        # Method 6: Lucas-Kanade
        frame6, det6 = self.detect_optical_flow_lk(frame)
        results['lk'] = {'frame': frame6, 'detections': det6}
        
        # Method 7: Farneback
        frame7, det7, flow7 = self.detect_optical_flow_farneback(frame)
        results['farneback'] = {'frame': frame7, 'detections': det7, 'debug': flow7}
        
        return results


def resize_for_display(frame, max_width=640, max_height=480):
    """Resize frame for display while maintaining aspect ratio"""
    h, w = frame.shape[:2]
    scale = min(max_width / w, max_height / h, 1.0)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(frame, (new_w, new_h))


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
    
    # Set lower exposure and brightness
    camera.set(cv2.CAP_PROP_EXPOSURE, -6)
    camera.set(cv2.CAP_PROP_BRIGHTNESS, 50)
    
    try:
        camera.set(cv2.CAP_PROP_ISO_SPEED, 100)
    except:
        pass
    
    logger.info("Camera initialized. Press 'q' to quit.")
    logger.info("Multiple windows will show different detection methods.")
    
    detector = MultiMethodDetector()
    
    while True:
        ret, frame = camera.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            break
        
        # Run all detection methods
        results = detector.detect_all(frame)
        
        # Display results in separate windows
        # Original frame
        cv2.imshow("0. Original", resize_for_display(frame))
        
        # Method 1: Hough Circles
        cv2.imshow("1. Hough Circles", resize_for_display(results['hough']['frame']))
        
        # Method 2: Contours
        cv2.imshow("2. Contours (Canny)", resize_for_display(results['contours']['frame']))
        if results['contours']['debug'] is not None:
            cv2.imshow("2b. Contours Debug (Edges)", resize_for_display(results['contours']['debug']))
        
        # Method 3: MOG2
        cv2.imshow("3. MOG2 Background", resize_for_display(results['mog2']['frame']))
        if results['mog2']['debug'] is not None:
            cv2.imshow("3b. MOG2 Mask", resize_for_display(results['mog2']['debug']))
        
        # Method 4: GMG
        cv2.imshow("4. GMG Background", resize_for_display(results['gmg']['frame']))
        if results['gmg']['debug'] is not None:
            cv2.imshow("4b. GMG Mask", resize_for_display(results['gmg']['debug']))
        
        # Method 5: Frame Difference
        cv2.imshow("5. Frame Difference", resize_for_display(results['framediff']['frame']))
        if results['framediff']['debug'] is not None:
            cv2.imshow("5b. Frame Diff Mask", resize_for_display(results['framediff']['debug']))
        
        # Method 6: Lucas-Kanade
        cv2.imshow("6. Lucas-Kanade Flow", resize_for_display(results['lk']['frame']))
        
        # Method 7: Farneback
        cv2.imshow("7. Farneback Flow", resize_for_display(results['farneback']['frame']))
        if results['farneback']['debug'] is not None:
            cv2.imshow("7b. Farneback Flow Vis", resize_for_display(results['farneback']['debug']))
        
        # Print detection counts
        print(f"\rHough: {len(results['hough']['detections'])} | "
              f"Contours: {len(results['contours']['detections'])} | "
              f"MOG2: {len(results['mog2']['detections'])} | "
              f"GMG: {len(results['gmg']['detections'])} | "
              f"FrameDiff: {len(results['framediff']['detections'])} | "
              f"LK: {len(results['lk']['detections'])} | "
              f"Farneback: {len(results['farneback']['detections'])}", end='')
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
    # Cleanup
    camera.release()
    cv2.destroyAllWindows()
    logger.info("\nCamera released, windows closed")


if __name__ == "__main__":
    main()


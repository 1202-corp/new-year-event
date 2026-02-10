"""Face capture module for saving audience faces using YOLO"""
import cv2
import numpy as np
import os
from pathlib import Path
from typing import Optional, List, Tuple
from game.config import Config
from game.logger import get_logger

logger = get_logger()

# Try to import YOLO
try:
    from ultralytics import YOLO
    import torch
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    torch = None
    logger.warning("ultralytics not available. Face detection will not work.")


class FaceCapture:
    """
    Captures and saves audience faces using YOLO person detection.
    
    Shows debug window with audience camera feed.
    Press SPACE to detect faces and save them as square images.
    """
    
    def __init__(self, ui_panel=None):
        """
        Initialize face capture
        
        Args:
            ui_panel: UIPanel instance to get camera frames from (optional)
        """
        self.camera = None
        self.ui_panel = ui_panel  # Use UI panel camera if available
        self.yolo_model = None
        self.output_dir = Path("faces")
        self.capture_count = 0
        self.window_created = False
        self.last_frame = None  # Keep last frame to prevent flickering
        
        # Create output directory
        self.output_dir.mkdir(exist_ok=True)
        logger.info(f"Face capture output directory: {self.output_dir.absolute()}")
        
        # Don't initialize camera if UI panel is provided (will use its camera)
        if self.ui_panel is None:
            self._init_camera()
        
        # Initialize YOLO model
        if YOLO_AVAILABLE:
            self._init_yolo()
        else:
            logger.error("YOLO not available. Face detection will not work.")
        
        # Check if camera is available
        if self.ui_panel is not None:
            logger.info("Face capture will use camera from UI panel")
        elif self.camera is None or not self.camera.isOpened():
            logger.warning("Face capture camera is not available. Face capture will not work.")
        else:
            logger.info("Face capture initialized successfully")
    
    def _init_camera(self) -> None:
        """Initialize audience camera"""
        try:
            # Use V4L2 backend on Linux, DSHOW on Windows
            import platform
            if platform.system() == "Linux":
                backend = cv2.CAP_V4L2
            else:
                backend = cv2.CAP_DSHOW
            
            self.camera = cv2.VideoCapture(Config.AUDIENCE_CAMERA_INDEX, backend)
            
            if not self.camera.isOpened():
                logger.error(f"Failed to open audience camera {Config.AUDIENCE_CAMERA_INDEX}")
                return
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, Config.AUDIENCE_CAMERA_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.AUDIENCE_CAMERA_HEIGHT)
            
            # Set MJPEG format if available
            self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            
            # Get actual resolution
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info(f"Audience camera initialized: {actual_width}x{actual_height}")
            
        except Exception as e:
            logger.error(f"Error initializing audience camera: {e}")
            self.camera = None
    
    def _init_yolo(self) -> None:
        """Initialize YOLO model for person detection"""
        try:
            # Use standard YOLO model (yolov8n.pt) for person detection
            model_path = Path("models/yolov8n.pt")
            
            if not model_path.exists():
                logger.error(f"YOLO model not found at {model_path}")
                return
            
            self.yolo_model = YOLO(str(model_path))
            
            # Set device
            device = 'cuda' if torch and torch.cuda.is_available() else 'cpu'
            logger.info(f"Face detection YOLO using device: {device}")
            
        except Exception as e:
            logger.error(f"Error initializing YOLO model: {e}")
            self.yolo_model = None
    
    def _detect_persons(self, frame: np.ndarray) -> List[Tuple[int, int, int, int, float]]:
        """
        Detect persons in frame using YOLO
        
        Returns:
            List of (x1, y1, x2, y2, confidence) bounding boxes
        """
        if self.yolo_model is None:
            return []
        
        try:
            # Run YOLO inference
            results = self.yolo_model(frame, conf=0.25, classes=[0])  # Class 0 = person
            
            detections = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    detections.append((int(x1), int(y1), int(x2), int(y2), confidence))
            
            return detections
            
        except Exception as e:
            logger.error(f"Error detecting persons: {e}")
            return []
    
    def _extract_face_region(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Extract face region from person bounding box as square
        
        Args:
            frame: Input frame
            bbox: (x1, y1, x2, y2) bounding box
            
        Returns:
            Square face image or None
        """
        x1, y1, x2, y2 = bbox
        
        # Get person region
        person_width = x2 - x1
        person_height = y2 - y1
        
        # Estimate face region (upper part of person, centered horizontally)
        # Face is typically in upper 1/3 of person bounding box
        face_y1 = y1
        face_y2 = y1 + int(person_height * 0.4)  # Upper 40% of person
        
        # Center face horizontally
        face_center_x = (x1 + x2) // 2
        face_width = min(person_width, person_height)  # Use smaller dimension
        face_half_width = face_width // 2
        
        face_x1 = max(0, face_center_x - face_half_width)
        face_x2 = min(frame.shape[1], face_center_x + face_half_width)
        face_y1 = max(0, face_y1)
        face_y2 = min(frame.shape[0], face_y2)
        
        # Extract face region
        face_region = frame[face_y1:face_y2, face_x1:face_x2]
        
        if face_region.size == 0:
            return None
        
        # Make it square by cropping or padding
        face_height = face_y2 - face_y1
        face_width = face_x2 - face_x1
        
        if face_width > face_height:
            # Wider than tall - crop width
            crop = (face_width - face_height) // 2
            face_region = face_region[:, crop:crop + face_height]
        elif face_height > face_width:
            # Taller than wide - crop height
            crop = (face_height - face_width) // 2
            face_region = face_region[crop:crop + face_width, :]
        
        # Resize to standard size (e.g., 256x256)
        face_square = cv2.resize(face_region, (256, 256))
        
        return face_square
    
    def _save_faces(self, frame: np.ndarray, detections: List[Tuple[int, int, int, int, float]]) -> int:
        """
        Save detected faces as square images
        
        Returns:
            Number of faces saved
        """
        saved_count = 0
        
        for i, (x1, y1, x2, y2, confidence) in enumerate(detections):
            # Extract face region
            face_image = self._extract_face_region(frame, (x1, y1, x2, y2))
            
            if face_image is None:
                continue
            
            # Save face image
            self.capture_count += 1
            filename = self.output_dir / f"face_{self.capture_count:05d}.jpg"
            cv2.imwrite(str(filename), face_image)
            saved_count += 1
            logger.info(f"Saved face: {filename} (confidence: {confidence:.2f})")
        
        return saved_count
    
    def update(self) -> Optional[int]:
        """
        Update face capture - read frame and handle keyboard input
        
        Returns:
            Key code if key was pressed, None otherwise
        """
        # Get frame from UI panel if available, otherwise from direct camera
        frame = None
        
        if self.ui_panel is not None:
            # Get frame from UI panel camera
            if self.ui_panel.camera_thread and self.ui_panel.camera_thread.camera_enabled:
                frame = self.ui_panel.camera_thread.get_latest_frame()
        elif self.camera is not None and self.camera.isOpened():
            # Get frame from direct camera access
            ret, frame_read = self.camera.read()
            if ret:
                frame = frame_read
        
        # Use last frame if new one is not available (prevent flickering)
        if frame is not None:
            self.last_frame = frame
        elif self.last_frame is not None:
            frame = self.last_frame
        
        # Show window even if no frame available
        if frame is None:
            # Create a black frame with error message (smaller size)
            display_frame = np.zeros((240, 320, 3), dtype=np.uint8)
            error_text = [
                "Face Capture Debug",
                "",
                "Camera not available!",
                f"Camera index: {Config.AUDIENCE_CAMERA_INDEX}",
                "",
                "Press 'q' to quit"
            ]
            y_offset = 20
            for i, text in enumerate(error_text):
                cv2.putText(display_frame, text, (5, y_offset + i * 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            # Resize window to smaller size
            cv2.imshow("Face Capture Debug", display_frame)
            if not self.window_created:
                self.window_created = True
                logger.info("Face Capture Debug window created (camera unavailable)")
            
            # Handle keyboard input even without camera
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return -1
            return key if key != 255 else None
        
        # Resize frame to smaller size (half the original)
        height, width = frame.shape[:2]
        small_frame = cv2.resize(frame, (width // 2, height // 2))
        
        # Draw info on frame
        display_frame = small_frame.copy()
        
        info_text = [
            "Face Capture Debug",
            f"Faces saved: {self.capture_count}",
            "",
            "Press SPACE to detect and save faces",
            "Press 'q' to quit"
        ]
        
        y_offset = 15
        for i, text in enumerate(info_text):
            cv2.putText(display_frame, text, (5, y_offset + i * 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 0), 1)
        
        # Show frame
        cv2.imshow("Face Capture Debug", display_frame)
        if not self.window_created:
            self.window_created = True
            logger.info("Face Capture Debug window created")
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return -1  # Quit signal
        elif key == ord(' '):  # SPACE
            # Detect and save faces (use original frame, not resized)
            detections = self._detect_persons(frame)
            
            if detections:
                # Draw detections on display frame (scaled coordinates)
                for x1, y1, x2, y2, confidence in detections:
                    # Scale coordinates to match resized display
                    x1_small = x1 // 2
                    y1_small = y1 // 2
                    x2_small = x2 // 2
                    y2_small = y2 // 2
                    cv2.rectangle(display_frame, (x1_small, y1_small), (x2_small, y2_small), (0, 255, 0), 1)
                    cv2.putText(display_frame, f"Person {confidence:.2f}", (x1_small, y1_small - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
                
                # Save faces
                saved_count = self._save_faces(frame, detections)
                
                # Show success message
                cv2.putText(display_frame, f"Saved {saved_count} face(s)!", (5, display_frame.shape[0] - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                
                logger.info(f"Detected {len(detections)} person(s), saved {saved_count} face(s)")
            else:
                cv2.putText(display_frame, "No persons detected", (5, display_frame.shape[0] - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                logger.info("No persons detected")
            
            # Show updated frame
            cv2.imshow("Face Capture Debug", display_frame)
            cv2.waitKey(500)  # Show message for 500ms
        
        return key if key != 255 else None
    
    def release(self) -> None:
        """Release camera resources"""
        # Only release camera if we opened it directly (not from UI panel)
        if self.camera is not None and self.ui_panel is None:
            self.camera.release()
        cv2.destroyAllWindows()
        logger.info("Face capture released")


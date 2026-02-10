"""UI Panel module for bottom RPG-style panel"""
import pygame
import threading
import queue
from typing import Optional
try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
from game.config import Config
from game.constants import DARK_BLUE, WHITE, BLACK, GRAY
from game.scaling import get_scaling
from game.logger import get_logger

logger = get_logger()


class CameraThread(threading.Thread):
    """Thread for reading camera frames asynchronously"""
    
    def __init__(self, camera_index: int, camera_width: int, camera_height: int):
        super().__init__(daemon=True)
        self.camera_index = camera_index
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.camera = None
        self.frame_queue = queue.Queue(maxsize=2)  # Keep only latest 2 frames
        self.running = False
        self.camera_enabled = False
    
    def run(self):
        """Main thread loop for reading camera frames"""
        if not CV2_AVAILABLE:
            return
        
        try:
            # Use platform-appropriate backend (DirectShow on Windows, V4L2 on Linux)
            import platform
            system = platform.system()
            backend = cv2.CAP_DSHOW if system == "Windows" else (cv2.CAP_V4L2 if system == "Linux" else 0)
            self.camera = cv2.VideoCapture(self.camera_index, backend)
            if self.camera.isOpened():
                # Set camera format to MJPEG (compressed) instead of RAW (uncompressed)
                # MJPEG is much faster and lighter than RAW format
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                self.camera.set(cv2.CAP_PROP_FOURCC, fourcc)
                
                # Set camera resolution
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
                
                self.camera_enabled = True
                
                # Verify format
                current_fourcc = int(self.camera.get(cv2.CAP_PROP_FOURCC))
                fourcc_str = "".join([chr((current_fourcc >> 8 * i) & 0xFF) for i in range(4)])
                logger.info(f"Camera {self.camera_index} initialized in thread")
                logger.info(f"Camera format: {fourcc_str} (should be MJPG for MJPEG)")
            else:
                logger.warning(f"Could not open camera {self.camera_index}")
                return
        except Exception as e:
            logger.warning(f"Camera initialization failed: {e}")
            return
        
        self.running = True
        while self.running:
            try:
                ret, frame = self.camera.read()
                if ret:
                    # Clear old frames, keep only latest
                    while not self.frame_queue.empty():
                        try:
                            self.frame_queue.get_nowait()
                        except queue.Empty:
                            break
                    self.frame_queue.put(frame)
            except Exception as e:
                logger.debug(f"Error reading camera frame: {e}")
        
        if self.camera is not None:
            self.camera.release()
            self.camera = None
    
    def stop(self):
        """Stop the camera thread"""
        self.running = False
    
    def get_latest_frame(self) -> Optional['np.ndarray']:
        """Get the latest frame from queue (non-blocking)"""
        try:
            return self.frame_queue.get_nowait()
        except queue.Empty:
            return None


class UIPanel:
    """Vertical UI panel on the right side with information and camera preview"""
    
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        # Panel is now vertical on the right side
        self.panel_width = int(screen_width * Config.UI_PANEL_WIDTH_PERCENT / 100)
        self.panel_x = screen_width - self.panel_width
        # Keep panel_height for backward compatibility (full screen height)
        self.panel_height = screen_height
        
        # Camera settings - size based on panel width (square, with padding)
        self.camera_window_size = self.panel_width - 40  # Leave 20px padding on left and right
        self.camera_thread = None
        self.current_frame_surface = None
        
        # Start camera thread
        self._init_camera()
        
        # Fonts
        scaling = get_scaling()
        self.font_large = pygame.font.Font(None, scaling.scale_font_size(36))
        self.font_medium = pygame.font.Font(None, scaling.scale_font_size(24))
        self.font_small = pygame.font.Font(None, scaling.scale_font_size(18))
    
    def _init_camera(self) -> None:
        """Initialize camera thread for audience preview"""
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available, camera preview disabled")
            return
        
        try:
            self.camera_thread = CameraThread(
                Config.AUDIENCE_CAMERA_INDEX,
                Config.AUDIENCE_CAMERA_WIDTH,
                Config.AUDIENCE_CAMERA_HEIGHT
            )
            self.camera_thread.start()
            logger.info("Camera thread started")
        except Exception as e:
            logger.warning(f"Camera thread initialization failed: {e}")
    
    def update(self, screen_width: int, screen_height: int) -> None:
        """Update panel dimensions"""
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Apply safe area margin to panel position
        from game.safe_area import get_safe_area_margin
        margin = get_safe_area_margin(screen_width, screen_height, 0)
        
        # Panel width and position (accounting for safe area) - vertical panel on right
        self.panel_width = int((screen_width - margin * 2) * Config.UI_PANEL_WIDTH_PERCENT / 100)
        self.panel_x = screen_width - self.panel_width - margin  # Position from right edge
        # Keep panel_height for backward compatibility (full screen height)
        self.panel_height = screen_height
        
        # Update camera window size based on panel width (square, with padding)
        self.camera_window_size = self.panel_width - 40  # Leave 20px padding on left and right
        
        # Update fonts
        scaling = get_scaling()
        self.font_large = pygame.font.Font(None, scaling.scale_font_size(36))
        self.font_medium = pygame.font.Font(None, scaling.scale_font_size(24))
        self.font_small = pygame.font.Font(None, scaling.scale_font_size(18))
    
    def process_camera_frame(self) -> None:
        """Process camera frame asynchronously (non-blocking) - call this every frame"""
        if self.camera_thread and self.camera_thread.camera_enabled:
            frame = self.camera_thread.get_latest_frame()
            if frame is not None:
                self._process_camera_frame(frame)
    
    def _process_camera_frame(self, frame: 'np.ndarray') -> None:
        """Process camera frame: crop to square and convert to pygame Surface"""
        if not CV2_AVAILABLE:
            return
        
        try:
            # Get frame dimensions
            frame_height, frame_width = frame.shape[:2]
            
            # Calculate square crop (center crop)
            size = min(frame_width, frame_height)
            start_x = (frame_width - size) // 2
            start_y = (frame_height - size) // 2
            
            # Crop to square
            frame_cropped = frame[start_y:start_y + size, start_x:start_x + size]
            
            # Mirror horizontally (flip left-right)
            frame_mirrored = cv2.flip(frame_cropped, 1)
            
            # Resize to camera window size (maintain aspect ratio, but already square)
            frame_resized = cv2.resize(frame_mirrored, (self.camera_window_size, self.camera_window_size))
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            
            # Convert to pygame Surface
            self.current_frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        except Exception as e:
            logger.debug(f"Error processing camera frame: {e}")
            self.current_frame_surface = None
    
    def draw(self, screen: pygame.Surface, enemies_count: int, max_enemies: int) -> None:
        """Draw the UI panel (vertical, on right side)"""
        # Apply safe area margin
        from game.safe_area import get_safe_area_margin
        margin = get_safe_area_margin(self.screen_width, self.screen_height, 0)
        
        # Draw panel background (vertical, on right side, respecting safe area margins)
        panel_rect = pygame.Rect(self.panel_x, margin, self.panel_width, self.screen_height - margin * 2)
        pygame.draw.rect(screen, DARK_BLUE, panel_rect)
        pygame.draw.rect(screen, WHITE, panel_rect, 2)  # Border
        
        # Camera preview is now only shown in debug window, not in UI panel
        # Camera still runs in background for face capture debug window
        
        # Draw information - use scaled padding
        scaling = get_scaling()
        padding = int(scaling.scale_value(20))  # Scale padding based on screen size
        line_spacing = int(scaling.scale_value(40))  # Scale line spacing
        
        info_x = self.panel_x + padding
        info_y = margin + padding  # Start from top with padding
        
        # Enemies count
        enemies_text = self.font_medium.render(f"Enemies: {enemies_count}/{max_enemies}", True, WHITE)
        screen.blit(enemies_text, (info_x, info_y))
    
    def cleanup(self) -> None:
        """Cleanup camera resources"""
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread.join(timeout=1.0)
            self.camera_thread = None

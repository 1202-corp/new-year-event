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
            self.camera = cv2.VideoCapture(self.camera_index)
            if self.camera.isOpened():
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera_width)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera_height)
                self.camera_enabled = True
                logger.info(f"Camera {self.camera_index} initialized in thread")
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
    """Bottom UI panel with information and camera preview"""
    
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.panel_height = int(screen_height * Config.UI_PANEL_HEIGHT_PERCENT / 100)
        self.panel_y = screen_height - self.panel_height
        
        # Camera settings - size based on panel height (square, with padding)
        self.camera_window_size = self.panel_height - 40  # Leave 20px padding on top and bottom
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
        
        # Panel height and position (accounting for safe area)
        self.panel_height = int((screen_height - margin * 2) * Config.UI_PANEL_HEIGHT_PERCENT / 100)
        self.panel_y = screen_height - self.panel_height - margin  # Position above bottom margin
        
        # Update camera window size based on panel height (square, with padding)
        self.camera_window_size = self.panel_height - 40  # Leave 20px padding on top and bottom
        
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
    
    def draw(self, screen: pygame.Surface, score: int, enemies_count: int, max_enemies: int) -> None:
        """Draw the UI panel"""
        # Apply safe area margin
        from game.safe_area import get_safe_area_margin
        margin = get_safe_area_margin(self.screen_width, self.screen_height, 0)
        
        # Draw panel background (respecting safe area margins)
        panel_rect = pygame.Rect(margin, self.panel_y, self.screen_width - margin * 2, self.panel_height)
        pygame.draw.rect(screen, DARK_BLUE, panel_rect)
        pygame.draw.rect(screen, WHITE, panel_rect, 2)  # Border
        
        # Draw camera preview (right side, bottom-aligned, respecting safe area)
        # Position: right edge with padding, vertically centered in panel
        camera_x = self.screen_width - self.camera_window_size - 20 - margin  # 20px padding + margin from right edge
        camera_y = self.panel_y + (self.panel_height - self.camera_window_size) // 2  # Centered vertically
        
        if self.current_frame_surface:
            screen.blit(self.current_frame_surface, (camera_x, camera_y))
        else:
            # Draw placeholder if camera not available
            camera_rect = pygame.Rect(camera_x, camera_y, self.camera_window_size, self.camera_window_size)
            pygame.draw.rect(screen, GRAY, camera_rect)
            pygame.draw.rect(screen, WHITE, camera_rect, 2)
            placeholder_text = self.font_small.render("Camera", True, WHITE)
            text_rect = placeholder_text.get_rect(center=camera_rect.center)
            screen.blit(placeholder_text, text_rect)
        
        # Draw information (left side) - use scaled padding
        scaling = get_scaling()
        padding = int(scaling.scale_value(20))  # Scale padding based on screen size
        line_spacing = int(scaling.scale_value(40))  # Scale line spacing
        
        info_x = margin + padding
        info_y = self.panel_y + padding
        
        # Score
        score_text = self.font_large.render(f"Score: {score}", True, WHITE)
        screen.blit(score_text, (info_x, info_y))
        
        # Enemies count
        enemies_text = self.font_medium.render(f"Enemies: {enemies_count}/{max_enemies}", True, WHITE)
        screen.blit(enemies_text, (info_x, info_y + line_spacing))
    
    def cleanup(self) -> None:
        """Cleanup camera resources"""
        if self.camera_thread:
            self.camera_thread.stop()
            self.camera_thread.join(timeout=1.0)
            self.camera_thread = None

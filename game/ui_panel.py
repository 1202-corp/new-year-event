"""UI Panel module for bottom RPG-style panel"""
import pygame
from typing import Optional
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
from game.config import Config
from game.constants import DARK_BLUE, WHITE, BLACK, GRAY
from game.scaling import get_scaling
from game.logger import get_logger

logger = get_logger()


class UIPanel:
    """Bottom UI panel with information and camera preview"""
    
    def __init__(self, screen_width: int, screen_height: int):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.panel_height = int(screen_height * Config.UI_PANEL_HEIGHT_PERCENT / 100)
        self.panel_y = screen_height - self.panel_height
        
        # Camera settings
        self.camera = None
        self.camera_enabled = False
        self.camera_window_size = int(Config.CAMERA_WINDOW_SIZE)
        self._init_camera()
        
        # Fonts
        scaling = get_scaling()
        self.font_large = pygame.font.Font(None, scaling.scale_font_size(36))
        self.font_medium = pygame.font.Font(None, scaling.scale_font_size(24))
        self.font_small = pygame.font.Font(None, scaling.scale_font_size(18))
    
    def _init_camera(self) -> None:
        """Initialize camera for preview"""
        if not CV2_AVAILABLE:
            logger.warning("OpenCV not available, camera preview disabled")
            return
        try:
            # Try to open camera (index from config)
            self.camera = cv2.VideoCapture(Config.CAMERA_INDEX)
            if self.camera.isOpened():
                self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAMERA_WIDTH)
                self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAMERA_HEIGHT)
                self.camera_enabled = True
                logger.info(f"Camera {Config.CAMERA_INDEX} initialized successfully")
            else:
                logger.warning(f"Could not open camera {Config.CAMERA_INDEX}")
        except Exception as e:
            logger.warning(f"Camera initialization failed: {e}")
    
    def update(self, screen_width: int, screen_height: int) -> None:
        """Update panel dimensions"""
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.panel_height = int(screen_height * Config.UI_PANEL_HEIGHT_PERCENT / 100)
        self.panel_y = screen_height - self.panel_height
        
        # Update fonts
        scaling = get_scaling()
        self.font_large = pygame.font.Font(None, scaling.scale_font_size(36))
        self.font_medium = pygame.font.Font(None, scaling.scale_font_size(24))
        self.font_small = pygame.font.Font(None, scaling.scale_font_size(18))
    
    def get_camera_frame(self) -> Optional[pygame.Surface]:
        """Get current camera frame as pygame Surface"""
        if not CV2_AVAILABLE or not self.camera_enabled or self.camera is None:
            return None
        
        try:
            ret, frame = self.camera.read()
            if not ret:
                return None
            
            # Resize to camera window size
            frame = cv2.resize(frame, (self.camera_window_size, self.camera_window_size))
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Convert to pygame Surface
            frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            
            return frame_surface
        except Exception as e:
            logger.debug(f"Error reading camera frame: {e}")
            return None
    
    def draw(self, screen: pygame.Surface, score: int, enemies_count: int, max_enemies: int) -> None:
        """Draw the UI panel"""
        # Draw panel background
        panel_rect = pygame.Rect(0, self.panel_y, self.screen_width, self.panel_height)
        pygame.draw.rect(screen, DARK_BLUE, panel_rect)
        pygame.draw.rect(screen, WHITE, panel_rect, 2)  # Border
        
        # Draw camera preview (right side)
        camera_x = self.screen_width - self.camera_window_size - 20
        camera_y = self.panel_y + (self.panel_height - self.camera_window_size) // 2
        
        camera_frame = self.get_camera_frame()
        if camera_frame:
            screen.blit(camera_frame, (camera_x, camera_y))
        else:
            # Draw placeholder if camera not available
            camera_rect = pygame.Rect(camera_x, camera_y, self.camera_window_size, self.camera_window_size)
            pygame.draw.rect(screen, GRAY, camera_rect)
            pygame.draw.rect(screen, WHITE, camera_rect, 2)
            placeholder_text = self.font_small.render("Camera", True, WHITE)
            text_rect = placeholder_text.get_rect(center=camera_rect.center)
            screen.blit(placeholder_text, text_rect)
        
        # Draw information (left side)
        info_x = 20
        info_y = self.panel_y + 20
        
        # Score
        score_text = self.font_large.render(f"Score: {score}", True, WHITE)
        screen.blit(score_text, (info_x, info_y))
        
        # Enemies count
        enemies_text = self.font_medium.render(f"Enemies: {enemies_count}/{max_enemies}", True, WHITE)
        screen.blit(enemies_text, (info_x, info_y + 40))
    
    def cleanup(self) -> None:
        """Cleanup camera resources"""
        if self.camera is not None:
            self.camera.release()
            self.camera = None


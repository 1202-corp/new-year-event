"""Configuration module for loading settings from .env file"""
import os
from typing import Optional
from dotenv import load_dotenv
from game.logger import get_logger

# Load .env file
load_dotenv()

logger = get_logger()


class Config:
    """Configuration class for game settings"""
    
    # Display settings
    DISPLAY_NUMBER: int = int(os.getenv("DISPLAY_NUMBER", "0"))
    
    # Screen settings
    SCREEN_WIDTH: int = int(os.getenv("SCREEN_WIDTH", "1280"))
    SCREEN_HEIGHT: int = int(os.getenv("SCREEN_HEIGHT", "720"))
    FPS: int = int(os.getenv("FPS", "60"))
    FULLSCREEN: bool = os.getenv("FULLSCREEN", "false").lower() == "true"
    
    # Safe area settings (percentage, 0-100)
    SAFE_AREA_MARGIN_PERCENT: float = float(os.getenv("SAFE_AREA_MARGIN_PERCENT", "3.9"))
    
    # UI Panel settings
    UI_PANEL_WIDTH_PERCENT: float = float(os.getenv("UI_PANEL_WIDTH_PERCENT", "20"))
    # Camera window size is now calculated dynamically from panel width
    
    # Gameplay settings
    MAX_ENEMIES: int = int(os.getenv("MAX_ENEMIES", "20"))
    NUM_LINES: int = int(os.getenv("NUM_LINES", "3"))
    ENEMY_WIDTH: int = int(os.getenv("ENEMY_WIDTH", "60"))
    ENEMY_HEIGHT: int = int(os.getenv("ENEMY_HEIGHT", "60"))
    
    # Camera settings for snowball detection
    SNOWBALL_CAMERA_INDEX: int = int(os.getenv("SNOWBALL_CAMERA_INDEX", "0"))
    SNOWBALL_CAMERA_WIDTH: int = int(os.getenv("SNOWBALL_CAMERA_WIDTH", "1920"))
    SNOWBALL_CAMERA_HEIGHT: int = int(os.getenv("SNOWBALL_CAMERA_HEIGHT", "1080"))
    
    # Camera quality settings (for snowball/Aruco detection camera)
    SNOWBALL_CAMERA_EXPOSURE: float = float(os.getenv("SNOWBALL_CAMERA_EXPOSURE", "-20"))
    SNOWBALL_CAMERA_BRIGHTNESS: float = float(os.getenv("SNOWBALL_CAMERA_BRIGHTNESS", "42.0"))
    SNOWBALL_CAMERA_CONTRAST: float = float(os.getenv("SNOWBALL_CAMERA_CONTRAST", "128.0"))
    SNOWBALL_CAMERA_SATURATION: float = float(os.getenv("SNOWBALL_CAMERA_SATURATION", "53.0"))
    SNOWBALL_CAMERA_SHARPNESS: float = float(os.getenv("SNOWBALL_CAMERA_SHARPNESS", "255.0"))
    SNOWBALL_CAMERA_GAIN: float = float(os.getenv("SNOWBALL_CAMERA_GAIN", "88.0"))
    SNOWBALL_CAMERA_FOCUS: float = float(os.getenv("SNOWBALL_CAMERA_FOCUS", "0.0"))
    
    # Camera settings for audience face display
    AUDIENCE_CAMERA_INDEX: int = int(os.getenv("AUDIENCE_CAMERA_INDEX", "1"))
    AUDIENCE_CAMERA_WIDTH: int = int(os.getenv("AUDIENCE_CAMERA_WIDTH", "1920"))
    AUDIENCE_CAMERA_HEIGHT: int = int(os.getenv("AUDIENCE_CAMERA_HEIGHT", "1080"))
    
    # Aruco settings
    ARUCO_DICT_TYPE: int = int(os.getenv("ARUCO_DICT_TYPE", "4"))
    ARUCO_MARKER_SIZE: int = int(os.getenv("ARUCO_MARKER_SIZE", "100"))
    
    # Game settings
    CHARACTER_SPAWN_INTERVAL: float = float(os.getenv("CHARACTER_SPAWN_INTERVAL", "2.0"))
    
    # Vision settings
    VISION_ENABLED: bool = os.getenv("VISION_ENABLED", "false").lower() == "true"
    CALIBRATION_ENABLED: bool = os.getenv("CALIBRATION_ENABLED", "false").lower() == "true"
    CAMERA_DELAY_MS: float = float(os.getenv("CAMERA_DELAY_MS", "100"))  # Camera delay in milliseconds
    BALL_COLLISION_RADIUS: int = int(os.getenv("BALL_COLLISION_RADIUS", "100"))  # Collision detection radius in pixels
    YOLO_CONFIDENCE_THRESHOLD: float = float(os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.25"))  # YOLO confidence threshold (0.0-1.0)
    YOLO_IMAGE_SIZE: int = int(os.getenv("YOLO_IMAGE_SIZE", "640"))  # YOLO input image size (320, 416, 640)
    YOLO_MAX_DETECTIONS: int = int(os.getenv("YOLO_MAX_DETECTIONS", "10"))  # Maximum number of detections per frame
    YOLO_ASYNC_PROCESSING: bool = os.getenv("YOLO_ASYNC_PROCESSING", "true").lower() == "true"  # Use async processing
    
    @classmethod
    def setup_display(cls) -> None:
        """Setup display environment variable for multi-monitor support"""
        # Set DISPLAY environment variable BEFORE pygame.init()
        # This is critical - must be set before any SDL/pygame initialization
        original_display = os.environ.get("DISPLAY", ":0.0")
        
        if cls.DISPLAY_NUMBER != 0:
            # For X11, format is :display.screen
            # DISPLAY_NUMBER=1 means :1.0 (display 1, screen 0)
            display_var = f":{cls.DISPLAY_NUMBER}.0"
            os.environ["DISPLAY"] = display_var
            logger.info(f"Setting DISPLAY environment variable to: {display_var}")
            logger.info(f"Original DISPLAY was: {original_display}")
            logger.info(f"New DISPLAY is: {os.environ.get('DISPLAY')}")
    
    @classmethod
    def get_display_env(cls) -> Optional[str]:
        """Returns DISPLAY environment variable if set"""
        return os.environ.get("DISPLAY")


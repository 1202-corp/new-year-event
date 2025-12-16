"""Configuration module for loading settings from .env file"""
import os
from typing import Optional
from dotenv import load_dotenv

# Load .env file
load_dotenv()


class Config:
    """Configuration class for game settings"""
    
    # Display settings
    DISPLAY_NUMBER: int = int(os.getenv("DISPLAY_NUMBER", "0"))
    
    # Screen settings
    SCREEN_WIDTH: int = int(os.getenv("SCREEN_WIDTH", "1280"))
    SCREEN_HEIGHT: int = int(os.getenv("SCREEN_HEIGHT", "720"))
    FPS: int = int(os.getenv("FPS", "60"))
    
    # Camera settings
    CAMERA_INDEX: int = int(os.getenv("CAMERA_INDEX", "0"))
    CAMERA_WIDTH: int = int(os.getenv("CAMERA_WIDTH", "1920"))
    CAMERA_HEIGHT: int = int(os.getenv("CAMERA_HEIGHT", "1080"))
    
    # Aruco settings
    ARUCO_DICT_TYPE: int = int(os.getenv("ARUCO_DICT_TYPE", "4"))
    ARUCO_MARKER_SIZE: int = int(os.getenv("ARUCO_MARKER_SIZE", "100"))
    
    # Game settings
    CHARACTER_SPAWN_INTERVAL: float = float(os.getenv("CHARACTER_SPAWN_INTERVAL", "2.0"))
    SNOWBALL_MIN_AREA: int = int(os.getenv("SNOWBALL_MIN_AREA", "50"))
    SNOWBALL_MAX_AREA: int = int(os.getenv("SNOWBALL_MAX_AREA", "5000"))
    
    # Vision settings
    VISION_ENABLED: bool = os.getenv("VISION_ENABLED", "false").lower() == "true"
    CALIBRATION_ENABLED: bool = os.getenv("CALIBRATION_ENABLED", "false").lower() == "true"
    
    @classmethod
    def setup_display(cls) -> None:
        """Setup display environment variable for multi-monitor support"""
        if cls.DISPLAY_NUMBER != 0:
            # Set DISPLAY environment variable for Linux/X11
            # Format: :0.0, :0.1, :1.0, etc.
            display_var = f":{cls.DISPLAY_NUMBER}.0"
            os.environ["DISPLAY"] = display_var
    
    @classmethod
    def get_display_env(cls) -> Optional[str]:
        """Returns DISPLAY environment variable if set"""
        return os.environ.get("DISPLAY")


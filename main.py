"""Main entry point for New Year Arcade Game"""
import os
import sys

# CRITICAL: Disable Qt backend in OpenCV to prevent connection errors
# OpenCV will try to use Qt for GUI, but Qt can't connect to display :2.0
# Remove QT_PLUGIN_PATH to avoid plugin loading issues
os.environ.pop("QT_PLUGIN_PATH", None)
# Try to use GTK backend instead of Qt (if available)
# This must be set before importing cv2
try:
    import cv2
    # Check if we can use GTK backend
    os.environ["OPENCV_GUI_BACKEND"] = "GTK"
except:
    pass

# CRITICAL: Load .env and set DISPLAY BEFORE importing pygame
# This must be done before any pygame imports
from dotenv import load_dotenv
load_dotenv()

# Setup logging first
from game.logger import setup_logging
logger = setup_logging()

# Set DISPLAY environment variable if needed
display_number = int(os.getenv("DISPLAY_NUMBER", "0"))
if display_number != 0:
    display_var = f":{display_number}.0"
    os.environ["DISPLAY"] = display_var
    logger.info(f"Setting DISPLAY to: {display_var}")
    logger.info(f"Current DISPLAY: {os.environ.get('DISPLAY')}")

# Now we can import game modules (which import pygame)
from game.config import Config
from game.game import Game


def main():
    """Entry point"""
    # Verify configuration
    logger.info(f"DISPLAY_NUMBER from config: {Config.DISPLAY_NUMBER}")
    logger.info(f"DISPLAY environment variable: {os.environ.get('DISPLAY')}")
    
    # Create and run game
    game = Game()
    game.run()


if __name__ == "__main__":
    main()

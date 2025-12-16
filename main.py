"""Main entry point for New Year Arcade Game"""
import os
import sys

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

# Set Qt platform plugin to xcb for OpenCV windows
# This must be set before importing cv2
os.environ["QT_QPA_PLATFORM"] = "xcb"
# Also set for OpenCV GUI backend
os.environ["OPENCV_GUI_BACKEND"] = "GTK"  # Try GTK instead of Qt

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

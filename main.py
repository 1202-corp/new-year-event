"""Main entry point for New Year Arcade Game"""
import os
import sys

# CRITICAL: Load .env and set DISPLAY BEFORE importing pygame
# This must be done before any pygame imports
from dotenv import load_dotenv
load_dotenv()

# Set DISPLAY environment variable if needed
display_number = int(os.getenv("DISPLAY_NUMBER", "0"))
if display_number != 0:
    display_var = f":{display_number}.0"
    os.environ["DISPLAY"] = display_var
    print(f"[Main] Setting DISPLAY to: {display_var}")
    print(f"[Main] Current DISPLAY: {os.environ.get('DISPLAY')}")

# Now we can import game modules (which import pygame)
from game.config import Config
from game.game import Game


def main():
    """Entry point"""
    # Verify configuration
    print(f"[Main] DISPLAY_NUMBER from config: {Config.DISPLAY_NUMBER}")
    print(f"[Main] DISPLAY environment variable: {os.environ.get('DISPLAY')}")
    
    # Create and run game
    game = Game()
    game.run()


if __name__ == "__main__":
    main()

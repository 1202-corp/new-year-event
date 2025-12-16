"""Game constants"""

# Screen settings
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 60

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
LIGHT_BLUE = (173, 216, 230)
DARK_BLUE = (20, 30, 60)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)

# Character settings
CHARACTER_WIDTH = 60
CHARACTER_HEIGHT = 60
CHARACTER_SPAWN_INTERVAL = 2.0  # seconds
CHARACTER_SPAWN_X = -60

# UI settings
BUTTON_WIDTH = 200
BUTTON_HEIGHT = 50
BUTTON_SPACING = 20

# Safe area (projector edge cutoff compensation)
# This is now calculated dynamically based on screen size and percentage
# Use get_safe_area_margin() from game.safe_area module
try:
    from game.safe_area import get_safe_area_margin
    # For backward compatibility, provide a function that calculates margin
    def SAFE_AREA_MARGIN(screen_width=None, screen_height=None):
        return get_safe_area_margin(screen_width, screen_height)
except ImportError:
    def SAFE_AREA_MARGIN(screen_width=None, screen_height=None):
        return 10  # fallback


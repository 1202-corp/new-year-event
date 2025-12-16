"""Safe area calculation module"""
from game.config import Config


def get_safe_area_margin(screen_width: int = None, screen_height: int = None) -> int:
    """
    Calculate safe area margin in pixels based on percentage.
    Uses average of width and height percentages for uniform margins.
    """
    if screen_width is None or screen_height is None:
        from game.config import Config
        screen_width = Config.SCREEN_WIDTH
        screen_height = Config.SCREEN_HEIGHT
    
    # Calculate margin as percentage of average dimension
    avg_dimension = (screen_width + screen_height) / 2
    margin_pixels = int(avg_dimension * Config.SAFE_AREA_MARGIN_PERCENT / 100)
    
    return margin_pixels


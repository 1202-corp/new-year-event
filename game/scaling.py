"""Scaling module for responsive UI and game elements"""
from game.config import Config


class ScalingManager:
    """Manages scaling based on screen size"""
    
    def __init__(self, base_width: int, base_height: int):
        self.base_width = base_width
        self.base_height = base_height
        self.current_width = base_width
        self.current_height = base_height
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.scale = 1.0  # Uniform scale (min of scale_x and scale_y)
    
    def update(self, current_width: int, current_height: int) -> None:
        """Updates scaling based on current screen size"""
        self.current_width = current_width
        self.current_height = current_height
        self.scale_x = current_width / self.base_width
        self.scale_y = current_height / self.base_height
        # Use uniform scaling to maintain aspect ratio
        self.scale = min(self.scale_x, self.scale_y)
    
    def scale_value(self, value: float) -> float:
        """Scales a value uniformly"""
        return value * self.scale
    
    def scale_width(self, value: float) -> float:
        """Scales a width value"""
        return value * self.scale_x
    
    def scale_height(self, value: float) -> float:
        """Scales a height value"""
        return value * self.scale_y
    
    def scale_font_size(self, base_size: int) -> int:
        """Scales font size"""
        return max(12, int(base_size * self.scale))


# Global scaling manager instance
_scaling_manager: ScalingManager = None


def init_scaling(base_width: int = None, base_height: int = None) -> None:
    """Initialize scaling manager"""
    global _scaling_manager
    if base_width is None:
        base_width = Config.SCREEN_WIDTH
    if base_height is None:
        base_height = Config.SCREEN_HEIGHT
    _scaling_manager = ScalingManager(base_width, base_height)


def get_scaling() -> ScalingManager:
    """Get the global scaling manager"""
    global _scaling_manager
    if _scaling_manager is None:
        init_scaling()
    return _scaling_manager


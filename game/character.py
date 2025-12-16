"""Character module"""
import pygame
from typing import Tuple, Optional
from game.enums import CharacterType
from game.constants import (
    WHITE, GREEN, RED, YELLOW, ORANGE, BLACK,
    CHARACTER_WIDTH, CHARACTER_HEIGHT
)
from game.scaling import get_scaling


class Character:
    """Character class representing an enemy"""
    
    def __init__(self, char_type: CharacterType, x: float, y: float, speed: float, screen_height: Optional[int] = None):
        self.type = char_type
        self.x = x
        self.y = y
        self.speed = speed
        self.screen_height = screen_height
        
        # Scale character size
        scaling = get_scaling()
        self.width = int(scaling.scale_value(CHARACTER_WIDTH))
        self.height = int(scaling.scale_value(CHARACTER_HEIGHT))
        
        self.is_alive = True
        self.points = self._get_points()
        self.color = self._get_color()
        
    def _get_points(self) -> int:
        """Returns points for killing this character"""
        points_map = {
            CharacterType.SNOWMAN: 10,
            CharacterType.GRINCH: 25,
            CharacterType.SANTA: 50,
            CharacterType.ELF: 15,
            CharacterType.RUDOLPH: 100
        }
        return points_map.get(self.type, 10)
    
    def _get_color(self) -> Tuple[int, int, int]:
        """Returns character color"""
        color_map = {
            CharacterType.SNOWMAN: WHITE,
            CharacterType.GRINCH: GREEN,
            CharacterType.SANTA: RED,
            CharacterType.ELF: YELLOW,
            CharacterType.RUDOLPH: ORANGE
        }
        return color_map.get(self.type, WHITE)
    
    def update(self, dt: float, screen_width: int = None) -> None:
        """Updates character position, respecting safe area"""
        if not self.is_alive:
            return
        
        from game.constants import SAFE_AREA_MARGIN
        
        # Move character
        new_x = self.x + self.speed * dt
        
        # Check if character would go outside safe area on the right
        if screen_width is not None:
            safe_right = screen_width - SAFE_AREA_MARGIN
            # Allow character to move past safe area (they'll be cleaned up)
            # But don't let them spawn or stay in the safe area border
            if new_x > safe_right:
                # Character is past safe area, will be cleaned up
                pass
        
        self.x = new_x
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draws the character, ensuring it doesn't draw in safe area borders"""
        if not self.is_alive:
            return
        
        from game.constants import SAFE_AREA_MARGIN
        
        screen_width = screen.get_width()
        screen_height = screen.get_height()
        
        # Check if character is completely outside safe area (shouldn't happen, but safety check)
        safe_left = SAFE_AREA_MARGIN
        safe_right = screen_width - SAFE_AREA_MARGIN
        safe_top = SAFE_AREA_MARGIN
        safe_bottom = screen_height - SAFE_AREA_MARGIN
        
        # Don't draw if character is in safe area borders
        if (self.x < safe_left or 
            self.x + self.width > safe_right or
            self.y < safe_top or
            self.y + self.height > safe_bottom):
            # Character is in safe area, don't draw
            return
        
        # Simple character representation (sprites will be added later)
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))
        
        # Add text label
        scaling = get_scaling()
        font_size = scaling.scale_font_size(24)
        font = pygame.font.Font(None, font_size)
        label = self._get_label()
        text = font.render(label, True, BLACK)
        text_rect = text.get_rect(center=(self.x + self.width // 2, self.y + self.height // 2))
        screen.blit(text, text_rect)
    
    def _get_label(self) -> str:
        """Returns character label"""
        label_map = {
            CharacterType.SNOWMAN: "⛄",
            CharacterType.GRINCH: "👹",
            CharacterType.SANTA: "🎅",
            CharacterType.ELF: "🧝",
            CharacterType.RUDOLPH: "🦌"
        }
        return label_map.get(self.type, "?")
    
    def get_rect(self) -> pygame.Rect:
        """Returns rectangle for collision detection"""
        return pygame.Rect(self.x, self.y, self.width, self.height)
    
    def is_point_inside(self, point: Tuple[int, int]) -> bool:
        """Checks if point is inside character"""
        return self.get_rect().collidepoint(point)


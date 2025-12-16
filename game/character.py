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
    
    def update(self, dt: float) -> None:
        """Updates character position"""
        if self.is_alive:
            self.x += self.speed * dt
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draws the character"""
        if not self.is_alive:
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


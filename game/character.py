"""Character module"""
import pygame
import random
from typing import Tuple, Optional
from game.enums import CharacterType, MovementType
from game.constants import (
    WHITE, GREEN, RED, YELLOW, ORANGE, BLACK,
    CHARACTER_WIDTH, CHARACTER_HEIGHT
)
from game.scaling import get_scaling


class Character:
    """Character class representing an enemy"""
    
    def __init__(
        self, 
        char_type: CharacterType, 
        x: float, 
        y: float, 
        speed: float, 
        movement_type: MovementType,
        lane: int,
        screen_height: Optional[int] = None
    ):
        self.type = char_type
        self.x = x
        self.y = y
        self.speed = speed
        self.movement_type = movement_type
        self.lane = lane
        self.screen_height = screen_height
        
        # Direction: 1 = right, -1 = left
        self.direction = 1 if movement_type == MovementType.FLYING else random.choice([-1, 1])
        
        # For patrolling characters
        self.patrol_change_timer = 0.0
        self.patrol_change_interval = random.uniform(2.0, 5.0)  # Change direction every 2-5 seconds
        self.current_speed = speed  # Current speed (can vary)
        self.speed_variation = 0.3  # Speed can vary by ±30%
        
        # Height offset for visual depth (so shadows overlap)
        # Each enemy on the same lane has slightly different Y offset
        self.height_offset = random.randint(-8, 8)  # Small random offset for depth
        
        # Scale character size
        scaling = get_scaling()
        self.width = int(scaling.scale_value(CHARACTER_WIDTH))
        self.height = int(scaling.scale_value(CHARACTER_HEIGHT))
        
        self.is_alive = True
        self.points = self._get_points()
        self.color = self._get_color()
        
        # Store base values for scaling
        self.base_width = CHARACTER_WIDTH
        self.base_height = CHARACTER_HEIGHT
        self.base_speed = speed
    
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
    
    def update(self, dt: float, screen_width: int = None, screen_height: int = None) -> None:
        """Updates character position based on movement type"""
        if not self.is_alive:
            return
        
        from game.safe_area import get_safe_area_margin
        
        if screen_width is None or screen_height is None:
            return
        
        margin = get_safe_area_margin(screen_width, screen_height)
        safe_left = margin
        safe_right = screen_width - margin
        
        if self.movement_type == MovementType.FLYING:
            # Flying: move in one direction until off screen
            self.x += self.speed * self.direction * dt
        else:
            # Patrolling: move left-right, change direction at edges or randomly
            self.patrol_change_timer += dt
            
            # Check if at edge
            direction_changed = False
            if self.x <= safe_left:
                self.direction = 1
                self.patrol_change_timer = 0.0
                direction_changed = True
            elif self.x + self.width >= safe_right:
                self.direction = -1
                self.patrol_change_timer = 0.0
                direction_changed = True
            # Random direction change
            elif self.patrol_change_timer >= self.patrol_change_interval:
                self.direction = random.choice([-1, 1])
                self.patrol_change_timer = 0.0
                self.patrol_change_interval = random.uniform(2.0, 5.0)
                direction_changed = True
            
            # Change speed when direction changes
            if direction_changed:
                # Random speed variation: base_speed * (1 ± speed_variation)
                speed_multiplier = random.uniform(1.0 - self.speed_variation, 1.0 + self.speed_variation)
                self.current_speed = self.base_speed * speed_multiplier
            
            # Move in current direction with current speed
            self.x += self.current_speed * self.direction * dt
            
            # Keep within bounds
            self.x = max(safe_left, min(self.x, safe_right - self.width))
    
    def update_scaling(self) -> None:
        """Updates character size based on current scaling"""
        scaling = get_scaling()
        self.width = int(scaling.scale_value(self.base_width))
        self.height = int(scaling.scale_value(self.base_height))
        # Speed should NOT scale - keep original speed
        self.speed = self.base_speed
    
    def draw(self, screen: pygame.Surface, ui_panel_height: int = 0) -> None:
        """Draws the character, ensuring it doesn't draw in safe area borders or UI panel"""
        if not self.is_alive:
            return
        
        from game.safe_area import get_safe_area_margin
        
        screen_width = screen.get_width()
        screen_height = screen.get_height()
        margin = get_safe_area_margin(screen_width, screen_height, ui_panel_height)
        
        # Check if character is completely outside safe area (shouldn't happen, but safety check)
        safe_left = margin
        safe_right = screen_width - margin
        safe_top = margin
        safe_bottom = screen_height - ui_panel_height - margin  # Don't draw in UI panel area
        
        # For flying characters, allow drawing even if partially off screen
        # For patrolling characters, don't draw if in safe area borders
        if self.movement_type == MovementType.PATROLLING:
            if (self.x < safe_left or 
                self.x + self.width > safe_right or
                self.y < safe_top or
                self.y + self.height > safe_bottom):
                # Character is in safe area, don't draw
                return
        else:
            # Flying: only check vertical bounds (top/bottom), allow horizontal off-screen
            if (self.y < safe_top or
                self.y + self.height > safe_bottom):
                # Character is outside vertical safe area, don't draw
                return
        
        # Draw shadow first (below character)
        # Shadow is a scaled-up version of the character shape (square/rectangle)
        shadow_scale = 1.15  # Shadow is 15% larger
        shadow_width = int(self.width * shadow_scale)
        shadow_height = int(self.height * shadow_scale)
        
        # Shadow position: centered on lane line, slightly offset down
        shadow_offset_y = 3  # Small vertical offset
        shadow_x = self.x - (shadow_width - self.width) // 2
        shadow_y = self.y + self.height // 2 - shadow_height // 2 + shadow_offset_y
        
        # Create semi-transparent shadow surface
        shadow_surface = pygame.Surface((shadow_width, shadow_height), pygame.SRCALPHA)
        shadow_color = (10, 10, 20, 150)  # Dark color with alpha (semi-transparent)
        
        # Draw shadow as rectangle (same shape as character, but larger)
        pygame.draw.rect(shadow_surface, shadow_color, (0, 0, shadow_width, shadow_height))
        
        # Blit shadow to screen (will be semi-transparent and can overlap enemies)
        screen.blit(shadow_surface, (shadow_x, shadow_y))
        
        # Draw character with height offset for depth
        character_y = self.y + self.height_offset
        pygame.draw.rect(screen, self.color, (self.x, character_y, self.width, self.height))
        
        # Add text label
        scaling = get_scaling()
        font_size = scaling.scale_font_size(24)
        font = pygame.font.Font(None, font_size)
        label = self._get_label()
        text = font.render(label, True, BLACK)
        text_rect = text.get_rect(center=(self.x + self.width // 2, character_y + self.height // 2))
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

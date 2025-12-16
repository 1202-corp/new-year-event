"""Character spawner module"""
import random
from typing import List, Optional
from game.character import Character
from game.enums import CharacterType
from game.config import Config
from game.constants import CHARACTER_SPAWN_X
from game.scaling import get_scaling
from game.safe_area import get_safe_area_margin


class CharacterSpawner:
    """Responsible for spawning characters"""
    
    def __init__(self):
        self.char_types = [
            CharacterType.SNOWMAN,
            CharacterType.SNOWMAN,  # Snowmen appear more often
            CharacterType.GRINCH,
            CharacterType.ELF,
            CharacterType.SANTA,
            CharacterType.RUDOLPH  # Rudolph is rare
        ]
        
        self.speed_map = {
            CharacterType.SNOWMAN: 50,
            CharacterType.GRINCH: 75,
            CharacterType.SANTA: 100,
            CharacterType.ELF: 90,
            CharacterType.RUDOLPH: 40
        }
    
    def spawn_character(
        self, 
        min_y: Optional[int] = None, 
        max_y: Optional[int] = None, 
        screen_width: Optional[int] = None,
        screen_height: Optional[int] = None
    ) -> Character:
        """Spawns a new character at random position, respecting safe area"""
        if screen_height is None:
            screen_height = Config.SCREEN_HEIGHT
        if screen_width is None:
            screen_width = Config.SCREEN_WIDTH
        
        # Calculate safe area bounds (using percentage-based margin)
        margin = get_safe_area_margin(screen_width, screen_height)
        safe_top = margin
        safe_bottom = screen_height - margin
        safe_left = margin
        safe_right = screen_width - margin
        
        # Spawn within safe area
        if min_y is None:
            min_y = safe_top + 20  # Small margin from top safe area
        if max_y is None:
            max_y = safe_bottom - 20  # Small margin from bottom safe area
        
        # Ensure we're within safe area
        min_y = max(min_y, safe_top)
        max_y = min(max_y, safe_bottom - 60)  # Account for character height
        
        char_type = random.choice(self.char_types)
        y = random.randint(min_y, max_y)
        base_speed = self.speed_map.get(char_type, 50)
        
        # Spawn from left safe area edge
        spawn_x = safe_left + CHARACTER_SPAWN_X if CHARACTER_SPAWN_X < 0 else safe_left
        
        # Character will scale speed internally based on current scaling
        return Character(char_type, spawn_x, y, base_speed, screen_height=screen_height)
    
    def cleanup_characters(self, characters: List[Character], screen_width: int) -> List[Character]:
        """Removes characters that are off screen"""
        return [c for c in characters if c.x < screen_width + 100]


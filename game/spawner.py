"""Character spawner module"""
import random
from typing import List
from game.character import Character
from game.enums import CharacterType
from game.config import Config
from game.constants import CHARACTER_SPAWN_X
from game.scaling import get_scaling


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
    
    def spawn_character(self, min_y: int = None, max_y: int = None, screen_height: int = None) -> Character:
        """Spawns a new character at random position"""
        if screen_height is None:
            screen_height = Config.SCREEN_HEIGHT
        
        if min_y is None:
            min_y = int(screen_height * 0.1)  # 10% from top
        if max_y is None:
            max_y = int(screen_height * 0.75)  # 75% from top (leaving space at bottom)
        
        char_type = random.choice(self.char_types)
        y = random.randint(min_y, max_y)
        base_speed = self.speed_map.get(char_type, 50)
        
        # Scale speed based on screen width
        scaling = get_scaling()
        speed = base_speed * scaling.scale_x
        
        return Character(char_type, CHARACTER_SPAWN_X, y, speed, screen_height=screen_height)
    
    def cleanup_characters(self, characters: List[Character], screen_width: int) -> List[Character]:
        """Removes characters that are off screen"""
        return [c for c in characters if c.x < screen_width + 100]


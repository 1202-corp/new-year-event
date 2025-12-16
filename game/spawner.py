"""Character spawner module"""
import random
from typing import List
from game.character import Character
from game.enums import CharacterType
from game.config import Config
from game.constants import CHARACTER_SPAWN_X


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
    
    def spawn_character(self, min_y: int = 100, max_y: int = None) -> Character:
        """Spawns a new character at random position"""
        if max_y is None:
            max_y = Config.SCREEN_HEIGHT - 200
        
        char_type = random.choice(self.char_types)
        y = random.randint(min_y, max_y)
        speed = self.speed_map.get(char_type, 50)
        
        return Character(char_type, CHARACTER_SPAWN_X, y, speed)
    
    def cleanup_characters(self, characters: List[Character], screen_width: int) -> List[Character]:
        """Removes characters that are off screen"""
        return [c for c in characters if c.x < screen_width + 100]


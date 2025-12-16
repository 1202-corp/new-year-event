"""Character spawner module"""
import random
from typing import List, Optional
from game.character import Character
from game.enums import CharacterType, MovementType
from game.config import Config
from game.constants import CHARACTER_SPAWN_X
from game.scaling import get_scaling
from game.safe_area import get_safe_area_margin


class CharacterSpawner:
    """Responsible for spawning characters on lanes"""
    
    def __init__(self):
        # Flying characters (fast, one direction)
        self.flying_types = [
            CharacterType.SNOWMAN,
            CharacterType.SNOWMAN,  # More common
            CharacterType.ELF,
            CharacterType.ELF
        ]
        
        # Patrolling characters (always on screen, patrol left-right)
        # Santa, Grinch, Rudolph can ONLY be patrolling
        self.patrolling_types = [
            CharacterType.SNOWMAN,
            CharacterType.GRINCH,
            CharacterType.SANTA,
            CharacterType.RUDOLPH,
            CharacterType.ELF
        ]
        
        self.speed_map = {
            CharacterType.SNOWMAN: 80,  # Faster for flying
            CharacterType.GRINCH: 50,   # Slower for patrolling
            CharacterType.SANTA: 60,    # Medium for patrolling
            CharacterType.ELF: 100,     # Fast for flying
            CharacterType.RUDOLPH: 40   # Slow for patrolling
        }
        
        self.flying_speed_map = {
            CharacterType.SNOWMAN: 150,  # Fast flying
            CharacterType.ELF: 200        # Very fast flying
        }
    
    def calculate_lane_y(self, lane: int, screen_height: int, ui_panel_height: int, margin: int) -> int:
        """Calculate Y position for a specific lane"""
        # Available height = screen height - UI panel - margins
        available_height = screen_height - ui_panel_height - margin * 2
        lane_spacing = available_height / (Config.NUM_LANES + 1)
        return margin + int(lane_spacing * (lane + 1))
    
    def spawn_character(
        self, 
        screen_width: Optional[int] = None,
        screen_height: Optional[int] = None,
        ui_panel_height: int = 0,
        existing_patrolling: int = 0,
        max_enemies: int = 20,
        enemies_per_lane: Optional[dict] = None
    ) -> Optional[Character]:
        """Spawns a new character on a random lane (respecting MAX_ENEMIES per lane)"""
        if screen_height is None:
            screen_height = Config.SCREEN_HEIGHT
        if screen_width is None:
            screen_width = Config.SCREEN_WIDTH
        if enemies_per_lane is None:
            enemies_per_lane = {}
        
        # Calculate safe area bounds (accounting for UI panel)
        margin = get_safe_area_margin(screen_width, screen_height, ui_panel_height)
        safe_left = margin
        safe_right = screen_width - margin
        
        # Determine movement type based on ratio (2/3 patrolling, 1/3 flying)
        # But special characters (Santa, Grinch, Rudolph) must be patrolling
        total_enemies = sum(enemies_per_lane.values())
        patrolling_ratio = existing_patrolling / total_enemies if total_enemies > 0 else 0
        
        # Decide movement type
        if patrolling_ratio < 0.67:  # Less than 2/3 are patrolling
            # Can spawn either type, but special chars must be patrolling
            char_type = random.choice(self.patrolling_types + self.flying_types)
            
            # Special characters must be patrolling
            if char_type in [CharacterType.SANTA, CharacterType.GRINCH, CharacterType.RUDOLPH]:
                movement_type = MovementType.PATROLLING
            else:
                # Random choice for others, but favor patrolling to reach 2/3
                movement_type = random.choices(
                    [MovementType.PATROLLING, MovementType.FLYING],
                    weights=[2, 1]  # 2/3 chance patrolling
                )[0]
        else:
            # Already have enough patrolling, spawn flying
            char_type = random.choice(self.flying_types)
            movement_type = MovementType.FLYING
        
        # Select random lane that has space (MAX_ENEMIES per lane)
        available_lanes = [lane for lane in range(Config.NUM_LANES) 
                          if enemies_per_lane.get(lane, 0) < max_enemies]
        if not available_lanes:
            return None  # All lanes are full
        
        lane = random.choice(available_lanes)
        y = self.calculate_lane_y(lane, screen_height, ui_panel_height, margin)
        
        # Set speed based on movement type
        if movement_type == MovementType.FLYING:
            speed = self.flying_speed_map.get(char_type, 150)
        else:
            speed = self.speed_map.get(char_type, 50)
        
        # Set spawn position and direction
        if movement_type == MovementType.FLYING:
            # Flying: spawn from left or right randomly, OFF SCREEN
            direction = random.choice([-1, 1])
            if direction == 1:  # Moving right, spawn from left (off screen)
                spawn_x = -100  # Spawn completely off screen on the left
            else:  # Moving left, spawn from right (off screen)
                spawn_x = screen_width + 100  # Spawn completely off screen on the right
        else:
            # Patrolling: spawn somewhere in the middle
            spawn_x = random.randint(safe_left + 50, safe_right - 50)
            direction = random.choice([-1, 1])
        
        character = Character(
            char_type, 
            spawn_x, 
            y, 
            speed, 
            movement_type,
            lane,
            screen_height=screen_height
        )
        character.direction = direction
        
        return character
    
    def cleanup_characters(self, characters: List[Character], screen_width: int) -> List[Character]:
        """Removes characters that are off screen (only for flying type)"""
        cleaned = []
        for c in characters:
            # Only remove flying characters that are completely off screen
            # Patrolling characters should always stay on screen
            if c.movement_type == MovementType.FLYING:
                # Remove if completely off screen (accounting for character width)
                if c.x + c.width < -50 or c.x > screen_width + 50:
                    continue  # Remove this character
            cleaned.append(c)
        return cleaned

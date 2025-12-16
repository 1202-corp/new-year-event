"""Game enumerations"""
from enum import Enum


class CharacterType(Enum):
    """Types of New Year characters"""
    SNOWMAN = "snowman"
    GRINCH = "grinch"
    SANTA = "santa"
    ELF = "elf"
    RUDOLPH = "rudolph"


class GameState(Enum):
    """Game states"""
    MENU = "menu"
    PLAYING = "playing"
    PAUSED = "paused"
    GAME_OVER = "game_over"


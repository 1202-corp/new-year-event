"""Game enumerations"""
from enum import Enum


class CharacterType(Enum):
    """Types of New Year characters"""
    SNOWMAN = "snowman"
    GRINCH = "grinch"
    SANTA = "santa"
    ELF = "elf"
    RUDOLPH = "rudolph"


class MovementType(Enum):
    """Character movement types"""
    FLYING = "flying"  # Fast flying left-right, one direction
    PATROLLING = "patrolling"  # Always on screen, patrols left-right


class GameState(Enum):
    """Game states"""
    MENU = "menu"
    CALIBRATING = "calibrating"  # Calibration phase
    CALIBRATION_PREVIEW = "calibration_preview"  # Showing corrected preview
    PLAYING = "playing"
    PAUSED = "paused"
    GAME_OVER = "game_over"


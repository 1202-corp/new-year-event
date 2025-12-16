"""Main game class"""
import pygame
import sys
import os
from typing import List, Tuple
from game.config import Config
from game.constants import DARK_BLUE, WHITE
from game.enums import GameState
from game.character import Character
from game.spawner import CharacterSpawner
from game.score import ScoreManager
from game.ui.menu import PauseMenu
from game.scaling import init_scaling, get_scaling
from game.logger import get_logger

logger = get_logger()


class Game:
    """Main game class"""
    
    def __init__(self):
        # Initialize pygame (but don't create window yet)
        pygame.init()
        
        # Check available displays and get info
        display_to_use = 0
        display_x_offset = 0
        
        try:
            num_displays = pygame.display.get_num_displays()
            logger.info(f"Number of displays detected: {num_displays}")
            
            if Config.DISPLAY_NUMBER >= num_displays:
                logger.warning(f"DISPLAY_NUMBER={Config.DISPLAY_NUMBER} >= available displays ({num_displays})")
                logger.info("Using display 0 instead")
                display_to_use = 0
            else:
                display_to_use = Config.DISPLAY_NUMBER
                logger.info(f"Using display {display_to_use}")
            
            # Get display info and calculate position offset
            if num_displays > 0:
                try:
                    desktop_sizes = pygame.display.get_desktop_sizes()
                    logger.info(f"Desktop sizes: {desktop_sizes}")
                    
                    if display_to_use < len(desktop_sizes):
                        logger.info(f"Display {display_to_use} size: {desktop_sizes[display_to_use]}")
                        
                        # Calculate X offset for positioning window on correct display
                        # Sum up widths of all displays before the target one
                        for i in range(display_to_use):
                            if i < len(desktop_sizes):
                                display_x_offset += desktop_sizes[i][0]
                        
                        logger.info(f"Calculated X offset for display {display_to_use}: {display_x_offset}")
                except Exception as e:
                    logger.error(f"Could not get desktop sizes: {e}")
        except Exception as e:
            logger.error(f"Could not query displays: {e}")
        
        # Set window position BEFORE creating window (SDL approach)
        # SDL_VIDEO_WINDOW_POS must be set before set_mode()
        if display_to_use > 0 and display_x_offset > 0:
            # SDL_VIDEO_WINDOW_POS format: "x,y" or "x" for x only
            window_pos = f"{display_x_offset},0"
            os.environ["SDL_VIDEO_WINDOW_POS"] = window_pos
            logger.info(f"Setting SDL_VIDEO_WINDOW_POS to: {window_pos} (before window creation)")
        elif display_to_use == 0:
            # Clear SDL_VIDEO_WINDOW_POS for display 0 to use default positioning
            if "SDL_VIDEO_WINDOW_POS" in os.environ:
                del os.environ["SDL_VIDEO_WINDOW_POS"]
        
        # Create window (SDL_VIDEO_WINDOW_POS will be used here)
        self.screen = pygame.display.set_mode((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
        pygame.display.set_caption("New Year Arcade Game")
        
        logger.info("Window created")
        
        self.clock = pygame.time.Clock()
        
        # Game state
        self.state = GameState.PLAYING
        self.running = True
        
        # Game components (MUST be initialized before _update_scaling)
        self.characters: List[Character] = []
        self.spawner = CharacterSpawner()
        self.score_manager = ScoreManager()
        
        # Initialize scaling
        init_scaling(Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT)
        
        # Fullscreen state (from config) - use borderless windowed fullscreen
        self.fullscreen = Config.FULLSCREEN
        if self.fullscreen:
            # Switch to borderless fullscreen (windowed fullscreen)
            # Get screen dimensions
            screen_info = pygame.display.Info()
            screen_width = screen_info.current_w
            screen_height = screen_info.current_h
            
            logger.info(f"Setting borderless fullscreen: {screen_width}x{screen_height}")
            
            # Use NOFRAME to remove window borders and set size to full screen
            try:
                self.screen = pygame.display.set_mode(
                    (screen_width, screen_height),
                    pygame.NOFRAME
                )
                logger.info("Borderless fullscreen activated with NOFRAME")
            except pygame.error as e:
                # Fallback: try with RESIZABLE flag
                logger.warning(f"NOFRAME failed: {e}, trying with RESIZABLE")
                try:
                    self.screen = pygame.display.set_mode(
                        (screen_width, screen_height),
                        pygame.RESIZABLE | pygame.NOFRAME
                    )
                    logger.info("Borderless fullscreen activated with RESIZABLE|NOFRAME")
                except pygame.error as e2:
                    # Last fallback: regular fullscreen
                    logger.warning(f"Borderless fullscreen not supported: {e2}, using regular fullscreen")
                    self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        
        # Update scaling after window is set up
        self._update_scaling()
        
        # Spawn timing
        self.spawn_timer = 0.0
        self.spawn_interval = Config.CHARACTER_SPAWN_INTERVAL
        
        # UI fonts (will be updated based on scaling)
        self._update_fonts()
        
        # Pause menu
        self.pause_menu = PauseMenu(
            on_continue=self.resume_game,
            on_restart=self.restart_game,
            on_quit=self.quit_game
        )
    
    def _update_scaling(self) -> None:
        """Updates scaling based on current screen size and updates all existing objects"""
        scaling = get_scaling()
        old_width = scaling.current_width if scaling.current_width > 0 else 0
        old_height = scaling.current_height if scaling.current_height > 0 else 0
        
        new_width = self.screen.get_width()
        new_height = self.screen.get_height()
        
        # Update scaling
        scaling.update(new_width, new_height)
        self._update_fonts()
        
        # Update all existing characters to match new scale
        # Only scale if we had a previous size (not first initialization)
        if old_width > 0 and old_height > 0 and (old_width != new_width or old_height != new_height):
            scale_x_ratio = new_width / old_width
            scale_y_ratio = new_height / old_height
            
            for character in self.characters:
                if character.is_alive:
                    # Update character size first
                    character.update_scaling()
                    
                    # Update character position (scale relative to new screen size)
                    character.x = character.x * scale_x_ratio
                    character.y = character.y * scale_y_ratio
                    
                    # Update character speed (scale with width)
                    character.speed = character.speed * scale_x_ratio
        else:
            # First initialization or no size change - just update scaling for new characters
            for character in self.characters:
                if character.is_alive:
                    character.update_scaling()
    
    def _update_fonts(self) -> None:
        """Updates font sizes based on current scaling"""
        scaling = get_scaling()
        self.font_large = pygame.font.Font(None, scaling.scale_font_size(48))
        self.font_small = pygame.font.Font(None, scaling.scale_font_size(24))
    
    def resume_game(self) -> None:
        """Resumes the game from pause"""
        self.state = GameState.PLAYING
    
    def restart_game(self) -> None:
        """Restarts the game"""
        self.characters.clear()
        self.score_manager.reset()
        self.spawn_timer = 0.0
        self.state = GameState.PLAYING
    
    def quit_game(self) -> None:
        """Quits the game"""
        self.running = False
    
    def toggle_fullscreen(self) -> None:
        """Toggles fullscreen mode (borderless windowed)"""
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            # Borderless windowed fullscreen
            screen_info = pygame.display.Info()
            screen_width = screen_info.current_w
            screen_height = screen_info.current_h
            
            logger.info(f"Toggling to borderless fullscreen: {screen_width}x{screen_height}")
            
            try:
                self.screen = pygame.display.set_mode(
                    (screen_width, screen_height),
                    pygame.NOFRAME
                )
            except pygame.error as e:
                logger.warning(f"NOFRAME failed: {e}, trying with RESIZABLE")
                try:
                    self.screen = pygame.display.set_mode(
                        (screen_width, screen_height),
                        pygame.RESIZABLE | pygame.NOFRAME
                    )
                except pygame.error as e2:
                    logger.warning(f"Borderless fullscreen not supported: {e2}, using regular fullscreen")
                    self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            logger.info("Toggling to windowed mode")
            self.screen = pygame.display.set_mode((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
        # Update scaling and all existing objects
        self._update_scaling()
    
    def handle_click(self, pos: Tuple[int, int]) -> None:
        """Handles mouse click"""
        if self.state != GameState.PLAYING:
            return
        
        for character in self.characters:
            if character.is_alive and character.is_point_inside(pos):
                character.is_alive = False
                points = character.points
                self.score_manager.add_points(points)
                logger.debug(f"Killed {character.type.value}! Points: +{points} (Total: {self.score_manager.get_score()})")
                break
    
    def handle_events(self) -> None:
        """Handles all pygame events"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if self.state == GameState.PLAYING:
                        self.state = GameState.PAUSED
                    elif self.state == GameState.PAUSED:
                        self.state = GameState.PLAYING
                elif event.key == pygame.K_F11:
                    self.toggle_fullscreen()
                elif event.key == pygame.K_RETURN:
                    # Check if ALT is pressed
                    mods = pygame.key.get_mods()
                    if mods & pygame.KMOD_ALT:
                        self.toggle_fullscreen()
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left mouse button
                    if self.state == GameState.PAUSED:
                        self.pause_menu.handle_event(event)
                    else:
                        self.handle_click(event.pos)
            
            elif self.state == GameState.PAUSED:
                self.pause_menu.handle_event(event)
    
    def update(self, dt: float) -> None:
        """Updates game state"""
        if self.state != GameState.PLAYING:
            return
        
        # Update characters
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()
        for character in self.characters:
            character.update(dt, screen_width=screen_width, screen_height=screen_height)
        
        # Cleanup off-screen characters (accounting for safe area)
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()
        from game.safe_area import get_safe_area_margin
        margin = get_safe_area_margin(screen_width, screen_height)
        # Remove characters that are past the right edge (including safe area)
        self.characters = self.spawner.cleanup_characters(
            self.characters, 
            screen_width - margin
        )
        
        # Spawn new characters
        self.spawn_timer += dt
        if self.spawn_timer >= self.spawn_interval:
            screen_width = self.screen.get_width()
            screen_height = self.screen.get_height()
            self.characters.append(self.spawner.spawn_character(
                screen_width=screen_width,
                screen_height=screen_height
            ))
            self.spawn_timer = 0.0
    
    def draw(self) -> None:
        """Draws the game"""
        # Background
        self.screen.fill(DARK_BLUE)
        
        # Draw safe area borders first (will be covered by characters if they overlap)
        self.draw_safe_area()
        
        # Draw characters
        for character in self.characters:
            character.draw(self.screen)
        
        # Draw UI (currently disabled)
        self.draw_ui()
        
        # Draw pause menu if paused
        if self.state == GameState.PAUSED:
            self.pause_menu.draw(self.screen)
        
        pygame.display.flip()
    
    def draw_ui(self) -> None:
        """Draws game UI (currently disabled - no on-screen text)"""
        # UI elements removed for projector setup
        pass
    
    def draw_safe_area(self) -> None:
        """Draws safe area borders (for projector edge cutoff)"""
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()
        from game.safe_area import get_safe_area_margin
        margin = get_safe_area_margin(screen_width, screen_height)
        
        # Draw safe area borders with background color
        # Top border
        pygame.draw.rect(self.screen, DARK_BLUE, (0, 0, screen_width, margin))
        # Bottom border
        pygame.draw.rect(self.screen, DARK_BLUE, (0, screen_height - margin, screen_width, margin))
        # Left border
        pygame.draw.rect(self.screen, DARK_BLUE, (0, 0, margin, screen_height))
        # Right border
        pygame.draw.rect(self.screen, DARK_BLUE, (screen_width - margin, 0, margin, screen_height))
    
    def run(self) -> None:
        """Main game loop"""
        last_time = pygame.time.get_ticks()
        
        while self.running:
            current_time = pygame.time.get_ticks()
            dt = (current_time - last_time) / 1000.0
            last_time = current_time
            
            self.handle_events()
            self.update(dt)
            self.draw()
            
            self.clock.tick(Config.FPS)
        
        pygame.quit()
        sys.exit()


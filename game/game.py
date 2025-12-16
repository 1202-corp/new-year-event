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
from game.ui_panel import UIPanel
from game.scaling import init_scaling, get_scaling
from game.logger import get_logger
from game.aruco_transform import ArucoTransform

logger = get_logger()


class Game:
    """Main game class"""
    
    def __init__(self):
        # Set SDL environment variables to prevent window minimization
        # This must be done before pygame.init()
        os.environ["SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS"] = "0"
        
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
        self.state = GameState.CALIBRATING if Config.CALIBRATION_ENABLED else GameState.PLAYING
        self.running = True
        
        # Game components (MUST be initialized before _update_scaling)
        self.characters: List[Character] = []
        self.spawner = CharacterSpawner()
        self.score_manager = ScoreManager()
        
        # Initialize scaling
        init_scaling(Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT)
        
        # Fullscreen state (from config)
        self.fullscreen = Config.FULLSCREEN
        if self.fullscreen:
            # Switch to regular fullscreen mode
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            logger.info("Fullscreen mode activated")
        
        # UI Panel (bottom RPG-style panel) - must be created before _update_scaling()
        self.ui_panel = UIPanel(
            self.screen.get_width(),
            self.screen.get_height()
        )
        
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
        
        # Aruco transform for perspective correction (optional, enabled via config)
        self.aruco_transform = None
        if Config.CALIBRATION_ENABLED:
            try:
                self.aruco_transform = ArucoTransform(
                    camera_index=Config.SNOWBALL_CAMERA_INDEX,
                    camera_width=Config.SNOWBALL_CAMERA_WIDTH,
                    camera_height=Config.SNOWBALL_CAMERA_HEIGHT
                )
                logger.info("Aruco transform enabled for perspective correction")
            except Exception as e:
                logger.warning(f"Failed to initialize Aruco transform: {e}")
                self.aruco_transform = None
        
        # Create calibration test characters (random positions for calibration screen)
        self.calibration_characters = []
        if self.state == GameState.CALIBRATING:
            self._create_calibration_characters()
    
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
        
        # Update UI panel with new screen dimensions (if it exists)
        if hasattr(self, 'ui_panel'):
            self.ui_panel.update(new_width, new_height)
        
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
                    
                    # Don't update speed - keep original speed
                    # Speed should remain constant regardless of screen size
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
    
    def _create_calibration_characters(self) -> None:
        """Create random characters for calibration screen"""
        import random
        from game.character import Character
        from game.enums import CharacterType, MovementType
        from game.safe_area import get_safe_area_margin
        
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()
        ui_panel_height = self.ui_panel.panel_height
        margin = get_safe_area_margin(screen_width, screen_height, ui_panel_height)
        
        # Create 10-15 random characters at random positions
        num_chars = random.randint(10, 15)
        self.calibration_characters = []
        
        for _ in range(num_chars):
            # Random position within safe area
            x = random.randint(margin, screen_width - margin - 60)
            y = random.randint(margin, screen_height - ui_panel_height - margin - 60)
            
            # Random character type
            char_type = random.choice(list(CharacterType))
            
            # Get speed for this character type
            speed = self.spawner.speed_map.get(char_type, 100)
            
            character = Character(
                char_type=char_type,
                x=x,
                y=y,
                speed=speed,
                movement_type=MovementType.PATROLLING,  # All stationary for calibration
                lane=0,  # Not used for calibration
                screen_height=screen_height
            )
            self.calibration_characters.append(character)
    
    def restart_game(self) -> None:
        """Restarts the game"""
        self.characters.clear()
        self.score_manager.reset()
        self.spawn_timer = 0.0
        self.state = GameState.PLAYING
    
    def quit_game(self) -> None:
        """Quits the game"""
        # Release Aruco transform resources
        if self.aruco_transform is not None:
            self.aruco_transform.release()
        self.running = False
    
    def toggle_fullscreen(self) -> None:
        """Toggles fullscreen mode"""
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            # Regular fullscreen mode
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
            logger.info("Switched to fullscreen mode")
        else:
            # Windowed mode
            self.screen = pygame.display.set_mode((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
            logger.info("Switched to windowed mode")
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
            
            elif event.type == pygame.WINDOWFOCUSLOST:
                # Prevent window from minimizing when focus is lost
                # Keep window visible even when other windows are focused
                logger.debug("Window lost focus, but keeping it visible")
            
            elif event.type == pygame.VIDEORESIZE:
                # Handle window resize
                # Note: This event is only generated if window is resizable
                # For fullscreen mode, size changes are handled in toggle_fullscreen()
                logger.debug(f"Window resized to {event.w} x {event.h}")
                self._update_scaling()
            
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
                elif event.key == pygame.K_SPACE:
                    # Calibration: space bar to calibrate and start game
                    if self.state == GameState.CALIBRATING:
                        if self.aruco_transform is not None:
                            screen_width = self.screen.get_width()
                            screen_height = self.screen.get_height()
                            if self.aruco_transform.calibrate(screen_width, screen_height):
                                self.state = GameState.PLAYING
                                logger.info("Aruco calibration successful. Starting game.")
                            else:
                                logger.warning("Calibration failed. Make sure all 4 Aruco markers are visible.")
            
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
        # During calibration, update Aruco detection for debug display
        if self.state == GameState.CALIBRATING:
            if self.aruco_transform is not None:
                # Continuously update to show debug window with camera feed
                screen_width = self.screen.get_width()
                screen_height = self.screen.get_height()
                key = self.aruco_transform.update(screen_width, screen_height)
                
                # Check if SPACE was pressed in debug window (key code 32)
                if key == 32:  # SPACE key
                    if self.aruco_transform.calibrate(screen_width, screen_height):
                        self.state = GameState.PLAYING
                        logger.info("Aruco calibration successful. Starting game.")
                    else:
                        logger.warning("Calibration failed. Make sure all 4 Aruco markers are visible.")
            return
        
        if self.state != GameState.PLAYING:
            return
        
        # Update characters (accounting for UI panel)
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()
        ui_panel_height = self.ui_panel.panel_height
        # Effective game area height (excluding UI panel)
        game_area_height = screen_height - ui_panel_height
        
        # Process camera frames every frame (not just on resize)
        self.ui_panel.process_camera_frame()
        
        for character in self.characters:
            character.update(dt, screen_width=screen_width, screen_height=game_area_height)
        
        # Cleanup off-screen characters (only flying type)
        self.characters = self.spawner.cleanup_characters(self.characters, screen_width)
        
        # Count existing enemies per lane (MAX_ENEMIES is per lane, not total)
        from game.enums import MovementType
        patrolling_count = sum(1 for c in self.characters if c.is_alive and c.movement_type == MovementType.PATROLLING)
        
        # Count enemies per lane
        enemies_per_lane = {}
        for c in self.characters:
            if c.is_alive:
                lane = c.lane
                enemies_per_lane[lane] = enemies_per_lane.get(lane, 0) + 1
        
        # Spawn new characters if under max limit per lane
        self.spawn_timer += dt
        if self.spawn_timer >= self.spawn_interval:
            # Check if we can spawn on any lane
            can_spawn = False
            for lane in range(Config.NUM_LINES):
                if enemies_per_lane.get(lane, 0) < Config.MAX_ENEMIES:
                    can_spawn = True
                    break
            
            if can_spawn:
                new_character = self.spawner.spawn_character(
                    screen_width=screen_width,
                    screen_height=screen_height,
                    ui_panel_height=ui_panel_height,
                    existing_patrolling=patrolling_count,
                    max_enemies=Config.MAX_ENEMIES,
                    enemies_per_lane=enemies_per_lane
                )
                if new_character:
                    self.characters.append(new_character)
            self.spawn_timer = 0.0
    
    def draw(self) -> None:
        """Draws the game"""
        # Background
        self.screen.fill(DARK_BLUE)
        
        # Draw calibration screen or game
        if self.state == GameState.CALIBRATING:
            self._draw_calibration_screen()
        else:
            self._draw_game()
        
        # No Aruco transformation applied to game screen - game displays normally
        # Aruco markers are only used for calibrating the snowball detection camera
        pygame.display.flip()
    
    def _draw_calibration_screen(self) -> None:
        """Draw calibration screen with frozen characters"""
        # Draw safe area borders
        self.draw_safe_area()
        
        # Draw lane lines
        self.draw_lane_lines()
        
        # Draw calibration characters (frozen)
        ui_panel_height = self.ui_panel.panel_height
        sorted_characters = sorted(self.calibration_characters, key=lambda c: (c.y + c.height_offset))
        for character in sorted_characters:
            character.draw(self.screen, ui_panel_height=ui_panel_height)
        
        # Draw UI panel
        self.ui_panel.draw(self.screen, 0, len(self.calibration_characters), len(self.calibration_characters))
    
    def _draw_game(self) -> None:
        """Draw normal game screen"""
        # Draw safe area borders first (will be covered by characters if they overlap)
        self.draw_safe_area()
        
        # Draw lane lines (visual guides)
        self.draw_lane_lines()
        
        # Draw characters (only in game area, above UI panel)
        # Sort by Y position to draw shadows correctly (back to front)
        ui_panel_height = self.ui_panel.panel_height
        sorted_characters = sorted(self.characters, key=lambda c: (c.y + c.height_offset) if c.is_alive else 0)
        for character in sorted_characters:
            character.draw(self.screen, ui_panel_height=ui_panel_height)
        
        # Draw UI panel (bottom)
        alive_count = sum(1 for c in self.characters if c.is_alive)
        # Maximum enemies = MAX_ENEMIES per lane * NUM_LINES
        max_total_enemies = Config.MAX_ENEMIES * Config.NUM_LINES
        self.ui_panel.draw(self.screen, self.score_manager.get_score(), alive_count, max_total_enemies)
        
        # Draw UI (currently disabled)
        self.draw_ui()
        
        # Draw pause menu if paused
        if self.state == GameState.PAUSED:
            self.pause_menu.draw(self.screen)
    
    def draw_ui(self) -> None:
        """Draws game UI (currently disabled - no on-screen text)"""
        # UI elements removed for projector setup
        pass
    
    def _apply_aruco_transform(self) -> None:
        """Apply Aruco perspective transformation to the game screen"""
        try:
            import numpy as np
            import cv2
            
            # Update transform based on current camera frame
            screen_width = self.screen.get_width()
            screen_height = self.screen.get_height()
            
            if self.aruco_transform.update(screen_width, screen_height):
                # Convert pygame surface to numpy array
                # Pygame surface is in RGB format
                screen_array = pygame.surfarray.array3d(self.screen)
                # Transpose from (width, height, channels) to (height, width, channels)
                screen_array = np.transpose(screen_array, (1, 0, 2))
                # Convert RGB to BGR for OpenCV
                screen_bgr = cv2.cvtColor(screen_array, cv2.COLOR_RGB2BGR)
                
                # Apply perspective transformation
                transformed = self.aruco_transform.apply_transform(screen_bgr)
                
                if transformed is not None and transformed.shape[0] > 0 and transformed.shape[1] > 0:
                    # Convert back to RGB
                    transformed_rgb = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB)
                    # Transpose back to (width, height, channels)
                    transformed_rgb = np.transpose(transformed_rgb, (1, 0, 2))
                    # Create new pygame surface from transformed array
                    transformed_surface = pygame.surfarray.make_surface(transformed_rgb)
                    
                    # Scale transformed surface to exactly fill the screen
                    # Stretch to fill entire screen (may distort, but fills completely)
                    transformed_surface = pygame.transform.scale(transformed_surface, (screen_width, screen_height))
                    
                    # Fill screen with black first (for empty areas, though surface should fill it)
                    self.screen.fill((0, 0, 0))
                    
                    # Blit transformed surface to screen (should fill entire screen now)
                    self.screen.blit(transformed_surface, (0, 0))
            
            pygame.display.flip()
        except ImportError:
            # OpenCV or numpy not available, skip transform
            pygame.display.flip()
        except Exception as e:
            logger.debug(f"Error applying Aruco transform: {e}")
            pygame.display.flip()
    
    def draw_safe_area(self) -> None:
        """Draws safe area borders (for projector edge cutoff)"""
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()
        ui_panel_height = self.ui_panel.panel_height
        from game.safe_area import get_safe_area_margin
        margin = get_safe_area_margin(screen_width, screen_height, ui_panel_height)
        
        # Draw safe area borders with background color (only in game area, not UI panel)
        game_area_height = screen_height - ui_panel_height
        # Top border
        pygame.draw.rect(self.screen, DARK_BLUE, (0, 0, screen_width, margin))
        # Bottom border (above UI panel)
        pygame.draw.rect(self.screen, DARK_BLUE, (0, game_area_height - margin, screen_width, margin))
        # Left border
        pygame.draw.rect(self.screen, DARK_BLUE, (0, 0, margin, game_area_height))
        # Right border
        pygame.draw.rect(self.screen, DARK_BLUE, (screen_width - margin, 0, margin, game_area_height))
    
    def draw_lane_lines(self) -> None:
        """Draws visual lane lines (dark blue) to show where enemies move"""
        screen_width = self.screen.get_width()
        screen_height = self.screen.get_height()
        ui_panel_height = self.ui_panel.panel_height
        from game.safe_area import get_safe_area_margin
        from game.config import Config
        
        margin = get_safe_area_margin(screen_width, screen_height, ui_panel_height)
        available_height = screen_height - ui_panel_height - margin * 2
        lane_spacing = available_height / (Config.NUM_LINES + 1)
        
        # Dark blue color for lane lines (darker)
        lane_color = (5, 10, 25)  # Very dark blue-gray
        
        # Draw lines for each lane
        for lane in range(Config.NUM_LANES):
            y = margin + int(lane_spacing * (lane + 1))
            # Draw horizontal line across the screen
            pygame.draw.line(
                self.screen,
                lane_color,
                (margin, y),
                (screen_width - margin, y),
                8  # Line width (thicker)
            )
    
    def run(self) -> None:
        """Main game loop"""
        last_time = pygame.time.get_ticks()
        last_width = self.screen.get_width()
        last_height = self.screen.get_height()
        
        while self.running:
            current_time = pygame.time.get_ticks()
            dt = (current_time - last_time) / 1000.0
            last_time = current_time
            
            self.handle_events()
            
            # Check if screen size changed (for cases where VIDEORESIZE event might not fire)
            current_width = self.screen.get_width()
            current_height = self.screen.get_height()
            if current_width != last_width or current_height != last_height:
                logger.debug(f"Screen size changed from {last_width}x{last_height} to {current_width}x{current_height}")
                self._update_scaling()
                last_width = current_width
                last_height = current_height
            
            self.update(dt)
            self.draw()
            
            self.clock.tick(Config.FPS)
        
        pygame.quit()
        sys.exit()


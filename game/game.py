"""Main game class"""
import pygame
import sys
from typing import List, Tuple
from game.config import Config
from game.constants import DARK_BLUE, WHITE
from game.enums import GameState
from game.character import Character
from game.spawner import CharacterSpawner
from game.score import ScoreManager
from game.ui.menu import PauseMenu
from game.scaling import init_scaling, get_scaling


class Game:
    """Main game class"""
    
    def __init__(self):
        # Initialize pygame
        pygame.init()
        
        # Check available displays and get info
        display_to_use = 0
        try:
            num_displays = pygame.display.get_num_displays()
            print(f"[Game] Number of displays detected: {num_displays}")
            
            if Config.DISPLAY_NUMBER >= num_displays:
                print(f"[Game] WARNING: DISPLAY_NUMBER={Config.DISPLAY_NUMBER} >= available displays ({num_displays})")
                print(f"[Game] Using display 0 instead")
            else:
                display_to_use = Config.DISPLAY_NUMBER
                print(f"[Game] Using display {display_to_use}")
            
            # Get display info
            if num_displays > 0:
                try:
                    desktop_sizes = pygame.display.get_desktop_sizes()
                    print(f"[Game] Desktop sizes: {desktop_sizes}")
                    if display_to_use < len(desktop_sizes):
                        print(f"[Game] Display {display_to_use} size: {desktop_sizes[display_to_use]}")
                except Exception as e:
                    print(f"[Game] Could not get desktop sizes: {e}")
        except Exception as e:
            print(f"[Game] Could not query displays: {e}")
        
        # Create window
        # Note: pygame.display.set_mode() doesn't support selecting display directly
        # The DISPLAY environment variable should handle this for X11
        self.screen = pygame.display.set_mode((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
        pygame.display.set_caption("New Year Arcade Game")
        
        # Get window position info
        window_info = pygame.display.get_wm_info()
        print(f"[Game] Window info: {window_info}")
        
        self.clock = pygame.time.Clock()
        
        # Initialize scaling
        init_scaling(Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT)
        self._update_scaling()
        
        # Fullscreen state
        self.fullscreen = False
        
        # Game state
        self.state = GameState.PLAYING
        self.running = True
        
        # Game components
        self.characters: List[Character] = []
        self.spawner = CharacterSpawner()
        self.score_manager = ScoreManager()
        
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
        """Updates scaling based on current screen size"""
        scaling = get_scaling()
        scaling.update(self.screen.get_width(), self.screen.get_height())
        self._update_fonts()
    
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
        """Toggles fullscreen mode"""
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
        else:
            self.screen = pygame.display.set_mode((Config.SCREEN_WIDTH, Config.SCREEN_HEIGHT))
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
                print(f"Killed {character.type.value}! Points: +{points} (Total: {self.score_manager.get_score()})")
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
        for character in self.characters:
            character.update(dt)
        
        # Cleanup off-screen characters
        screen_width = self.screen.get_width()
        self.characters = self.spawner.cleanup_characters(self.characters, screen_width)
        
        # Spawn new characters
        self.spawn_timer += dt
        if self.spawn_timer >= self.spawn_interval:
            screen_height = self.screen.get_height()
            self.characters.append(self.spawner.spawn_character(screen_height=screen_height))
            self.spawn_timer = 0.0
    
    def draw(self) -> None:
        """Draws the game"""
        # Background
        self.screen.fill(DARK_BLUE)
        
        # Draw characters
        for character in self.characters:
            character.draw(self.screen)
        
        # Draw UI
        self.draw_ui()
        
        # Draw pause menu if paused
        if self.state == GameState.PAUSED:
            self.pause_menu.draw(self.screen)
        
        pygame.display.flip()
    
    def draw_ui(self) -> None:
        """Draws game UI"""
        # Score
        score_text = self.font_large.render(f"Score: {self.score_manager.get_score()}", True, WHITE)
        self.screen.blit(score_text, (20, 20))
        
        # Instructions
        if self.state == GameState.PLAYING:
            screen_height = self.screen.get_height()
            instruction_text = self.font_small.render(
                "Click on characters to kill them | ESC to pause | F11/ALT+ENTER for fullscreen",
                True, WHITE
            )
            self.screen.blit(instruction_text, (20, screen_height - 40))
    
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


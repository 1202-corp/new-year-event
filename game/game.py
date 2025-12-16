"""Main game class"""
import pygame
import sys
from typing import List, Tuple
from game.constants import SCREEN_WIDTH, SCREEN_HEIGHT, FPS, DARK_BLUE, WHITE
from game.enums import GameState
from game.character import Character
from game.spawner import CharacterSpawner
from game.score import ScoreManager
from game.ui.menu import PauseMenu


class Game:
    """Main game class"""
    
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("New Year Arcade Game")
        self.clock = pygame.time.Clock()
        
        # Game state
        self.state = GameState.PLAYING
        self.running = True
        
        # Game components
        self.characters: List[Character] = []
        self.spawner = CharacterSpawner()
        self.score_manager = ScoreManager()
        
        # Spawn timing
        self.spawn_timer = 0.0
        self.spawn_interval = 2.0
        
        # UI
        self.font_large = pygame.font.Font(None, 48)
        self.font_small = pygame.font.Font(None, 24)
        
        # Pause menu
        self.pause_menu = PauseMenu(
            on_continue=self.resume_game,
            on_restart=self.restart_game,
            on_quit=self.quit_game
        )
    
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
        self.characters = self.spawner.cleanup_characters(self.characters, SCREEN_WIDTH)
        
        # Spawn new characters
        self.spawn_timer += dt
        if self.spawn_timer >= self.spawn_interval:
            self.characters.append(self.spawner.spawn_character())
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
            instruction_text = self.font_small.render("Click on characters to kill them | ESC to pause", True, WHITE)
            self.screen.blit(instruction_text, (20, SCREEN_HEIGHT - 40))
    
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
            
            self.clock.tick(FPS)
        
        pygame.quit()
        sys.exit()


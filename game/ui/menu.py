"""Menu UI components"""
import pygame
from typing import Callable, List
from game.constants import SCREEN_WIDTH, SCREEN_HEIGHT, WHITE, BLACK, BUTTON_WIDTH, BUTTON_HEIGHT, BUTTON_SPACING
from game.ui.button import Button


class PauseMenu:
    """Pause menu with Continue, Restart, Quit buttons"""
    
    def __init__(
        self,
        on_continue: Callable[[], None],
        on_restart: Callable[[], None],
        on_quit: Callable[[], None]
    ):
        self.on_continue = on_continue
        self.on_restart = on_restart
        self.on_quit = on_quit
        
        # Calculate button positions (centered)
        menu_width = BUTTON_WIDTH
        menu_height = (BUTTON_HEIGHT + BUTTON_SPACING) * 3 - BUTTON_SPACING
        start_x = (SCREEN_WIDTH - menu_width) // 2
        start_y = (SCREEN_HEIGHT - menu_height) // 2
        
        # Create buttons
        self.buttons: List[Button] = [
            Button(start_x, start_y, "Continue", on_continue),
            Button(start_x, start_y + BUTTON_HEIGHT + BUTTON_SPACING, "Restart", on_restart),
            Button(start_x, start_y + (BUTTON_HEIGHT + BUTTON_SPACING) * 2, "Quit", on_quit)
        ]
        
        self.font_title = pygame.font.Font(None, 72)
    
    def handle_event(self, event: pygame.event.Event) -> None:
        """Handles pygame events"""
        for button in self.buttons:
            button.handle_event(event)
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draws the pause menu"""
        # Draw semi-transparent overlay
        overlay = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        overlay.set_alpha(180)
        overlay.fill(BLACK)
        screen.blit(overlay, (0, 0))
        
        # Draw title
        title_text = self.font_title.render("PAUSED", True, WHITE)
        title_rect = title_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 150))
        screen.blit(title_text, title_rect)
        
        # Draw buttons
        for button in self.buttons:
            button.draw(screen)


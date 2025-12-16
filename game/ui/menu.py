"""Menu UI components"""
import pygame
from typing import Callable, List
from game.constants import WHITE, BLACK, BUTTON_WIDTH, BUTTON_HEIGHT, BUTTON_SPACING
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
        
        # Create buttons (positions will be calculated dynamically)
        self.buttons: List[Button] = [
            Button(0, 0, "Continue", on_continue),
            Button(0, 0, "Restart", on_restart),
            Button(0, 0, "Quit", on_quit)
        ]
        
        self.font_title = pygame.font.Font(None, 72)
    
    def _update_button_positions(self, screen: pygame.Surface) -> None:
        """Updates button positions based on screen size"""
        screen_width = screen.get_width()
        screen_height = screen.get_height()
        
        # Calculate button positions (centered)
        menu_width = BUTTON_WIDTH
        menu_height = (BUTTON_HEIGHT + BUTTON_SPACING) * 3 - BUTTON_SPACING
        start_x = (screen_width - menu_width) // 2
        start_y = (screen_height - menu_height) // 2
        
        # Update button positions
        self.buttons[0].x = start_x
        self.buttons[0].y = start_y
        self.buttons[1].x = start_x
        self.buttons[1].y = start_y + BUTTON_HEIGHT + BUTTON_SPACING
        self.buttons[2].x = start_x
        self.buttons[2].y = start_y + (BUTTON_HEIGHT + BUTTON_SPACING) * 2
    
    def handle_event(self, event: pygame.event.Event) -> None:
        """Handles pygame events"""
        for button in self.buttons:
            button.handle_event(event)
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draws the pause menu"""
        screen_width = screen.get_width()
        screen_height = screen.get_height()
        
        # Update button positions
        self._update_button_positions(screen)
        
        # Draw semi-transparent overlay
        overlay = pygame.Surface((screen_width, screen_height))
        overlay.set_alpha(180)
        overlay.fill(BLACK)
        screen.blit(overlay, (0, 0))
        
        # Draw title
        title_text = self.font_title.render("PAUSED", True, WHITE)
        title_rect = title_text.get_rect(center=(screen_width // 2, screen_height // 2 - 150))
        screen.blit(title_text, title_rect)
        
        # Draw buttons
        for button in self.buttons:
            button.draw(screen)


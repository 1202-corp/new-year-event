"""Menu UI components"""
import pygame
from typing import Callable, List
from game.constants import WHITE, BLACK, BUTTON_WIDTH, BUTTON_HEIGHT, BUTTON_SPACING
from game.ui.button import Button
from game.scaling import get_scaling


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
        
        self._update_fonts()
    
    def _update_fonts(self) -> None:
        """Updates font sizes based on scaling"""
        scaling = get_scaling()
        self.font_title = pygame.font.Font(None, scaling.scale_font_size(72))
    
    def _update_button_positions(self, screen: pygame.Surface) -> None:
        """Updates button positions and sizes based on screen size"""
        screen_width = screen.get_width()
        screen_height = screen.get_height()
        
        # Update scaling
        scaling = get_scaling()
        scaling.update(screen_width, screen_height)
        
        # Update button sizes
        for button in self.buttons:
            button.update_size()
        
        # Update fonts
        self._update_fonts()
        
        # Calculate button positions (centered)
        button_width = int(scaling.scale_value(BUTTON_WIDTH))
        button_height = int(scaling.scale_value(BUTTON_HEIGHT))
        button_spacing = int(scaling.scale_value(BUTTON_SPACING))
        
        menu_width = button_width
        menu_height = (button_height + button_spacing) * 3 - button_spacing
        start_x = (screen_width - menu_width) // 2
        start_y = (screen_height - menu_height) // 2
        
        # Update button positions
        self.buttons[0].x = start_x
        self.buttons[0].y = start_y
        self.buttons[1].x = start_x
        self.buttons[1].y = start_y + button_height + button_spacing
        self.buttons[2].x = start_x
        self.buttons[2].y = start_y + (button_height + button_spacing) * 2
    
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
        scaling = get_scaling()
        title_text = self.font_title.render("PAUSED", True, WHITE)
        title_offset = int(scaling.scale_value(150))
        title_rect = title_text.get_rect(center=(screen_width // 2, screen_height // 2 - title_offset))
        screen.blit(title_text, title_rect)
        
        # Draw buttons
        for button in self.buttons:
            button.draw(screen)


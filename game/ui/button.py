"""Button UI component"""
import pygame
from typing import Callable, Optional, Tuple
from game.constants import WHITE, BLACK, GRAY, LIGHT_GRAY, BUTTON_WIDTH, BUTTON_HEIGHT


class Button:
    """Button UI component"""
    
    def __init__(
        self,
        x: int,
        y: int,
        text: str,
        callback: Callable[[], None],
        width: int = BUTTON_WIDTH,
        height: int = BUTTON_HEIGHT
    ):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text = text
        self.callback = callback
        self.is_hovered = False
        self.font = pygame.font.Font(None, 36)
    
    def get_rect(self) -> pygame.Rect:
        """Returns button rectangle"""
        return pygame.Rect(self.x, self.y, self.width, self.height)
    
    def handle_event(self, event: pygame.event.Event) -> bool:
        """Handles pygame event, returns True if button was clicked"""
        if event.type == pygame.MOUSEMOTION:
            self.is_hovered = self.get_rect().collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1 and self.get_rect().collidepoint(event.pos):
                self.callback()
                return True
        return False
    
    def draw(self, screen: pygame.Surface) -> None:
        """Draws the button"""
        color = LIGHT_GRAY if self.is_hovered else GRAY
        pygame.draw.rect(screen, color, self.get_rect())
        pygame.draw.rect(screen, BLACK, self.get_rect(), 2)
        
        # Draw text
        text_surface = self.font.render(self.text, True, BLACK)
        text_rect = text_surface.get_rect(center=self.get_rect().center)
        screen.blit(text_surface, text_rect)


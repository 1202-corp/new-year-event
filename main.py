import pygame
import random
import sys
from enum import Enum
from typing import List, Tuple

# Инициализация Pygame
pygame.init()

# Константы
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 60

# Цвета
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
ORANGE = (255, 165, 0)
LIGHT_BLUE = (173, 216, 230)


class CharacterType(Enum):
    """Типы новогодних персонажей"""
    SNOWMAN = "snowman"
    GRINCH = "grinch"
    SANTA = "santa"
    ELF = "elf"
    RUDOLPH = "rudolph"


class Character:
    """Класс персонажа-врага"""
    
    def __init__(self, char_type: CharacterType, x: float, y: float, speed: float):
        self.type = char_type
        self.x = x
        self.y = y
        self.speed = speed
        self.width = 60
        self.height = 60
        self.is_alive = True
        self.points = self._get_points()
        self.color = self._get_color()
        
    def _get_points(self) -> int:
        """Возвращает очки за убийство персонажа"""
        points_map = {
            CharacterType.SNOWMAN: 10,
            CharacterType.GRINCH: 25,
            CharacterType.SANTA: 50,
            CharacterType.ELF: 15,
            CharacterType.RUDOLPH: 100
        }
        return points_map.get(self.type, 10)
    
    def _get_color(self) -> Tuple[int, int, int]:
        """Возвращает цвет персонажа"""
        color_map = {
            CharacterType.SNOWMAN: WHITE,
            CharacterType.GRINCH: GREEN,
            CharacterType.SANTA: RED,
            CharacterType.ELF: YELLOW,
            CharacterType.RUDOLPH: ORANGE
        }
        return color_map.get(self.type, WHITE)
    
    def update(self, dt: float):
        """Обновляет позицию персонажа"""
        if self.is_alive:
            self.x += self.speed * dt
    
    def draw(self, screen: pygame.Surface):
        """Отрисовывает персонажа"""
        if not self.is_alive:
            return
        
        # Простое представление персонажа (в будущем будут спрайты)
        pygame.draw.rect(screen, self.color, (self.x, self.y, self.width, self.height))
        
        # Добавляем текстовую метку
        font = pygame.font.Font(None, 24)
        label = self._get_label()
        text = font.render(label, True, BLACK)
        text_rect = text.get_rect(center=(self.x + self.width // 2, self.y + self.height // 2))
        screen.blit(text, text_rect)
    
    def _get_label(self) -> str:
        """Возвращает метку персонажа"""
        label_map = {
            CharacterType.SNOWMAN: "⛄",
            CharacterType.GRINCH: "👹",
            CharacterType.SANTA: "🎅",
            CharacterType.ELF: "🧝",
            CharacterType.RUDOLPH: "🦌"
        }
        return label_map.get(self.type, "?")
    
    def get_rect(self) -> pygame.Rect:
        """Возвращает прямоугольник для проверки коллизий"""
        return pygame.Rect(self.x, self.y, self.width, self.height)
    
    def is_point_inside(self, point: Tuple[int, int]) -> bool:
        """Проверяет, находится ли точка внутри персонажа"""
        return self.get_rect().collidepoint(point)


class Game:
    """Основной класс игры"""
    
    def __init__(self):
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Новогодняя Аркадная Игра")
        self.clock = pygame.time.Clock()
        self.running = True
        
        # Игровые переменные
        self.characters: List[Character] = []
        self.score = 0
        self.spawn_timer = 0.0
        self.spawn_interval = 2.0  # секунды между появлением врагов
        
        # Шрифты
        self.font_large = pygame.font.Font(None, 48)
        self.font_medium = pygame.font.Font(None, 36)
        self.font_small = pygame.font.Font(None, 24)
        
    def spawn_character(self):
        """Создает нового персонажа"""
        char_types = [
            CharacterType.SNOWMAN,
            CharacterType.SNOWMAN,  # Снеговики чаще всего
            CharacterType.GRINCH,
            CharacterType.ELF,
            CharacterType.SANTA,
            CharacterType.RUDOLPH  # Рудольф редко
        ]
        
        char_type = random.choice(char_types)
        
        # Случайная высота
        y = random.randint(100, SCREEN_HEIGHT - 200)
        
        # Скорость в зависимости от типа
        speed_map = {
            CharacterType.SNOWMAN: 50,
            CharacterType.GRINCH: 75,
            CharacterType.SANTA: 100,
            CharacterType.ELF: 90,
            CharacterType.RUDOLPH: 40
        }
        speed = speed_map.get(char_type, 50)
        
        character = Character(char_type, -60, y, speed)
        self.characters.append(character)
    
    def handle_click(self, pos: Tuple[int, int]):
        """Обрабатывает клик мыши"""
        for character in self.characters:
            if character.is_alive and character.is_point_inside(pos):
                character.is_alive = False
                self.score += character.points
                print(f"Убит {character.type.value}! Очки: +{character.points} (Всего: {self.score})")
                break
    
    def update(self, dt: float):
        """Обновляет состояние игры"""
        # Обновление персонажей
        for character in self.characters:
            character.update(dt)
        
        # Удаление персонажей, которые вышли за экран
        self.characters = [c for c in self.characters if c.x < SCREEN_WIDTH + 100]
        
        # Спавн новых персонажей
        self.spawn_timer += dt
        if self.spawn_timer >= self.spawn_interval:
            self.spawn_character()
            self.spawn_timer = 0.0
    
    def draw(self):
        """Отрисовывает игру"""
        # Фон (темно-синий как ночное небо)
        self.screen.fill((20, 30, 60))
        
        # Рисуем персонажей
        for character in self.characters:
            character.draw(self.screen)
        
        # Отрисовка UI
        self.draw_ui()
        
        pygame.display.flip()
    
    def draw_ui(self):
        """Отрисовывает интерфейс"""
        # Очки
        score_text = self.font_large.render(f"Очки: {self.score}", True, WHITE)
        self.screen.blit(score_text, (20, 20))
        
        # Инструкция
        instruction_text = self.font_small.render("Кликните по персонажу, чтобы убить его", True, WHITE)
        self.screen.blit(instruction_text, (20, SCREEN_HEIGHT - 40))
    
    def run(self):
        """Главный игровой цикл"""
        last_time = pygame.time.get_ticks()
        
        while self.running:
            current_time = pygame.time.get_ticks()
            dt = (current_time - last_time) / 1000.0  # Конвертация в секунды
            last_time = current_time
            
            # Обработка событий
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Левая кнопка мыши
                        self.handle_click(event.pos)
            
            # Обновление игры
            self.update(dt)
            
            # Отрисовка
            self.draw()
            
            # Ограничение FPS
            self.clock.tick(FPS)
        
        pygame.quit()
        sys.exit()


def main():
    """Точка входа в программу"""
    game = Game()
    game.run()


if __name__ == "__main__":
    main()


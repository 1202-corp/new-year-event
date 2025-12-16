"""Main entry point for New Year Arcade Game"""
from game.config import Config
from game.game import Game


def main():
    """Entry point"""
    # Setup display from config
    Config.setup_display()
    
    # Create and run game
    game = Game()
    game.run()


if __name__ == "__main__":
    main()

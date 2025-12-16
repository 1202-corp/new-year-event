"""Logging configuration module"""
import logging
import sys
from typing import Optional


def setup_logging(level: Optional[int] = None) -> logging.Logger:
    """Setup logging configuration"""
    if level is None:
        level = logging.INFO
    
    # Create logger
    logger = logging.getLogger("new_year_game")
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers.clear()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '[%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger


# Global logger instance
_logger: Optional[logging.Logger] = None


def get_logger() -> logging.Logger:
    """Get the global logger instance"""
    global _logger
    if _logger is None:
        _logger = setup_logging()
    return _logger


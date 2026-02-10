"""Face texture module for loading and applying faces to characters"""
import pygame
import random
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from game.logger import get_logger

logger = get_logger()


class FaceTextureManager:
    """Manages loading and applying face textures to characters"""
    
    def __init__(self, faces_dir: Path = Path("faces")):
        """
        Initialize face texture manager
        
        Args:
            faces_dir: Directory containing face images
        """
        self.faces_dir = faces_dir
        self.face_images: list[pygame.Surface] = []
        self.last_face_count = 0
        self._load_faces()
    
    def _load_faces(self) -> None:
        """Load all face images from faces directory"""
        if not self.faces_dir.exists():
            logger.debug(f"Faces directory {self.faces_dir} does not exist. No faces loaded.")
            self.last_face_count = 0
            return
        
        # Count image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        face_files = [f for f in self.faces_dir.iterdir() 
                     if f.suffix.lower() in image_extensions]
        
        # Only reload if count changed
        if len(face_files) == self.last_face_count and len(self.face_images) > 0:
            return  # No new faces, skip reload
        
        # Clear existing faces to reload
        self.face_images.clear()
        self.last_face_count = len(face_files)
        
        for face_file in face_files:
            try:
                face_surface = pygame.image.load(str(face_file))
                self.face_images.append(face_surface)
                logger.debug(f"Loaded face: {face_file.name}")
            except Exception as e:
                logger.warning(f"Failed to load face {face_file.name}: {e}")
        
        if len(self.face_images) > 0:
            logger.info(f"Loaded {len(self.face_images)} face images")
        else:
            logger.debug("No face images loaded")
    
    def get_random_face(self) -> Optional[pygame.Surface]:
        """
        Get a random face image
        
        Returns:
            Random face surface or None if no faces available
        """
        if not self.face_images:
            return None
        return random.choice(self.face_images)
    
    def apply_color_tint(self, face_surface: pygame.Surface, color: Tuple[int, int, int], 
                        intensity: float = 0.5) -> pygame.Surface:
        """
        Apply color tint to face surface with given intensity
        
        Args:
            face_surface: Original face surface
            color: RGB color tuple
            intensity: Tint intensity (0.0 = no tint, 1.0 = full color)
        
        Returns:
            Tinted face surface
        """
        # Create a copy to avoid modifying original
        # Convert to RGB format if needed (remove alpha channel)
        if face_surface.get_flags() & pygame.SRCALPHA:
            # Has alpha channel, convert to RGB
            tinted = pygame.Surface(face_surface.get_size(), pygame.SRCALPHA)
            tinted.blit(face_surface, (0, 0))
            # Convert to RGB for processing
            tinted_rgb = pygame.Surface(tinted.get_size())
            tinted_rgb.blit(tinted, (0, 0))
            tinted = tinted_rgb
        else:
            tinted = face_surface.copy()
        
        # Convert to numpy array for processing
        # Note: array3d returns (width, height, 3) array
        face_array = pygame.surfarray.array3d(tinted)
        
        # Normalize color to 0-1 range
        r, g, b = color
        color_norm = np.array([r / 255.0, g / 255.0, b / 255.0])
        
        # Apply tint: blend original with color based on intensity
        # Formula: result = original * (1 - intensity) + color * intensity
        # face_array shape is (width, height, 3), we need to process each channel
        face_array = face_array.astype(np.float32)
        for c in range(3):  # RGB channels
            face_array[:, :, c] = (
                face_array[:, :, c] * (1.0 - intensity) + 
                color_norm[c] * 255.0 * intensity
            )
        face_array = np.clip(face_array, 0, 255).astype(np.uint8)
        
        # Convert back to surface
        pygame.surfarray.blit_array(tinted, face_array)
        
        return tinted
    
    def should_use_face(self) -> bool:
        """
        Determine if face should be used (2/3 probability)
        
        Returns:
            True with 2/3 probability, False otherwise
        """
        return random.random() < (2.0 / 3.0)


# Global face texture manager instance
_face_manager: Optional[FaceTextureManager] = None


def get_face_manager() -> FaceTextureManager:
    """Get or create global face texture manager"""
    global _face_manager
    if _face_manager is None:
        _face_manager = FaceTextureManager()
    return _face_manager


def reload_faces() -> None:
    """Reload faces from directory"""
    global _face_manager
    if _face_manager is not None:
        _face_manager.face_images.clear()
        _face_manager._load_faces()


"""Score management module"""


class ScoreManager:
    """Manages game score"""
    
    def __init__(self):
        self.score = 0
    
    def add_points(self, points: int) -> None:
        """Adds points to score"""
        self.score += points
    
    def reset(self) -> None:
        """Resets score to zero"""
        self.score = 0
    
    def get_score(self) -> int:
        """Returns current score"""
        return self.score


# cricket_configs.py
from dataclasses import dataclass
from typing import List, Tuple
# Remove this line: from configs import CricketPitchConfiguration

@dataclass
class CricketPitchConfiguration:
    """Cricket pitch configuration with dimensions and key points"""
    
    # Cricket pitch dimensions in meters (22 yards x 10 feet)
    pitch_length: float = 20.12  # 22 yards in meters
    pitch_width: float = 3.05    # 10 feet in meters
    
    # Field dimensions (approximate)
    field_length: float = 150.0  # Boundary to boundary
    field_width: float = 130.0
    
    # Key points on cricket pitch for homography (12 points)
    # Pitch corners and important markers
    vertices: List[Tuple[float, float]] = None
    
    # Edges to connect vertices for pitch markings
    edges: List[Tuple[int, int]] = None
    
    # Labels for vertices
    labels: List[str] = None
    
    # Colors for different teams and entities
    colors: List[str] = None
    
    def __post_init__(self):
        if self.vertices is None:
            # Define 12 key points on cricket pitch
            self.vertices = [
                # Pitch corners (4 points)
                (0, 0),                    # 1: Top-left of pitch
                (self.pitch_length, 0),    # 2: Top-right
                (self.pitch_length, self.pitch_width),  # 3: Bottom-right
                (0, self.pitch_width),     # 4: Bottom-left
                
                # Crease markings (4 points)
                (self.pitch_length * 0.33, self.pitch_width * 0.33),  # 5: Bowling crease left
                (self.pitch_length * 0.67, self.pitch_width * 0.33),  # 6: Bowling crease right
                (self.pitch_length * 0.33, self.pitch_width * 0.67),  # 7: Batting crease left
                (self.pitch_length * 0.67, self.pitch_width * 0.67),  # 8: Batting crease right
                
                # Stumps positions (3 points)
                (self.pitch_length * 0.5, 0),              # 9: Top stumps
                (self.pitch_length * 0.5, self.pitch_width),  # 10: Bottom stumps
                (self.pitch_length * 0.5, self.pitch_width * 0.5),  # 11: Middle stumps
                
                # Pitch center
                (self.pitch_length * 0.5, self.pitch_width * 0.5)  # 12: Center
            ]
        
        if self.edges is None:
            # Connect vertices to form pitch markings
            self.edges = [
                (1, 2), (2, 3), (3, 4), (4, 1),  # Pitch boundary
                (5, 6), (7, 8),                   # Crease lines
                (9, 10),                          # Stumps line
                (11, 12)                          # Center line
            ]
        
        if self.labels is None:
            self.labels = [
                'TL', 'TR', 'BR', 'BL',  # Pitch corners
                'BCL', 'BCR', 'BTL', 'BTR',  # Crease points
                'TS', 'BS', 'MS', 'C'        # Stumps and center
            ]
        
        if self.colors is None:
            self.colors = [
                '#FF1493',  # Team A (Batting) - Deep Pink
                '#00BFFF',  # Team B (Fielding) - Deep Sky Blue
                '#32CD32',  # Umpire - Lime Green
                '#FFD700',  # Ball - Gold
                '#FFFFFF',  # Pitch markings - White
            ]
import numpy as np

def calculate_angle(a, b, c):
    """Calculate the angle between three points: a (first), b (mid), and c (end)."""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
        
    return angle

class ExerciseBase:
    def __init__(self):
        self.counter = 0
        self.stage = None
        self.feedback = ""
        # Default relevant landmarks (override in subclasses)
        self.relevant_landmarks = []
        
    def reset(self):
        self.counter = 0
        self.stage = None
        self.feedback = ""

    def process(self, landmarks):
        """Processes landmarks for a single frame, updates counter/stage, returns (counter, stage, feedback, rendering_points)."""
        raise NotImplementedError("Subclasses must implement process()")

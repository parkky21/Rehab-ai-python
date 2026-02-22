from .base import ExerciseBase

class HeelRaises(ExerciseBase):
    def __init__(self):
        super().__init__()
        # Left Heel (29) or Left Ankle (27), and Left Toe (31)
        # We can track the height distance of the ankle relative to the toe
        self.relevant_landmarks = [27, 31]
        
    def process(self, landmarks):
        ankle = landmarks[27]
        toe = landmarks[31]
        
        # Y is 0 at top, 1 at bottom.
        # When standing flat, ankle Y is basically equal to toe Y
        vertical_dist = toe.y - ankle.y
        
        if vertical_dist < 0.02: # Flat foot
            self.stage = "down"
            self.feedback = "Raise heels slowly"
        elif vertical_dist > 0.05 and self.stage == "down": # Heel is raised higher than toe
            self.stage = "up"
            self.counter += 1
            self.feedback = "Slowly lower down"
            
        return self.counter, self.stage, self.feedback, {"angle": 0, "points": [ankle, toe]}

from .base import ExerciseBase, calculate_angle
import math

class StandingHipAbduction(ExerciseBase):
    def __init__(self):
        super().__init__()
        # Left Hip, Knee, Ankle 
        self.relevant_landmarks = [23, 25, 27]
        
    def process(self, landmarks):
        # We also need the other hip to track lateral movement correctly, 
        # or we just see the angle of the leg pulling away from the vertical center line
        shoulder = landmarks[11]
        hip = landmarks[23]
        ankle = landmarks[27]
        
        # Abduction means laterally moving leg outward away from the body plane
        angle = calculate_angle(shoulder, hip, ankle)
        
        # Normally, standing straight = ~170-180 degrees
        if angle > 170:
            self.stage = "down"
            self.feedback = "Raise leg to side"
        if angle < 150 and self.stage == "down": # Kicking out to the side reduces the vertical torso-hip-ankle angle
            self.stage = "up"
            self.counter += 1
            self.feedback = "Good side raise!"
            
        return self.counter, self.stage, self.feedback, {"angle": angle, "points": [shoulder, hip, ankle]}

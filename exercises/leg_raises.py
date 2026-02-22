from .base import ExerciseBase, calculate_angle

class LegRaises(ExerciseBase):
    def __init__(self):
        super().__init__()
        # Left Hip, Knee, Ankle
        self.relevant_landmarks = [23, 25, 27]
        
    def process(self, landmarks):
        hip = landmarks[23]
        knee = landmarks[25]
        ankle = landmarks[27]
        
        # A straight leg raise while lying down or standing 
        # Typically looking at hip flexion angle (angle between torso, hip, knee)
        # We need shoulder to track torso angle correctly:
        shoulder = landmarks[11]
        
        angle = calculate_angle(shoulder, hip, knee)
        
        if angle > 160: # Leg straight down with torso
            self.stage = "down"
            self.feedback = "Raise leg"
        if angle < 110 and self.stage == "down": # Leg raised forward
            self.stage = "up"
            self.counter += 1
            self.feedback = "Good raise!"
            
        return self.counter, self.stage, self.feedback, {"angle": angle, "points": [shoulder, hip, knee]}

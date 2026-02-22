from .base import ExerciseBase, calculate_angle

class StandingHipExtension(ExerciseBase):
    def __init__(self):
        super().__init__()
        # Shoulder, Hip, Knee, Ankle (using left side for profile)
        self.relevant_landmarks = [11, 23, 25, 27]
        
    def process(self, landmarks):
        shoulder = landmarks[11]
        hip = landmarks[23]
        ankle = landmarks[27]
        
        # Extension means moving the leg backward past the torso line
        angle = calculate_angle(shoulder, hip, ankle)
        
        # In a standing position, this should be ~170-180
        if angle < 170:
            self.stage = "down"
            self.feedback = "Kick leg backward"
            
        # When leg kicks back, the interior angle increases beyond 180 (or decreases from the front depending on calculation)
        # Using our 0-180 absolute angle calculation:
        if angle > 190 or (angle < 165 and self.stage == "down"): # Adjust threshold based on testing
            self.stage = "up"
            self.counter += 1
            self.feedback = "Good backward extension!"
            
        return self.counter, self.stage, self.feedback, {"angle": angle, "points": [shoulder, hip, ankle]}

from .base import ExerciseBase, calculate_angle

class Squats(ExerciseBase):
    def __init__(self):
        super().__init__()
        # Left Hip, Left Knee, Left Ankle (using left side for profile view)
        self.relevant_landmarks = [23, 25, 27] 
        
    def process(self, landmarks):
        hip = landmarks[23]
        knee = landmarks[25]
        ankle = landmarks[27]
        
        angle = calculate_angle(hip, knee, ankle)
        
        if angle > 160:
            self.stage = "up"
            self.feedback = "Squat down"
        if angle < 90 and self.stage == "up":
            self.stage = "down"
            self.counter += 1
            self.feedback = "Good depth!"
            
        return self.counter, self.stage, self.feedback, {"angle": angle, "points": [hip, knee, ankle]}

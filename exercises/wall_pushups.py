from .base import ExerciseBase, calculate_angle

class WallPushups(ExerciseBase):
    def __init__(self):
        super().__init__()
        # Left Shoulder, Elbow, Wrist
        self.relevant_landmarks = [11, 13, 15]
        
    def process(self, landmarks):
        shoulder = landmarks[11]
        elbow = landmarks[13]
        wrist = landmarks[15]
        
        angle = calculate_angle(shoulder, elbow, wrist)
        
        if angle > 150:
            self.stage = "up"
            self.feedback = "Leaning into wall"
        if angle < 90 and self.stage == "up":
            self.stage = "down"
            self.counter += 1
            self.feedback = "Good push!"
            
        return self.counter, self.stage, self.feedback, {"angle": angle, "points": [shoulder, elbow, wrist]}

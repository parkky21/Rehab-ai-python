from .base import ExerciseBase, calculate_angle

class SideArmRaises(ExerciseBase):
    def __init__(self):
        super().__init__()
        # Coronal plane shoulder abduction
        # Left Hip, Shoulder, Wrist
        self.relevant_landmarks = [23, 11, 15]
        
    def process(self, landmarks):
        hip = landmarks[23]
        shoulder = landmarks[11]
        wrist = landmarks[15]
        
        angle = calculate_angle(hip, shoulder, wrist)
        
        if angle < 35:
            self.stage = "down"
            self.feedback = "Raise arms to side"
        if angle > 85 and angle < 110 and self.stage == "down":
            self.stage = "up"
            self.counter += 1
            self.feedback = "Good shoulder abduction!"
            
        return self.counter, self.stage, self.feedback, {"angle": angle, "points": [hip, shoulder, wrist]}

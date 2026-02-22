from .base import ExerciseBase, calculate_angle

class ForwardArmRaises(ExerciseBase):
    def __init__(self):
        super().__init__()
        # Left Hip, Shoulder, Elbow
        self.relevant_landmarks = [23, 11, 13]
        
    def process(self, landmarks):
        hip = landmarks[23]
        shoulder = landmarks[11]
        elbow = landmarks[13]
        
        angle = calculate_angle(hip, shoulder, elbow)
        
        if angle < 30: # Arm straight down alongside hip
            self.stage = "down"
            self.feedback = "Raise arms forward"
        if angle > 80 and self.stage == "down": # Arm raised parallel to ground
            self.stage = "up"
            self.counter += 1
            self.feedback = "Good shoulder flexion!"
            
        return self.counter, self.stage, self.feedback, {"angle": angle, "points": [hip, shoulder, elbow]}

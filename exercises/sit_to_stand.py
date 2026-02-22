from .base import ExerciseBase

class SitToStand(ExerciseBase):
    def __init__(self):
        super().__init__()
        # Left Hip, Knee, Ankle
        self.relevant_landmarks = [23, 25, 27]
        self.seated = True
        
    def process(self, landmarks):
        # We look at the Y position of the hips relative to knees
        # In a standing position, hips are much higher (lower Y value) than knees
        # In a seated position, hips are closer to the knee's Y level.
        hip = landmarks[23]
        knee = landmarks[25]
        
        # Calculate vertical distance (normalized 0 to 1, where 0 is top of image)
        vertical_dist = knee.y - hip.y
        
        if vertical_dist < 0.1: # Hips are near or below knees (Seated)
            self.stage = "seated"
            self.feedback = "Stand up"
        elif vertical_dist > 0.3 and self.stage == "seated": # Hips are much higher than knees (Standing)
            self.stage = "standing"
            self.counter += 1
            self.feedback = "Good stand!"
            
        points = [hip, knee]
        return self.counter, self.stage, self.feedback, {"angle": 0, "points": points}

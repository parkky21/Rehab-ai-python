from .base import ExerciseBase

class Marching(ExerciseBase):
    def __init__(self):
        super().__init__()
        # Left Hip, Knee, Ankle, Right Hip, Knee, Ankle
        self.relevant_landmarks = [23, 25, 27, 24, 26, 28]
        self.last_leg_lifted = None # "left" or "right"
        
    def process(self, landmarks):
        l_hip = landmarks[23]
        l_knee = landmarks[25]
        r_hip = landmarks[24]
        r_knee = landmarks[26]
        
        # Marching process: lift alternating knees up toward hip level
        l_dist = l_hip.y - l_knee.y # Distance between hip and knee Y. If knee goes up, dist approaches 0.
        r_dist = r_hip.y - r_knee.y
        
        # A knee is "lifted" if it's raised significantly (its Y gets closer to the Hip's Y)
        threshold = -0.1 # This means knee is above hip, or close to it depending on camera angle
        
        if l_knee.y < (l_hip.y + 0.1): # Left knee is lifted up
            if self.last_leg_lifted != "left":
                self.stage = "left lifted"
                self.feedback = "Now lift right"
                if self.last_leg_lifted == "right": 
                    self.counter += 1  # 1 rep is a full cycle or just alternating? Let's say every knee lift is 1 side rep
                self.last_leg_lifted = "left"
        elif r_knee.y < (r_hip.y + 0.1): # Right knee is lifted
            if self.last_leg_lifted != "right":
                self.stage = "right lifted"
                self.feedback = "Now lift left"
                if self.last_leg_lifted == "left":
                    self.counter += 1
                self.last_leg_lifted = "right"
                
        # To show the tracking line, we'll draw both legs
        points = [l_hip, l_knee, r_hip, r_knee]
        return self.counter, self.stage, self.feedback, {"angle": 0, "points": points}

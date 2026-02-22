import numpy as np

def calculate_angle(a, b, c):
    """Calculate the angle between three points: a (first), b (mid), and c (end)."""
    a = np.array([a.x, a.y])
    b = np.array([b.x, b.y])
    c = np.array([c.x, c.y])
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360.0 - angle
        
    return angle

class ExerciseBase:
    def __init__(self):
        self.counter = 0
        self.stage = None
        self.feedback = ""
        # Default relevant landmarks (override in subclasses)
        self.relevant_landmarks = []
        
    def reset(self):
        self.counter = 0
        self.stage = None
        self.feedback = ""

    def process(self, landmarks):
        """Processes landmarks for a single frame, updates counter/stage, returns (counter, stage, feedback, rendering_points)."""
        raise NotImplementedError("Subclasses must implement process()")


class BicepCurlLeft(ExerciseBase):
    def __init__(self):
        super().__init__()
        self.relevant_landmarks = [11, 13, 15] # Left Shoulder, Elbow, Wrist
        
    def process(self, landmarks):
        shoulder = landmarks[11]
        elbow = landmarks[13]
        wrist = landmarks[15]
        
        angle = calculate_angle(shoulder, elbow, wrist)
        
        if angle > 160:
            self.stage = "down"
            self.feedback = "Curl up"
        if angle < 30 and self.stage == "down":
            self.stage = "up"
            self.counter += 1
            self.feedback = "Good rep!"
            
        return self.counter, self.stage, self.feedback, {"angle": angle, "points": [shoulder, elbow, wrist]}

class BicepCurlRight(ExerciseBase):
    def __init__(self):
        super().__init__()
        self.relevant_landmarks = [12, 14, 16] # Right Shoulder, Elbow, Wrist
        
    def process(self, landmarks):
        shoulder = landmarks[12]
        elbow = landmarks[14]
        wrist = landmarks[16]
        
        angle = calculate_angle(shoulder, elbow, wrist)
        
        if angle > 160:
            self.stage = "down"
            self.feedback = "Curl up"
        if angle < 30 and self.stage == "down":
            self.stage = "up"
            self.counter += 1
            self.feedback = "Good rep!"
            
        return self.counter, self.stage, self.feedback, {"angle": angle, "points": [shoulder, elbow, wrist]}

class Squat(ExerciseBase):
    def __init__(self):
        super().__init__()
        self.relevant_landmarks = [23, 25, 27] # Left Hip, Left Knee, Left Ankle (using left side for profile view)
        
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

class LungeLeft(ExerciseBase):
    def __init__(self):
        super().__init__()
        self.relevant_landmarks = [23, 25, 27] # Left Hip, Left Knee, Left Ankle
        
    def process(self, landmarks):
        hip = landmarks[23]
        knee = landmarks[25]
        ankle = landmarks[27]
        
        angle = calculate_angle(hip, knee, ankle)
        
        if angle > 160:
            self.stage = "up"
            self.feedback = "Lunge down"
        if angle < 90 and self.stage == "up":
            self.stage = "down"
            self.counter += 1
            self.feedback = "Good depth!"
            
        return self.counter, self.stage, self.feedback, {"angle": angle, "points": [hip, knee, ankle]}

class LungeRight(ExerciseBase):
    def __init__(self):
        super().__init__()
        self.relevant_landmarks = [24, 26, 28] # Right Hip, Right Knee, Right Ankle
        
    def process(self, landmarks):
        hip = landmarks[24]
        knee = landmarks[26]
        ankle = landmarks[28]
        
        angle = calculate_angle(hip, knee, ankle)
        
        if angle > 160:
            self.stage = "up"
            self.feedback = "Lunge down"
        if angle < 90 and self.stage == "up":
            self.stage = "down"
            self.counter += 1
            self.feedback = "Good depth!"
            
        return self.counter, self.stage, self.feedback, {"angle": angle, "points": [hip, knee, ankle]}

class ShoulderPress(ExerciseBase):
    def __init__(self):
        super().__init__()
        self.relevant_landmarks = [11, 13, 15, 12, 14, 16] # Both arms
        
    def process(self, landmarks):
        l_shoulder = landmarks[11]
        l_elbow = landmarks[13]
        l_wrist = landmarks[15]
        
        # We'll just track the left arm angle for counting, but we can draw both
        angle = calculate_angle(l_shoulder, l_elbow, l_wrist)
        
        if angle < 90:
            self.stage = "down"
            self.feedback = "Press up"
        if angle > 160 and self.stage == "down":
            self.stage = "up"
            self.counter += 1
            self.feedback = "Good extension!"
            
        return self.counter, self.stage, self.feedback, {"angle": angle, "points": [l_shoulder, l_elbow, l_wrist]}

class LateralRaise(ExerciseBase):
    def __init__(self):
        super().__init__()
        self.relevant_landmarks = [23, 11, 13] # Left Hip, Left Shoulder, Left Elbow
        
    def process(self, landmarks):
        hip = landmarks[23]
        shoulder = landmarks[11]
        elbow = landmarks[13]
        
        angle = calculate_angle(hip, shoulder, elbow)
        
        if angle < 30:
            self.stage = "down"
            self.feedback = "Raise arms"
        if angle > 80 and self.stage == "down":
            self.stage = "up"
            self.counter += 1
            self.feedback = "Good height!"
            
        return self.counter, self.stage, self.feedback, {"angle": angle, "points": [hip, shoulder, elbow]}

class HipAbduction(ExerciseBase):
    def __init__(self):
        super().__init__()
        self.relevant_landmarks = [11, 23, 27] # Left Shoulder, Left Hip, Left Ankle
        
    def process(self, landmarks):
        shoulder = landmarks[11]
        hip = landmarks[23]
        ankle = landmarks[27]
        
        angle = calculate_angle(shoulder, hip, ankle)
        
        if angle > 170:
            self.stage = "down"
            self.feedback = "Raise leg laterally"
        if angle < 150 and self.stage == "down":
            self.stage = "up"
            self.counter += 1
            self.feedback = "Good raise!"
            
        return self.counter, self.stage, self.feedback, {"angle": angle, "points": [shoulder, hip, ankle]}

class WallPushup(ExerciseBase):
    def __init__(self):
        super().__init__()
        self.relevant_landmarks = [11, 13, 15] # Left Shoulder, Elbow, Wrist
        
    def process(self, landmarks):
        shoulder = landmarks[11]
        elbow = landmarks[13]
        wrist = landmarks[15]
        
        angle = calculate_angle(shoulder, elbow, wrist)
        
        if angle > 160:
            self.stage = "up"
            self.feedback = "Push towards wall"
        if angle < 90 and self.stage == "up":
            self.stage = "down"
            self.counter += 1
            self.feedback = "Good push!"
            
        return self.counter, self.stage, self.feedback, {"angle": angle, "points": [shoulder, elbow, wrist]}

# Dictionary exposing the exercises
EXERCISES = {
    "Bicep Curl (Left)": BicepCurlLeft(),
    "Bicep Curl (Right)": BicepCurlRight(),
    "Squat": Squat(),
    "Lunge (Left)": LungeLeft(),
    "Lunge (Right)": LungeRight(),
    "Shoulder Press": ShoulderPress(),
    "Lateral Raise": LateralRaise(),
    "Hip Abduction": HipAbduction(),
    "Wall Push-up": WallPushup()
}

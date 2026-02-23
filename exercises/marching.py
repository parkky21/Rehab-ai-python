from .base import ExerciseBase
from pipeline.scorer import ExerciseConfig


class Marching(ExerciseBase):
    def __init__(self):
        super().__init__()
        self.relevant_landmarks = [23, 25, 24, 26]
        self.last_leg_lifted = None
        self.config = ExerciseConfig(
            target_rom=30.0,
            ideal_rep_time=3.0,
            acceptable_sway=0.025,
            weight_rom=0.3,
            weight_stability=0.4,
            weight_tempo=0.3,
        )
        self.scorer.config = self.config

    def reset(self):
        super().reset()
        self.last_leg_lifted = None

    def process(self, landmarks):
        l_hip = landmarks[23]
        l_knee = landmarks[25]
        r_hip = landmarks[24]
        r_knee = landmarks[26]

        l_lift = max(0, (l_hip.y + 0.05) - l_knee.y) * 100  # Lower threshold
        r_lift = max(0, (r_hip.y + 0.05) - r_knee.y) * 100
        self.rom_tracker.update(max(l_lift, r_lift))

        self.rep_completed = False

        if l_knee.y < (l_hip.y + 0.05):  # Lower threshold for detecting lift
            if self.last_leg_lifted != "left":
                self.stage = "left lifted"
                self.feedback = "Now lift right"
                if self.last_leg_lifted == "right":
                    self._on_rep_start()
                    self.counter += 1
                    self._on_rep_complete()
                    self.feedback = f"Rep done! Score: {self.last_rep_scores['final_score']}"
                self.last_leg_lifted = "left"
        elif r_knee.y < (r_hip.y + 0.05):
            if self.last_leg_lifted != "right":
                self.stage = "right lifted"
                self.feedback = "Now lift left"
                if self.last_leg_lifted == "left":
                    self._on_rep_start()
                    self.counter += 1
                    self._on_rep_complete()
                    self.feedback = f"Rep done! Score: {self.last_rep_scores['final_score']}"
                self.last_leg_lifted = "right"

        points = [l_hip, l_knee, r_hip, r_knee]
        return self.counter, self.stage, self.feedback, {"angle": 0, "points": points}

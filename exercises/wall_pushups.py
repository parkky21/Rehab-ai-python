from .base import ExerciseBase
from pipeline.scorer import ExerciseConfig
from pipeline.feature_engine import calculate_angle_2d


class WallPushups(ExerciseBase):
    def __init__(self):
        super().__init__()
        self.relevant_landmarks = [11, 13, 15]
        self.config = ExerciseConfig(
            target_rom=60.0,
            ideal_rep_time=4.0,
            acceptable_sway=0.015,
            weight_rom=0.45,
            weight_stability=0.25,
            weight_tempo=0.3,
        )
        self.scorer.config = self.config

    def process(self, landmarks):
        shoulder = landmarks[11]
        elbow = landmarks[13]
        wrist = landmarks[15]

        angle = calculate_angle_2d(shoulder, elbow, wrist)
        self.rom_tracker.update(angle)
        self.rep_completed = False

        if angle > 150:
            self._on_rep_start()
            self.stage = "up"
            self.feedback = "Lean into wall"
        if angle < 130 and self.stage == "up":  # Low threshold so partial pushups count
            self.stage = "down"
            self.counter += 1
            self._on_rep_complete()
            self.feedback = f"Rep done! Score: {self.last_rep_scores['final_score']}"

        return self.counter, self.stage, self.feedback, {"angle": angle, "points": [shoulder, elbow, wrist]}

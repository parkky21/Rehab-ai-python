from .base import ExerciseBase
from pipeline.scorer import ExerciseConfig
from pipeline.feature_engine import calculate_angle_2d


class LegRaises(ExerciseBase):
    def __init__(self):
        super().__init__()
        self.relevant_landmarks = [11, 23, 25]
        self.config = ExerciseConfig(
            target_rom=50.0,
            ideal_rep_time=4.0,
            acceptable_sway=0.02,
            weight_rom=0.45,
            weight_stability=0.3,
            weight_tempo=0.25,
        )
        self.scorer.config = self.config

    def process(self, landmarks):
        shoulder = landmarks[11]
        hip = landmarks[23]
        knee = landmarks[25]

        angle = calculate_angle_2d(shoulder, hip, knee)
        self.rom_tracker.update(angle)
        self.rep_completed = False

        if angle > 160:
            self._on_rep_start()
            self.stage = "down"
            self.feedback = "Raise leg"
        if angle < 150 and self.stage == "down":  # Low threshold: even small raises count
            self.stage = "up"
            self.counter += 1
            self._on_rep_complete()
            self.feedback = f"Rep done! Score: {self.last_rep_scores['final_score']}"

        return self.counter, self.stage, self.feedback, {"angle": angle, "points": [shoulder, hip, knee]}

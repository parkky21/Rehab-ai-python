from .base import ExerciseBase
from pipeline.scorer import ExerciseConfig
from pipeline.feature_engine import calculate_angle_2d


class StandingHipAbduction(ExerciseBase):
    def __init__(self):
        super().__init__()
        self.relevant_landmarks = [11, 23, 27]
        self.config = ExerciseConfig(
            target_rom=25.0,
            ideal_rep_time=4.0,
            acceptable_sway=0.025,
            weight_rom=0.35,
            weight_stability=0.4,
            weight_tempo=0.25,
        )
        self.scorer.config = self.config

    def process(self, landmarks):
        shoulder = landmarks[11]
        hip = landmarks[23]
        ankle = landmarks[27]

        angle = calculate_angle_2d(shoulder, hip, ankle)
        self.rom_tracker.update(angle)
        self.rep_completed = False

        if angle > 170:
            self._on_rep_start()
            self.stage = "down"
            self.feedback = "Raise leg to side"
        if angle < 165 and self.stage == "down":  # Low threshold: even slight abduction counts
            self.stage = "up"
            self.counter += 1
            self._on_rep_complete()
            self.feedback = f"Rep done! Score: {self.last_rep_scores['final_score']}"

        return self.counter, self.stage, self.feedback, {"angle": angle, "points": [shoulder, hip, ankle]}

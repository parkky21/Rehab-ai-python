from .base import ExerciseBase
from pipeline.scorer import ExerciseConfig
from pipeline.feature_engine import calculate_angle_2d


class SideArmRaises(ExerciseBase):
    def __init__(self):
        super().__init__()
        self.relevant_landmarks = [23, 11, 15]
        self.config = ExerciseConfig(
            target_rom=55.0,
            ideal_rep_time=4.0,
            acceptable_sway=0.015,
            weight_rom=0.45,
            weight_stability=0.25,
            weight_tempo=0.3,
        )
        self.scorer.config = self.config

    def process(self, landmarks):
        hip = landmarks[23]
        shoulder = landmarks[11]
        wrist = landmarks[15]

        angle = calculate_angle_2d(hip, shoulder, wrist)
        self.rom_tracker.update(angle)
        self.rep_completed = False

        if angle < 35:
            self._on_rep_start()
            self.stage = "down"
            self.feedback = "Raise arms to side"
        if angle > 50 and self.stage == "down":  # Low threshold: even small raises count
            self.stage = "up"
            self.counter += 1
            self._on_rep_complete()
            self.feedback = f"Rep done! Score: {self.last_rep_scores['final_score']}"

        return self.counter, self.stage, self.feedback, {"angle": angle, "points": [hip, shoulder, wrist]}

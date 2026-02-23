from .base import ExerciseBase
from pipeline.scorer import ExerciseConfig
from pipeline.feature_engine import calculate_angle_2d


class Squats(ExerciseBase):
    def __init__(self):
        super().__init__()
        self.relevant_landmarks = [23, 25, 27]
        self.config = ExerciseConfig(
            target_rom=70.0,       # ~160 - ~90 = 70 degrees knee ROM
            ideal_rep_time=4.0,    # 4 seconds per rep for rehab
            acceptable_sway=0.015,
            weight_rom=0.4,
            weight_stability=0.35,
            weight_tempo=0.25,
        )
        self.scorer.config = self.config

    def process(self, landmarks):
        hip = landmarks[23]
        knee = landmarks[25]
        ankle = landmarks[27]

        angle = calculate_angle_2d(hip, knee, ankle)
        self.rom_tracker.update(angle)
        self.rep_completed = False

        if angle > 160:
            self._on_rep_start()
            self.stage = "up"
            self.feedback = "Squat down"
        if angle < 140 and self.stage == "up":  # Low threshold so partial reps count
            self.stage = "down"
            self.counter += 1
            self._on_rep_complete()
            self.feedback = f"Rep done! Score: {self.last_rep_scores['final_score']}"

        return self.counter, self.stage, self.feedback, {"angle": angle, "points": [hip, knee, ankle]}

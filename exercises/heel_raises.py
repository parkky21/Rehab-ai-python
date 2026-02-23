from .base import ExerciseBase
from pipeline.scorer import ExerciseConfig


class HeelRaises(ExerciseBase):
    def __init__(self):
        super().__init__()
        self.relevant_landmarks = [27, 31]
        self.config = ExerciseConfig(
            target_rom=5.0,
            ideal_rep_time=5.0,    # Very controlled tempo for heel raises
            acceptable_sway=0.02,
            tempo_penalty_factor=15.0,
            weight_rom=0.25,
            weight_stability=0.35,
            weight_tempo=0.4,
        )
        self.scorer.config = self.config

    def process(self, landmarks):
        ankle = landmarks[27]
        toe = landmarks[31]

        vertical_dist = toe.y - ankle.y
        proxy_angle = vertical_dist * 100
        self.rom_tracker.update(proxy_angle)
        self.rep_completed = False

        if vertical_dist < 0.02:
            self._on_rep_start()
            self.stage = "down"
            self.feedback = "Raise heels slowly"
        elif vertical_dist > 0.03 and self.stage == "down":  # Lower threshold
            self.stage = "up"
            self.counter += 1
            self._on_rep_complete()
            self.feedback = f"Rep done! Score: {self.last_rep_scores['final_score']}"

        return self.counter, self.stage, self.feedback, {"angle": 0, "points": [ankle, toe]}

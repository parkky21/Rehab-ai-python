from .base import ExerciseBase
from pipeline.scorer import ExerciseConfig


class SitToStand(ExerciseBase):
    def __init__(self):
        super().__init__()
        self.relevant_landmarks = [23, 25]
        self.config = ExerciseConfig(
            target_rom=50.0,
            ideal_rep_time=5.0,    # Slow, controlled sit-to-stand
            acceptable_sway=0.02,
            weight_rom=0.3,
            weight_stability=0.4,
            weight_tempo=0.3,
        )
        self.scorer.config = self.config

    def process(self, landmarks):
        hip = landmarks[23]
        knee = landmarks[25]

        vertical_dist = knee.y - hip.y
        proxy_angle = vertical_dist * 100
        self.rom_tracker.update(proxy_angle)
        self.rep_completed = False

        if vertical_dist < 0.1:
            self._on_rep_start()
            self.stage = "seated"
            self.feedback = "Stand up"
        elif vertical_dist > 0.15 and self.stage == "seated":  # Lower threshold
            self.stage = "standing"
            self.counter += 1
            self._on_rep_complete()
            self.feedback = f"Rep done! Score: {self.last_rep_scores['final_score']}"

        points = [hip, knee]
        return self.counter, self.stage, self.feedback, {"angle": 0, "points": points}

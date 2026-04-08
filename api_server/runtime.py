from dataclasses import dataclass

from api_server.exercise_factory import create_exercise_instance
from pipeline import (
    EMALandmarkSmoother,
    Session,
    SwayTracker,
    create_default_feedback_engine,
    process_landmarks,
)


@dataclass
class RuntimeLandmark:
    x: float
    y: float
    z: float
    visibility: float = 1.0


class RealtimeSessionRuntime:
    def __init__(self, exercise_name: str) -> None:
        self.exercise_name = exercise_name
        self.exercise = create_exercise_instance(exercise_name)
        self.smoother = EMALandmarkSmoother(alpha=0.3)
        self.sway_tracker = SwayTracker(window_size=30)
        self.feedback_engine = create_default_feedback_engine()
        self.session = Session(exercise_name=exercise_name)

    def process_frame(self, landmarks_payload: list[dict]) -> dict:
        if len(landmarks_payload) < 33:
            raise ValueError("Expected 33 landmarks from MediaPipe pose")

        landmarks = [RuntimeLandmark(**lm) for lm in landmarks_payload[:33]]
        smoothed = self.smoother.smooth(landmarks)
        processed, hip_center, _ = process_landmarks(smoothed)
        sway = self.sway_tracker.update(float(hip_center[0]))

        counter, stage, feedback, _ = self.exercise.process(processed)

        rep_event = None
        if self.exercise.rep_completed and self.exercise.last_rep_scores:
            rep_scores = dict(self.exercise.last_rep_scores)
            rep_time = float(rep_scores.get("rep_time", 0.0))
            rom_value = float(rep_scores.get("rom_value", 0.0))
            self.session.add_rep(rep_scores, rom_value=rom_value, rep_time=rep_time)
            rep_event = {
                "rep_number": counter,
                "scores": rep_scores,
                "rep_time": round(rep_time, 3),
                "rom_value": round(rom_value, 2),
                "session_avg": round(self.session.avg_final_score, 1),
            }
            self.exercise.rep_completed = False

        current_rom = 0.0
        if self.exercise.rom_tracker.current_max > float("-inf") and self.exercise.rom_tracker.current_min < float("inf"):
            current_rom = max(0.0, self.exercise.rom_tracker.current_max - self.exercise.rom_tracker.current_min)

        context = {
            "current_rom": current_rom,
            "target_rom": self.exercise.config.target_rom,
            "ideal_rep_time": self.exercise.config.ideal_rep_time,
            "sway": sway,
            "asymmetry_value": 0.0,
        }
        feedback_messages = self.feedback_engine.evaluate(processed, context)

        return {
            "counter": counter,
            "stage": stage,
            "feedback": feedback,
            "feedback_rules": feedback_messages,
            "sway": round(sway, 5),
            "rep_event": rep_event,
        }

    def finalize(self) -> dict:
        self.session.end_session()
        return self.session.summary()

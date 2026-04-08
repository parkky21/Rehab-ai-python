"""
Exercise Base Class
Each exercise inherits from this and provides:
- FSM rep detection logic
- Exercise-specific scoring config (target ROM, ideal tempo, weights, etc.)
- Feedback rules
"""

import numpy as np
from pipeline.scorer import ExerciseConfig, RepScorer
from pipeline.feature_engine import (
    calculate_angle_2d,
    ROMTracker,
    TempoTracker,
)


class ExerciseBase:
    """
    Base class for all exercises.
    Subclasses must implement:
        - process(landmarks) -> (counter, stage, feedback, render_data)
    And should set self.config to an ExerciseConfig instance.
    """

    def __init__(self):
        self.counter = 0
        self.stage = None
        self.feedback = ""
        self.relevant_landmarks = []

        # Scoring Config (override in subclasses)
        self.config = ExerciseConfig()
        self.scorer = RepScorer(self.config)

        # Feature trackers
        self.rom_tracker = ROMTracker()
        self.tempo_tracker = TempoTracker()

        # Latest rep scores
        self.last_rep_scores = None
        self.rep_completed = False  # Flag for the UI to detect new rep

    def reset(self):
        self.counter = 0
        self.stage = None
        self.feedback = ""
        self.last_rep_scores = None
        self.rep_completed = False
        self.rom_tracker.reset()
        self.tempo_tracker.reset()
        
        # Reset ML buffers if they exist
        if hasattr(self, 'ml_lstm'):
            self.ml_lstm.reset_buffer()
        if hasattr(self, 'ml_transformer'):
            self.ml_transformer.reset_buffer()

    def _get_or_create_ml_scorers(self):
        if not hasattr(self, 'ml_lstm'):
            try:
                from ml_scoring.ml_scorer import MLRepScorer
                # Map class name to an ID (0-6) based on training data
                name_map = {
                    "Squats": 0, "SitToStand": 1, "HeelRaises": 2,
                    "StandingHipAbduction": 3, "Marching": 4,
                    "StandingHipExtension": 5, "LegRaises": 6
                }
                ex_id = name_map.get(self.__class__.__name__, 0)
                self.ml_lstm = MLRepScorer("lstm", exercise_id=ex_id)
                self.ml_transformer = MLRepScorer("transformer", exercise_id=ex_id)
            except ImportError:
                self.ml_lstm = None
                self.ml_transformer = None
        return self.ml_lstm, self.ml_transformer

    def record_ml_frame(self, angle, landmarks):
        """Called by subclasses per frame to feed the ML models."""
        lstm, trans = self._get_or_create_ml_scorers()
        if lstm is None or trans is None:
            return

        hip_x = (landmarks[23].x + landmarks[24].x) / 2.0
        
        # Approximate rep_progress (very rough heuristic since we're streaming realtime)
        # 0 at start, 0.5 at target_rom, 1.0 returning
        target = self.config.target_rom
        rom_so_far = self.rom_tracker.current_max - self.rom_tracker.current_min if self.rom_tracker.current_max > -float('inf') else 0
        
        # We just pass 0.0 for streaming, the model relies more on the sequence trajectory anyway
        rep_prog = 0.0 

        lstm.record_frame(angle=angle, hip_x=hip_x, rep_progress=rep_prog)
        trans.record_frame(angle=angle, hip_x=hip_x, rep_progress=rep_prog)

    def _on_rep_complete(self, sway: float = 0.0):
        """
        Called internally when a rep completes.
        Computes scores for the completed rep.
        """
        rom = self.rom_tracker.complete_rep()
        rep_time = self.tempo_tracker.complete_rep()

        self.last_rep_scores = self.scorer.score_rep(
            user_rom=rom,
            sway=sway,
            rep_time=rep_time,
        )
        self.last_rep_scores["rom_value"] = round(rom, 2)
        self.last_rep_scores["rep_time"] = round(rep_time, 3)
        
        # Score ML models
        lstm, trans = self._get_or_create_ml_scorers()
        if lstm and trans:
            lstm_scores = lstm.score_rep(user_rom=rom, sway=sway, rep_time=rep_time)
            trans_scores = trans.score_rep(user_rom=rom, sway=sway, rep_time=rep_time)
            self.last_rep_scores["lstm_final"] = lstm_scores["final_score"]
            self.last_rep_scores["transformer_final"] = trans_scores["final_score"]

        self.rep_completed = True

    def _on_rep_start(self):
        """Called when a new rep movement begins."""
        if self.tempo_tracker.rep_start_time is None:
            self.tempo_tracker.start_rep()

    def process(self, landmarks):
        """
        Process landmarks for a single frame.
        Returns (counter, stage, feedback, render_data).
        """
        raise NotImplementedError("Subclasses must implement process()")

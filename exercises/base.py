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

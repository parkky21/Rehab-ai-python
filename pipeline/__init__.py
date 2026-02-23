"""
Pipeline Package
Exports all pipeline modules for easy import.
"""

from .landmark_processor import process_landmarks, ProcessedLandmark
from .smoother import EMALandmarkSmoother
from .feature_engine import (
    calculate_angle_3d,
    calculate_angle_2d,
    ROMTracker,
    VelocityTracker,
    SwayTracker,
    TempoTracker,
)
from .scorer import ExerciseConfig, RepScorer, compute_final_score
from .feedback import FeedbackEngine, FeedbackRule, create_default_feedback_engine
from .session import Session, RepRecord
from .progression import ProgressionState

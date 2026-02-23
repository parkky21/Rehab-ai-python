"""
Landmark Processor Module
- Visibility filtering
- Hip-center coordinate normalization
- Torso-length scale normalization (device independence)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


VISIBILITY_THRESHOLD = 0.5


@dataclass
class ProcessedLandmark:
    """A single processed landmark with normalized coordinates."""
    x: float
    y: float
    z: float
    visibility: float
    valid: bool = True


def _lm_to_array(lm):
    """Convert a mediapipe landmark to numpy array [x, y, z]."""
    return np.array([lm.x, lm.y, lm.z])


def filter_visibility(landmarks, threshold=VISIBILITY_THRESHOLD):
    """
    Returns a list of ProcessedLandmark with validity flags.
    Landmarks below the visibility threshold are marked invalid.
    """
    processed = []
    for lm in landmarks:
        vis = getattr(lm, 'visibility', 1.0)
        valid = vis >= threshold
        processed.append(ProcessedLandmark(
            x=lm.x, y=lm.y, z=lm.z,
            visibility=vis, valid=valid
        ))
    return processed


def compute_hip_center(landmarks):
    """Compute the midpoint between left hip (23) and right hip (24)."""
    left_hip = _lm_to_array(landmarks[23])
    right_hip = _lm_to_array(landmarks[24])
    return (left_hip + right_hip) / 2.0


def compute_mid_shoulder(landmarks):
    """Compute the midpoint between left shoulder (11) and right shoulder (12)."""
    left_shoulder = _lm_to_array(landmarks[11])
    right_shoulder = _lm_to_array(landmarks[12])
    return (left_shoulder + right_shoulder) / 2.0


def normalize_landmarks(processed_landmarks, hip_center, torso_length):
    """
    Translate to hip-center origin and scale by torso length.
    Returns a new list of ProcessedLandmark with normalized coordinates.
    """
    if torso_length < 0.01:
        torso_length = 0.01  # prevent division by zero

    normalized = []
    for lm in processed_landmarks:
        if lm.valid:
            normalized.append(ProcessedLandmark(
                x=(lm.x - hip_center[0]) / torso_length,
                y=(lm.y - hip_center[1]) / torso_length,
                z=(lm.z - hip_center[2]) / torso_length,
                visibility=lm.visibility,
                valid=True
            ))
        else:
            normalized.append(ProcessedLandmark(
                x=lm.x, y=lm.y, z=lm.z,
                visibility=lm.visibility, valid=False
            ))
    return normalized


def process_landmarks(raw_landmarks):
    """
    Full landmark processing pipeline:
    1. Visibility filtering
    2. Compute hip center + torso length
    3. Normalize coordinates
    
    Returns (processed_landmarks, hip_center, torso_length)
    """
    # Step 1: Visibility filter
    processed = filter_visibility(raw_landmarks)

    # Step 2: Compute reference points (use raw landmarks for reference calc)
    hip_center = compute_hip_center(raw_landmarks)
    mid_shoulder = compute_mid_shoulder(raw_landmarks)
    torso_length = float(np.linalg.norm(mid_shoulder - hip_center))

    # Step 3: Normalize
    normalized = normalize_landmarks(processed, hip_center, torso_length)

    return normalized, hip_center, torso_length

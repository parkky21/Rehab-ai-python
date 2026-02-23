"""
Landmark Smoother Module
Exponential Moving Average (EMA) per-joint, per-coordinate.
Reduces noise from raw MediaPipe pose output.
"""

import numpy as np
from typing import Optional


class EMALandmarkSmoother:
    """
    Applies Exponential Moving Average smoothing to each landmark coordinate.
    
    S_t = alpha * X_t + (1 - alpha) * S_(t-1)
    
    Args:
        alpha: Smoothing factor (0-1). Lower = smoother but more lag.
               Recommended: 0.2-0.3 for elderly/slow rehab, 0.3-0.4 for normal.
        num_landmarks: Number of landmarks to track (33 for MediaPipe Pose).
    """

    def __init__(self, alpha: float = 0.3, num_landmarks: int = 33):
        self.alpha = alpha
        self.num_landmarks = num_landmarks
        self.state: Optional[np.ndarray] = None  # Shape: (num_landmarks, 3) for x,y,z

    def reset(self):
        """Reset smoothing state."""
        self.state = None

    def smooth(self, landmarks):
        """
        Apply EMA smoothing to a list of landmarks.
        
        Args:
            landmarks: List of landmark objects with .x, .y, .z attributes.
                       Can be raw MediaPipe landmarks or ProcessedLandmark.
        
        Returns:
            List of landmarks with smoothed coordinates (same type as input).
        """
        current = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

        if self.state is None:
            self.state = current.copy()
        else:
            self.state = self.alpha * current + (1.0 - self.alpha) * self.state

        # Write smoothed values back onto the landmark objects
        for i, lm in enumerate(landmarks):
            lm.x = float(self.state[i, 0])
            lm.y = float(self.state[i, 1])
            lm.z = float(self.state[i, 2])

        return landmarks

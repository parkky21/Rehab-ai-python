"""
Feature Engineering Module
Converts smoothed/normalized landmarks into biomechanical metrics:
- Joint angle calculation (3D cosine rule)
- Range of Motion (ROM) per rep
- Joint velocity
- Hip sway (stability metric)
"""

import numpy as np
from collections import deque
import time


def calculate_angle_3d(a, b, c):
    """
    Calculate angle at vertex B formed by points A-B-C using the cosine rule.
    Works with any object that has .x, .y, .z attributes.
    
    Returns angle in degrees (0-180).
    """
    ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
    bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])

    mag_ba = np.linalg.norm(ba)
    mag_bc = np.linalg.norm(bc)

    if mag_ba < 1e-6 or mag_bc < 1e-6:
        return 0.0

    cosine = np.dot(ba, bc) / (mag_ba * mag_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))
    return float(angle)


def calculate_angle_2d(a, b, c):
    """
    Calculate angle at vertex B formed by points A-B-C using atan2 (2D, x/y only).
    Backward compatible with original exercise logic.
    """
    a_arr = np.array([a.x, a.y])
    b_arr = np.array([b.x, b.y])
    c_arr = np.array([c.x, c.y])

    radians = np.arctan2(c_arr[1] - b_arr[1], c_arr[0] - b_arr[0]) - \
              np.arctan2(a_arr[1] - b_arr[1], a_arr[0] - b_arr[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360.0 - angle

    return float(angle)


class ROMTracker:
    """
    Tracks Range of Motion per rep.
    Call update() every frame with the current primary angle.
    Call complete_rep() when a rep finishes to get the ROM for that rep.
    """

    def __init__(self):
        self.current_max = -float('inf')
        self.current_min = float('inf')
        self.rep_roms = []  # ROM per completed rep

    def reset(self):
        self.current_max = -float('inf')
        self.current_min = float('inf')
        self.rep_roms = []

    def update(self, angle: float):
        """Update with current frame's angle."""
        self.current_max = max(self.current_max, angle)
        self.current_min = min(self.current_min, angle)

    def complete_rep(self) -> float:
        """Called when a rep is completed. Returns the ROM for this rep and resets."""
        rom = self.current_max - self.current_min
        if rom < 0:
            rom = 0.0
        self.rep_roms.append(rom)
        # Reset for next rep
        self.current_max = -float('inf')
        self.current_min = float('inf')
        return rom

    @property
    def average_rom(self) -> float:
        if not self.rep_roms:
            return 0.0
        return sum(self.rep_roms) / len(self.rep_roms)


class VelocityTracker:
    """
    Tracks velocity of a specific joint between frames.
    """

    def __init__(self):
        self.prev_position = None
        self.prev_time = None
        self.current_velocity = 0.0

    def reset(self):
        self.prev_position = None
        self.prev_time = None
        self.current_velocity = 0.0

    def update(self, landmark, current_time: float = None):
        """
        Update with a new landmark position.
        Returns the current velocity magnitude.
        """
        if current_time is None:
            current_time = time.time()

        position = np.array([landmark.x, landmark.y, landmark.z])

        if self.prev_position is not None and self.prev_time is not None:
            dt = current_time - self.prev_time
            if dt > 1e-6:
                delta = position - self.prev_position
                self.current_velocity = float(np.linalg.norm(delta) / dt)
            else:
                self.current_velocity = 0.0
        else:
            self.current_velocity = 0.0

        self.prev_position = position
        self.prev_time = current_time

        return self.current_velocity


class SwayTracker:
    """
    Measures horizontal hip sway (stability) over a sliding window.
    Higher sway = lower stability.
    """

    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.hip_x_history = deque(maxlen=window_size)
        self.current_sway = 0.0

    def reset(self):
        self.hip_x_history.clear()
        self.current_sway = 0.0

    def update(self, hip_center_x: float) -> float:
        """
        Update with current hip center X position.
        Returns the current sway (standard deviation of recent hip_x values).
        """
        self.hip_x_history.append(hip_center_x)

        if len(self.hip_x_history) >= 5:
            self.current_sway = float(np.std(list(self.hip_x_history)))
        else:
            self.current_sway = 0.0

        return self.current_sway


class TempoTracker:
    """
    Tracks the time taken per rep for tempo scoring.
    """

    def __init__(self):
        self.rep_start_time = None
        self.rep_times = []  # seconds per completed rep

    def reset(self):
        self.rep_start_time = None
        self.rep_times = []

    def start_rep(self):
        """Call when a new rep movement begins."""
        self.rep_start_time = time.time()

    def complete_rep(self) -> float:
        """Call when a rep completes. Returns the time for this rep in seconds."""
        if self.rep_start_time is None:
            return 0.0
        rep_time = time.time() - self.rep_start_time
        self.rep_times.append(rep_time)
        self.rep_start_time = time.time()  # Next rep starts immediately
        return rep_time

    @property
    def average_tempo(self) -> float:
        if not self.rep_times:
            return 0.0
        return sum(self.rep_times) / len(self.rep_times)

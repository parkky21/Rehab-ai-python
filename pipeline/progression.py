"""
Adaptive Progression Engine
Adjusts exercise difficulty after each session based on performance trends.
"""

import json
import os
from typing import Optional


class ProgressionState:
    """Persistent state for tracking progression across sessions."""

    def __init__(self):
        self.session_scores = []  # list of avg_final_score per session
        self.target_reps = 10
        self.target_rom_multiplier = 1.0  # applied to exercise config target_rom
        self.sway_tolerance_multiplier = 1.0  # applied to exercise config acceptable_sway

    def record_session(self, avg_score: float):
        """Record a session's average score."""
        self.session_scores.append(avg_score)

    @property
    def consecutive_good_sessions(self) -> int:
        """Count of recent consecutive sessions scoring > 85."""
        count = 0
        for score in reversed(self.session_scores):
            if score > 85:
                count += 1
            else:
                break
        return count

    def compute_progression(self) -> dict:
        """
        Evaluate if difficulty should change.
        Returns a dict describing the adjustment.
        """
        if not self.session_scores:
            return {"action": "none", "reason": "No sessions recorded"}

        latest = self.session_scores[-1]

        # Upgrade: avg > 85 for 3 consecutive sessions
        if self.consecutive_good_sessions >= 3:
            self.target_reps = min(self.target_reps + 2, 30)
            self.target_rom_multiplier = min(self.target_rom_multiplier + 0.05, 1.5)
            self.sway_tolerance_multiplier = max(self.sway_tolerance_multiplier - 0.1, 0.5)
            return {
                "action": "upgrade",
                "reason": f"3+ sessions scoring >85 (latest: {latest:.0f})",
                "new_target_reps": self.target_reps,
                "rom_multiplier": round(self.target_rom_multiplier, 2),
                "sway_multiplier": round(self.sway_tolerance_multiplier, 2),
            }

        # Regress: avg < 60
        if latest < 60:
            self.target_reps = max(self.target_reps - 2, 5)
            self.target_rom_multiplier = max(self.target_rom_multiplier - 0.05, 0.7)
            self.sway_tolerance_multiplier = min(self.sway_tolerance_multiplier + 0.1, 2.0)
            return {
                "action": "regress",
                "reason": f"Latest session scored {latest:.0f} (<60)",
                "new_target_reps": self.target_reps,
                "rom_multiplier": round(self.target_rom_multiplier, 2),
                "sway_multiplier": round(self.sway_tolerance_multiplier, 2),
            }

        return {
            "action": "maintain",
            "reason": f"Latest score {latest:.0f} — within normal range",
        }

    def save(self, filepath: str):
        """Persist progression state to disk."""
        data = {
            "session_scores": self.session_scores,
            "target_reps": self.target_reps,
            "target_rom_multiplier": self.target_rom_multiplier,
            "sway_tolerance_multiplier": self.sway_tolerance_multiplier,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """Load progression state from disk."""
        if not os.path.exists(filepath):
            return
        with open(filepath, "r") as f:
            data = json.load(f)
        self.session_scores = data.get("session_scores", [])
        self.target_reps = data.get("target_reps", 10)
        self.target_rom_multiplier = data.get("target_rom_multiplier", 1.0)
        self.sway_tolerance_multiplier = data.get("sway_tolerance_multiplier", 1.0)

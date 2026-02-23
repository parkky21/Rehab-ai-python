"""
Session Module
Accumulates per-rep data into a session summary.
Supports JSON export for persistence.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RepRecord:
    """Data for a single completed rep."""
    rep_number: int
    rom_score: float = 0.0
    stability_score: float = 0.0
    tempo_score: float = 0.0
    asymmetry_score: float = 0.0
    final_score: float = 0.0
    rom_value: float = 0.0
    rep_time: float = 0.0
    feedback: list = field(default_factory=list)


@dataclass
class Session:
    """Accumulates rep data for a full exercise session."""
    exercise_name: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    reps: list[RepRecord] = field(default_factory=list)
    feedback_events: list[str] = field(default_factory=list)

    def add_rep(self, scores: dict, rom_value: float = 0.0, rep_time: float = 0.0, feedback: list = None):
        """Add a completed rep to the session."""
        rep = RepRecord(
            rep_number=len(self.reps) + 1,
            rom_score=scores.get("rom_score", 0),
            stability_score=scores.get("stability_score", 0),
            tempo_score=scores.get("tempo_score", 0),
            asymmetry_score=scores.get("asymmetry_score", 0),
            final_score=scores.get("final_score", 0),
            rom_value=rom_value,
            rep_time=rep_time,
            feedback=feedback or [],
        )
        self.reps.append(rep)
        if feedback:
            self.feedback_events.extend(feedback)

    def end_session(self):
        """Mark session as ended."""
        self.end_time = time.time()

    @property
    def total_reps(self) -> int:
        return len(self.reps)

    @property
    def avg_final_score(self) -> float:
        if not self.reps:
            return 0.0
        return sum(r.final_score for r in self.reps) / len(self.reps)

    @property
    def avg_rom_score(self) -> float:
        if not self.reps:
            return 0.0
        return sum(r.rom_score for r in self.reps) / len(self.reps)

    @property
    def avg_stability_score(self) -> float:
        if not self.reps:
            return 0.0
        return sum(r.stability_score for r in self.reps) / len(self.reps)

    @property
    def avg_tempo_score(self) -> float:
        if not self.reps:
            return 0.0
        return sum(r.tempo_score for r in self.reps) / len(self.reps)

    @property
    def avg_asymmetry_score(self) -> float:
        if not self.reps:
            return 0.0
        return sum(r.asymmetry_score for r in self.reps) / len(self.reps)

    def summary(self) -> dict:
        """Return a complete session summary."""
        return {
            "exercise": self.exercise_name,
            "total_reps": self.total_reps,
            "avg_final_score": round(self.avg_final_score, 1),
            "avg_rom_score": round(self.avg_rom_score, 1),
            "avg_stability_score": round(self.avg_stability_score, 1),
            "avg_tempo_score": round(self.avg_tempo_score, 1),
            "avg_asymmetry_score": round(self.avg_asymmetry_score, 1),
            "duration_seconds": round((self.end_time or time.time()) - self.start_time, 1),
            "feedback_events": list(set(self.feedback_events)),
        }

    def to_json(self) -> str:
        """Export session to JSON string."""
        data = self.summary()
        data["reps"] = [
            {
                "rep": r.rep_number,
                "rom_score": r.rom_score,
                "stability_score": r.stability_score,
                "tempo_score": r.tempo_score,
                "asymmetry_score": r.asymmetry_score,
                "final_score": r.final_score,
                "rom_value": round(r.rom_value, 1),
                "rep_time": round(r.rep_time, 2),
                "feedback": r.feedback,
            }
            for r in self.reps
        ]
        return json.dumps(data, indent=2)

    def save(self, filepath: str):
        """Save session data to a JSON file."""
        with open(filepath, "w") as f:
            f.write(self.to_json())

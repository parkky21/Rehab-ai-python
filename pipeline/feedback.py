"""
Feedback Engine Module
Rule-based per-frame and per-rep feedback generation.
Evaluates biomechanical conditions and returns prioritized corrections.
"""


class FeedbackRule:
    """A single feedback rule."""

    def __init__(self, name: str, condition_fn, message: str, priority: int = 5):
        """
        Args:
            name: Rule identifier.
            condition_fn: Callable(landmarks, context) -> bool.
            message: Feedback string to display when triggered.
            priority: Lower number = higher priority.
        """
        self.name = name
        self.condition_fn = condition_fn
        self.message = message
        self.priority = priority


class FeedbackEngine:
    """
    Evaluates a list of rules per frame, returns the highest-priority triggered feedback.
    """

    def __init__(self):
        self.rules: list[FeedbackRule] = []

    def add_rule(self, rule: FeedbackRule):
        self.rules.append(rule)
        self.rules.sort(key=lambda r: r.priority)

    def evaluate(self, landmarks, context: dict) -> list[str]:
        """
        Evaluate all rules against current frame landmarks and context.
        
        Args:
            landmarks: List of landmark objects (raw or processed).
            context: Dict with keys like 'primary_angle', 'sway', 'rep_time', etc.
        
        Returns:
            List of triggered feedback messages, ordered by priority.
        """
        triggered = []
        for rule in self.rules:
            try:
                if rule.condition_fn(landmarks, context):
                    triggered.append(rule.message)
            except Exception:
                pass
        return triggered


# ============== Common Rehab Feedback Rules ==============

def _knee_valgus_check(landmarks, context):
    """Check if left knee collapses inward (valgus)."""
    knee = landmarks[25]
    ankle = landmarks[27]
    return knee.x < ankle.x - 0.02


def _excessive_forward_lean(landmarks, context):
    """Check if torso is leaning too far forward."""
    import numpy as np
    shoulder = landmarks[11]
    hip = landmarks[23]
    
    # Compute torso angle from vertical
    dx = shoulder.x - hip.x
    dy = shoulder.y - hip.y
    angle_from_vertical = abs(np.degrees(np.arctan2(dx, -dy)))
    return angle_from_vertical > 25.0


def _left_right_asymmetry(landmarks, context):
    """Check if left-right angle asymmetry exceeds 15 degrees."""
    asymmetry = context.get("asymmetry_value", 0)
    return asymmetry > 15.0


def _poor_depth(landmarks, context):
    """Check if ROM is significantly below target."""
    rom = context.get("current_rom", 0)
    target = context.get("target_rom", 90)
    return rom > 0 and rom < target * 0.6


def _too_fast(landmarks, context):
    """Check if rep tempo is too fast."""
    rep_time = context.get("rep_time", 0)
    ideal = context.get("ideal_rep_time", 3.0)
    return rep_time > 0 and rep_time < ideal * 0.5


# Pre-built common rules
COMMON_RULES = [
    FeedbackRule("knee_valgus", _knee_valgus_check, "Keep knees aligned with toes", priority=1),
    FeedbackRule("forward_lean", _excessive_forward_lean, "Keep chest upright", priority=2),
    FeedbackRule("asymmetry", _left_right_asymmetry, "Distribute weight evenly", priority=3),
    FeedbackRule("poor_depth", _poor_depth, "Try to go deeper for full range", priority=4),
    FeedbackRule("too_fast", _too_fast, "Slow down for controlled tempo", priority=5),
]


def create_default_feedback_engine() -> FeedbackEngine:
    """Create a FeedbackEngine pre-loaded with common rehab rules."""
    engine = FeedbackEngine()
    for rule in COMMON_RULES:
        engine.add_rule(rule)
    return engine

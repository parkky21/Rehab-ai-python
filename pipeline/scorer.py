"""
Scoring Module
Computes per-rep scores for ROM, stability, tempo, and asymmetry.
Produces a weighted final score per rep.
"""


class ExerciseConfig:
    """
    Per-exercise scoring configuration.
    Each exercise should provide its own instance with tailored values.
    """

    def __init__(
        self,
        target_rom: float = 90.0,
        ideal_rep_time: float = 3.0,
        acceptable_sway: float = 0.02,
        stability_factor: float = 100.0,
        tempo_penalty_factor: float = 20.0,
        asymmetry_penalty_factor: float = 5.0,
        weight_rom: float = 0.4,
        weight_stability: float = 0.3,
        weight_tempo: float = 0.3,
    ):
        self.target_rom = target_rom
        self.ideal_rep_time = ideal_rep_time
        self.acceptable_sway = acceptable_sway
        self.stability_factor = stability_factor
        self.tempo_penalty_factor = tempo_penalty_factor
        self.asymmetry_penalty_factor = asymmetry_penalty_factor
        self.weight_rom = weight_rom
        self.weight_stability = weight_stability
        self.weight_tempo = weight_tempo


def _clamp(value, lo=0.0, hi=100.0):
    return max(lo, min(hi, value))


def compute_rom_score(user_rom: float, target_rom: float) -> float:
    """ROM_score = min((user_ROM / target_ROM) * 100, 100)"""
    if target_rom <= 0:
        return 100.0
    return _clamp((user_rom / target_rom) * 100.0)


def compute_stability_score(sway: float, acceptable_sway: float, stability_factor: float) -> float:
    """stability_score = 100 - (normalized_sway * stability_factor)"""
    if acceptable_sway <= 0:
        normalized = 0.0
    else:
        normalized = sway / acceptable_sway
    return _clamp(100.0 - (normalized * stability_factor))


def compute_tempo_score(rep_time: float, ideal_rep_time: float, penalty_factor: float) -> float:
    """
    Asymmetric tempo scoring for rehab:
    - Going too FAST is penalized 2x harder (unsafe, uncontrolled)
    - Going too SLOW is penalized less (controlled movement is fine)
    """
    diff = rep_time - ideal_rep_time
    if diff < 0:
        # Too fast — double penalty
        return _clamp(100.0 - abs(diff) * penalty_factor * 2.0)
    else:
        # Too slow — gentle penalty
        return _clamp(100.0 - abs(diff) * penalty_factor * 0.5)


def compute_asymmetry_score(left_angle: float, right_angle: float, penalty_factor: float) -> float:
    """asymmetry_score = 100 - (|left - right| * penalty)"""
    asymmetry = abs(left_angle - right_angle)
    return _clamp(100.0 - asymmetry * penalty_factor)


def compute_final_score(
    rom_score: float,
    stability_score: float,
    tempo_score: float,
    config: ExerciseConfig
) -> float:
    """
    Weighted final score per rep.
    final_score = w_rom * ROM + w_stability * stability + w_tempo * tempo
    """
    score = (
        config.weight_rom * rom_score +
        config.weight_stability * stability_score +
        config.weight_tempo * tempo_score
    )
    return _clamp(score)


class RepScorer:
    """
    Convenience class that computes all scores for a single completed rep.
    """

    def __init__(self, config: ExerciseConfig):
        self.config = config

    def score_rep(
        self,
        user_rom: float,
        sway: float,
        rep_time: float,
        left_angle: float = None,
        right_angle: float = None
    ) -> dict:
        """
        Score a completed rep, returns a dict with all component scores.
        """
        rom_score = compute_rom_score(user_rom, self.config.target_rom)
        stability_score = compute_stability_score(
            sway, self.config.acceptable_sway, self.config.stability_factor
        )
        tempo_score = compute_tempo_score(
            rep_time, self.config.ideal_rep_time, self.config.tempo_penalty_factor
        )

        asymmetry_score = 100.0
        if left_angle is not None and right_angle is not None:
            asymmetry_score = compute_asymmetry_score(
                left_angle, right_angle, self.config.asymmetry_penalty_factor
            )

        final_score = compute_final_score(rom_score, stability_score, tempo_score, self.config)

        return {
            "rom_score": round(rom_score, 1),
            "stability_score": round(stability_score, 1),
            "tempo_score": round(tempo_score, 1),
            "asymmetry_score": round(asymmetry_score, 1),
            "final_score": round(final_score, 1),
        }

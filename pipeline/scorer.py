"""
Scoring Module
==============

Computes per-rep scores for ROM, stability, tempo, and asymmetry.
Produces a weighted final score per rep.

HOW TO TWEAK SCORING
--------------------
Each exercise creates an ExerciseConfig instance. To adjust scoring behavior,
modify the config in the exercise file (e.g., exercises/squats.py).

Example:
    self.config = ExerciseConfig(
        target_rom=70.0,       # The "ideal" ROM in degrees
        ideal_rep_time=4.0,    # How many seconds a perfect rep should take
        acceptable_sway=0.015, # Maximum hip sway before penalty kicks in
        ...
    )
"""


class ExerciseConfig:
    """
    Per-exercise scoring configuration.
    Each exercise should provide its own instance with tailored values.

    Parameters
    ----------
    target_rom : float, default=90.0
        The ideal Range of Motion (in degrees or proxy units).
        A user who achieves this ROM or higher gets 100% ROM score.
        LOWER this to make the exercise easier (less movement needed for full score).
        RAISE this to demand more range of motion.

    ideal_rep_time : float, default=3.0
        The ideal time (in seconds) for one full repetition.
        Reps matching this time get a perfect tempo score.
        For slow rehab exercises, set to 4.0–6.0.
        For dynamic exercises, set to 2.0–3.0.

    acceptable_sway : float, default=0.02
        Maximum horizontal hip sway (normalized 0–1) before stability penalty.
        LOWER this to penalize body sway more harshly.
        RAISE this to be more lenient with balance.
        Typical: 0.01 (strict) to 0.03 (lenient).

    stability_factor : float, default=100.0
        Multiplier controlling how harshly sway is penalized.
        Formula: stability_score = 100 - (sway / acceptable_sway) * stability_factor
        RAISE to make stability scoring stricter.
        LOWER to be more lenient.

    tempo_penalty_factor : float, default=20.0
        Base penalty per second of deviation from ideal rep time.
        NOTE: This is applied asymmetrically:
          - Too FAST: penalty = deviation * factor * 2.0 (harsh)
          - Too SLOW: penalty = deviation * factor * 0.5 (gentle)
        RAISE for stricter tempo requirements.
        LOWER for more lenient tempo scoring.

    asymmetry_penalty_factor : float, default=5.0
        Penalty per degree of left-right angle difference.
        Formula: asymmetry_score = 100 - |left - right| * factor
        RAISE to penalize uneven movement more.
        Only used when both left and right angles are provided.

    weight_rom : float, default=0.4
        Weight of ROM score in the final weighted score (0.0–1.0).
        For strength exercises, use higher (0.4–0.5).
        For balance exercises, use lower (0.25–0.35).
        All weights should sum to 1.0.

    weight_stability : float, default=0.3
        Weight of stability score in the final weighted score (0.0–1.0).
        For balance exercises like hip abduction, use higher (0.35–0.45).
        For upper body exercises, use lower (0.2–0.25).

    weight_tempo : float, default=0.3
        Weight of tempo score in the final weighted score (0.0–1.0).
        For controlled-tempo exercises like heel raises, use higher (0.35–0.4).
        For strength exercises, use lower (0.2–0.25).
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
    """Clamp a value between lo and hi."""
    return max(lo, min(hi, value))


def compute_rom_score(user_rom: float, target_rom: float) -> float:
    """
    Compute ROM (Range of Motion) score.

    Formula: ROM_score = min((user_ROM / target_ROM) * 100, 100)

    Parameters
    ----------
    user_rom : float
        The ROM the user achieved during this rep (max_angle - min_angle).
    target_rom : float
        The ideal ROM defined in ExerciseConfig.target_rom.

    Returns
    -------
    float
        Score between 0 and 100. Higher = closer to target ROM.
        A user achieving 50% of target ROM gets a score of 50.
    """
    if target_rom <= 0:
        return 100.0
    return _clamp((user_rom / target_rom) * 100.0)


def compute_stability_score(sway: float, acceptable_sway: float, stability_factor: float) -> float:
    """
    Compute stability score based on horizontal hip sway.

    Formula: stability_score = 100 - (sway / acceptable_sway) * stability_factor

    Parameters
    ----------
    sway : float
        Standard deviation of hip_x over the rep window (from SwayTracker).
    acceptable_sway : float
        ExerciseConfig.acceptable_sway — the max sway before penalty.
    stability_factor : float
        ExerciseConfig.stability_factor — how harshly to penalize.

    Returns
    -------
    float
        Score between 0 and 100. Higher = more stable.
        If sway == acceptable_sway, score = 100 - stability_factor (typically 0).
    """
    if acceptable_sway <= 0:
        normalized = 0.0
    else:
        normalized = sway / acceptable_sway
    return _clamp(100.0 - (normalized * stability_factor))


def compute_tempo_score(rep_time: float, ideal_rep_time: float, penalty_factor: float) -> float:
    """
    Compute tempo score with ASYMMETRIC penalties.

    For rehab, slow movement is safer than fast movement, so:
    - Too FAST: penalty = |deviation| * penalty_factor * 2.0
    - Too SLOW: penalty = |deviation| * penalty_factor * 0.5

    Parameters
    ----------
    rep_time : float
        Actual time (seconds) the user took for this rep.
    ideal_rep_time : float
        ExerciseConfig.ideal_rep_time — the target rep duration.
    penalty_factor : float
        ExerciseConfig.tempo_penalty_factor — base penalty per second of deviation.

    Returns
    -------
    float
        Score between 0 and 100. Higher = closer to ideal tempo.

    Examples
    --------
    With ideal_rep_time=4.0 and penalty_factor=20.0:
      - rep_time=4.0 → score=100 (perfect)
      - rep_time=1.0 → score=100 - 3*20*2 = -20 → clamped to 0 (way too fast)
      - rep_time=6.0 → score=100 - 2*20*0.5 = 80 (slightly slow, gentle penalty)
      - rep_time=8.0 → score=100 - 4*20*0.5 = 60 (slow but still decent)
    """
    diff = rep_time - ideal_rep_time
    if diff < 0:
        # Too fast — double penalty
        return _clamp(100.0 - abs(diff) * penalty_factor * 2.0)
    else:
        # Too slow — gentle penalty
        return _clamp(100.0 - abs(diff) * penalty_factor * 0.5)


def compute_asymmetry_score(left_angle: float, right_angle: float, penalty_factor: float) -> float:
    """
    Compute left-right asymmetry score.

    Formula: asymmetry_score = 100 - |left_angle - right_angle| * penalty_factor

    Parameters
    ----------
    left_angle : float
        Joint angle on the left side (degrees).
    right_angle : float
        Joint angle on the right side (degrees).
    penalty_factor : float
        ExerciseConfig.asymmetry_penalty_factor.

    Returns
    -------
    float
        Score between 0 and 100. Higher = more symmetrical.
    """
    asymmetry = abs(left_angle - right_angle)
    return _clamp(100.0 - asymmetry * penalty_factor)


def compute_final_score(
    rom_score: float,
    stability_score: float,
    tempo_score: float,
    config: ExerciseConfig
) -> float:
    """
    Compute the weighted final score from component scores.

    Formula: final = w_rom * ROM + w_stability * stability + w_tempo * tempo

    Parameters
    ----------
    rom_score : float
        ROM component score (0–100).
    stability_score : float
        Stability component score (0–100).
    tempo_score : float
        Tempo component score (0–100).
    config : ExerciseConfig
        Holds the weight values (weight_rom, weight_stability, weight_tempo).

    Returns
    -------
    float
        Final weighted score between 0 and 100.
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

    Usage
    -----
    scorer = RepScorer(config)
    scores = scorer.score_rep(user_rom=45.0, sway=0.01, rep_time=3.5)
    print(scores)
    # {'rom_score': 64.3, 'stability_score': 95.0, 'tempo_score': 95.0,
    #  'asymmetry_score': 100.0, 'final_score': 82.1}
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
        Score a completed rep.

        Parameters
        ----------
        user_rom : float
            The ROM (max_angle - min_angle) achieved during this rep.
        sway : float
            Current hip sway value from SwayTracker.
        rep_time : float
            Time in seconds this rep took.
        left_angle : float, optional
            Left-side joint angle for asymmetry scoring.
        right_angle : float, optional
            Right-side joint angle for asymmetry scoring.

        Returns
        -------
        dict
            Keys: rom_score, stability_score, tempo_score, asymmetry_score, final_score.
            All values are floats rounded to 1 decimal, range 0–100.
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

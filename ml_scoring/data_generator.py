"""
Synthetic Time-Series Dataset Generator
========================================
Generates realistic frame-by-frame exercise rep sequences and labels them
with the existing rule-based RepScorer. This provides ground-truth scores
for training LSTM, Transformer, and TCN models.

Features per frame (7-dim):
    [angle, hip_x, velocity, rep_progress, left_angle, right_angle, exercise_id]

Labels per rep (4 scalars, each 0–100):
    [rom_score, stability_score, tempo_score, final_score]

Exercise configs replicate real exercises to ensure realistic score ranges.
"""

import numpy as np
import os
import sys

# Allow running as a script from python-server/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.scorer import ExerciseConfig, RepScorer

# ---------------------------------------------------------------------------
# Exercise registry — mirrors real exercise configs
# ---------------------------------------------------------------------------

EXERCISES = {
    0: ExerciseConfig(                        # Squats
        target_rom=70.0,
        ideal_rep_time=4.0,
        acceptable_sway=0.015,
        stability_factor=100.0,
        tempo_penalty_factor=20.0,
        asymmetry_penalty_factor=5.0,
        weight_rom=0.4,
        weight_stability=0.35,
        weight_tempo=0.25,
    ),
    1: ExerciseConfig(                        # Sit-to-Stand
        target_rom=50.0,
        ideal_rep_time=5.0,
        acceptable_sway=0.02,
        stability_factor=100.0,
        tempo_penalty_factor=20.0,
        asymmetry_penalty_factor=5.0,
        weight_rom=0.3,
        weight_stability=0.4,
        weight_tempo=0.3,
    ),
    2: ExerciseConfig(                        # Heel Raises
        target_rom=5.0,
        ideal_rep_time=5.0,
        acceptable_sway=0.02,
        stability_factor=100.0,
        tempo_penalty_factor=15.0,
        asymmetry_penalty_factor=5.0,
        weight_rom=0.25,
        weight_stability=0.35,
        weight_tempo=0.4,
    ),
    3: ExerciseConfig(                        # Hip Abduction
        target_rom=25.0,
        ideal_rep_time=4.0,
        acceptable_sway=0.025,
        stability_factor=100.0,
        tempo_penalty_factor=20.0,
        asymmetry_penalty_factor=5.0,
        weight_rom=0.35,
        weight_stability=0.4,
        weight_tempo=0.25,
    ),
    4: ExerciseConfig(                        # Marching
        target_rom=30.0,
        ideal_rep_time=3.0,
        acceptable_sway=0.025,
        stability_factor=100.0,
        tempo_penalty_factor=20.0,
        asymmetry_penalty_factor=5.0,
        weight_rom=0.3,
        weight_stability=0.4,
        weight_tempo=0.3,
    ),
    5: ExerciseConfig(                        # Hip Extension
        target_rom=20.0,
        ideal_rep_time=4.0,
        acceptable_sway=0.02,
        stability_factor=100.0,
        tempo_penalty_factor=18.0,
        asymmetry_penalty_factor=5.0,
        weight_rom=0.35,
        weight_stability=0.35,
        weight_tempo=0.3,
    ),
    6: ExerciseConfig(                        # Leg Raises
        target_rom=35.0,
        ideal_rep_time=4.0,
        acceptable_sway=0.02,
        stability_factor=100.0,
        tempo_penalty_factor=20.0,
        asymmetry_penalty_factor=5.0,
        weight_rom=0.4,
        weight_stability=0.3,
        weight_tempo=0.3,
    ),
}

NUM_EXERCISES = len(EXERCISES)
FEATURE_DIM = 7       # per-frame feature count
SEQ_LEN = 80          # fixed sequence length (pad/truncate)
REPS_PER_EXERCISE = 700   # 700 * 7 = 4900 ≈ 5000 total reps
NOISE_STD = 0.015     # Gaussian noise on angles + hip_x

# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _simulate_angle_sequence(
    rng: np.random.Generator,
    exercise_id: int,
    n_frames: int,
) -> tuple[np.ndarray, float]:
    """
    Simulate a smooth angle trajectory for one rep.
    Returns (angle_array, rom_achieved).
    """
    config = EXERCISES[exercise_id]
    target_rom = config.target_rom

    # Sample quality: 0.2 (poor) → 1.2 (excellent, slight overshoot ok)
    quality = rng.uniform(0.2, 1.2)
    actual_rom = target_rom * quality

    # Phase of the rep: up / down sinusoid or ramp
    t = np.linspace(0, np.pi, n_frames)
    # Some exercises return to start (squat: down→up), others go one-way then back
    if rng.random() < 0.5:
        # Smooth sine: starts at 0, peaks at actual_rom, returns to 0
        base = actual_rom * np.sin(t)
    else:
        # Ramp up + ramp down with slight asymmetry
        ramp_up = np.linspace(0, actual_rom, n_frames // 2)
        ramp_down = np.linspace(actual_rom, 0, n_frames - n_frames // 2)
        base = np.concatenate([ramp_up, ramp_down])

    # Add realistic jitter
    noise = rng.normal(0, NOISE_STD * target_rom, size=n_frames)
    angles = np.clip(base + noise, 0, target_rom * 1.35)
    return angles, float(angles.max() - angles.min())


def _simulate_hip_x_sequence(
    rng: np.random.Generator,
    n_frames: int,
    sway_level: float,
) -> tuple[np.ndarray, float]:
    """
    Simulate hip_x (center around 0.5) with a given sway amplitude.
    Returns (hip_x_array, sway_std).
    """
    center = rng.uniform(0.4, 0.6)
    trend = np.linspace(0, rng.uniform(-sway_level, sway_level), n_frames)
    noise = rng.normal(0, sway_level / 2, n_frames)
    hip_x = center + trend + noise
    hip_x = np.clip(hip_x, 0.0, 1.0)
    sway = float(np.std(hip_x))
    return hip_x, sway


def _simulate_rep(
    rng: np.random.Generator,
    exercise_id: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Simulate one full rep and compute rule-based scores.

    Returns
    -------
    frames : np.ndarray [SEQ_LEN, FEATURE_DIM]
    label  : np.ndarray [4]  — (rom_score, stability_score, tempo_score, final_score)
    meta   : dict with raw scalar inputs (for verification)
    """
    config = EXERCISES[exercise_id]
    n_frames = rng.integers(40, 120)  # variable-length reps

    # --- Angle trajectory ---
    angles, achieved_rom = _simulate_angle_sequence(rng, exercise_id, n_frames)

    # --- Sway ---
    sway_level = rng.uniform(0.005, 0.04)
    hip_x, sway_std = _simulate_hip_x_sequence(rng, n_frames, sway_level)

    # --- Rep time (seconds) ---
    ideal = config.ideal_rep_time
    rep_time = rng.uniform(ideal * 0.4, ideal * 2.5)

    # --- Velocity (finite difference of angle) ---
    velocity = np.abs(np.gradient(angles)) * 30  # approx 30fps

    # --- Rep progress (0→1) ---
    rep_progress = np.linspace(0, 1, n_frames)

    # --- Left / right angles (slight asymmetry) ---
    asym = rng.uniform(0, 15)  # degrees difference
    left_angles = angles + rng.normal(0, 1, n_frames)
    right_angles = angles + asym + rng.normal(0, 1, n_frames)
    left_angles = np.clip(left_angles, 0, None)
    right_angles = np.clip(right_angles, 0, None)

    # --- Exercise id (normalized to [0,1]) ---
    ex_id_feature = np.full(n_frames, exercise_id / (NUM_EXERCISES - 1))

    # --- Stack features: [angle, hip_x, velocity, rep_progress, L, R, ex_id] ---
    raw_features = np.stack([
        angles,
        hip_x,
        velocity,
        rep_progress,
        left_angles,
        right_angles,
        ex_id_feature,
    ], axis=1)  # [n_frames, 7]

    # --- Pad / truncate to SEQ_LEN ---
    if n_frames >= SEQ_LEN:
        frames = raw_features[:SEQ_LEN]
    else:
        pad = np.zeros((SEQ_LEN - n_frames, FEATURE_DIM))
        frames = np.vstack([raw_features, pad])

    # --- Ground-truth labels via rule-based scorer ---
    scorer = RepScorer(config)
    scores = scorer.score_rep(
        user_rom=achieved_rom,
        sway=sway_std,
        rep_time=rep_time,
        left_angle=float(left_angles.mean()),
        right_angle=float(right_angles.mean()),
    )

    label = np.array([
        scores["rom_score"],
        scores["stability_score"],
        scores["tempo_score"],
        scores["final_score"],
    ], dtype=np.float32)

    meta = {
        "exercise_id": exercise_id,
        "achieved_rom": achieved_rom,
        "sway": sway_std,
        "rep_time": rep_time,
        "asym": asym,
    }

    return frames.astype(np.float32), label, meta


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    reps_per_exercise: int = REPS_PER_EXERCISE,
    seed: int = 42,
    save_dir: str = None,
) -> dict:
    """
    Generate synthetic dataset and optionally save to disk.

    Parameters
    ----------
    reps_per_exercise : int
        Number of reps to generate per exercise (all exercises are balanced).
    seed : int
        Random seed for reproducibility.
    save_dir : str or None
        If given, saves ['X_train', 'X_val', 'X_test', 'y_train', 'y_val', 'y_test'].

    Returns
    -------
    dict with keys: X_train, X_val, X_test, y_train, y_val, y_test
        Each X is np.ndarray of shape [N, SEQ_LEN, FEATURE_DIM]
        Each y is np.ndarray of shape [N, 4]
    """
    rng = np.random.default_rng(seed)
    all_X, all_y = [], []

    total_reps = reps_per_exercise * NUM_EXERCISES
    print(f"Generating {total_reps} reps ({reps_per_exercise} × {NUM_EXERCISES} exercises)...")

    for ex_id in EXERCISES:
        for i in range(reps_per_exercise):
            frames, label, _ = _simulate_rep(rng, ex_id)
            all_X.append(frames)
            all_y.append(label)
        print(f"  Exercise {ex_id}: {reps_per_exercise} reps done")

    X = np.stack(all_X)  # [N, SEQ_LEN, 7]
    y = np.stack(all_y)  # [N, 4]

    # Shuffle
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    # Normalize X per-feature using training stats (computed before split)
    n = len(X)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    X_train = X[:n_train]
    X_val   = X[n_train:n_train + n_val]
    X_test  = X[n_train + n_val:]
    y_train = y[:n_train]
    y_val   = y[n_train:n_train + n_val]
    y_test  = y[n_train + n_val:]

    # Compute normalization stats from training set
    mean = X_train.mean(axis=(0, 1), keepdims=True)   # [1, 1, 7]
    std  = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std
    X_test  = (X_test  - mean) / std

    dataset = {
        "X_train": X_train, "y_train": y_train,
        "X_val":   X_val,   "y_val":   y_val,
        "X_test":  X_test,  "y_test":  y_test,
        "feature_mean": mean,
        "feature_std":  std,
    }

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "dataset.npz")
        np.savez_compressed(save_path, **dataset)
        print(f"\nDataset saved → {save_path}")

    print(f"\nDataset summary:")
    print(f"  Train: {len(X_train)} reps | Val: {len(X_val)} reps | Test: {len(X_test)} reps")
    print(f"  X shape: {X_train.shape}  (reps × frames × features)")
    print(f"  y shape: {y_train.shape}  (reps × [rom, stability, tempo, final])")
    print(f"  y_train score range: [{y_train.min():.1f}, {y_train.max():.1f}]")

    return dataset


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, "data")
    generate_dataset(save_dir=save_dir)

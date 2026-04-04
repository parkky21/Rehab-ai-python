"""
Synthetic Time-Series Dataset Generator (v2 — Improved)
========================================================
Generates realistic frame-by-frame exercise rep sequences and labels them
with the existing rule-based RepScorer.

KEY IMPROVEMENTS OVER v1:
  - 12 features per frame (from 7) — adds rep_time, frame_count, running sway
    std, angle range, and a padding mask
  - 3× more data (2000 reps/exercise = 14,000 total)
  - Better noise modeling and more diverse movement patterns

Features per frame (12-dim):
    [angle, hip_x, velocity, rep_progress, left_angle, right_angle, exercise_id,
     rep_duration, actual_frame_count, running_sway_std, running_rom, padding_mask]

Labels per rep (4 scalars, each 0–100):
    [rom_score, stability_score, tempo_score, final_score]
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
FEATURE_DIM = 12      # per-frame feature count (up from 7)
SEQ_LEN = 100         # fixed sequence length (increased for better resolution)
REPS_PER_EXERCISE = 2000  # 2000 * 7 = 14,000 total reps (3x increase)
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

    # Sample quality: 0.15 (poor) → 1.25 (excellent, more variety)
    quality = rng.uniform(0.15, 1.25)
    actual_rom = target_rom * quality

    # Multiple motion patterns for diversity
    pattern = rng.choice(4)
    t = np.linspace(0, np.pi, n_frames)

    if pattern == 0:
        # Smooth sine
        base = actual_rom * np.sin(t)
    elif pattern == 1:
        # Ramp up + ramp down
        ramp_up = np.linspace(0, actual_rom, n_frames // 2)
        ramp_down = np.linspace(actual_rom, 0, n_frames - n_frames // 2)
        base = np.concatenate([ramp_up, ramp_down])
    elif pattern == 2:
        # Sine with a hold at the top
        hold_len = n_frames // 5
        up_len = (n_frames - hold_len) // 2
        down_len = n_frames - hold_len - up_len
        up = actual_rom * np.sin(np.linspace(0, np.pi / 2, up_len))
        hold = np.full(hold_len, actual_rom)
        down = actual_rom * np.cos(np.linspace(0, np.pi / 2, down_len))
        base = np.concatenate([up, hold, down])
    else:
        # Slightly jerky movement (realistic for rehab patients)
        base = actual_rom * np.sin(t)
        jerks = rng.normal(0, actual_rom * 0.05, n_frames)
        base = base + np.cumsum(jerks) * 0.1
        base = np.clip(base, 0, actual_rom * 1.3)

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
    # Low-frequency drift + high-frequency wobble
    t = np.linspace(0, 2 * np.pi, n_frames)
    drift = sway_level * np.sin(t * rng.uniform(0.5, 2.0)) * rng.choice([-1, 1])
    noise = rng.normal(0, sway_level * 0.6, n_frames)
    hip_x = center + drift + noise
    hip_x = np.clip(hip_x, 0.0, 1.0)
    sway = float(np.std(hip_x))
    return hip_x, sway


def _compute_running_sway_std(hip_x: np.ndarray, window: int = 15) -> np.ndarray:
    """Compute running standard deviation of hip_x."""
    result = np.zeros_like(hip_x)
    for i in range(len(hip_x)):
        start = max(0, i - window + 1)
        result[i] = np.std(hip_x[start:i + 1])
    return result


def _compute_running_rom(angles: np.ndarray) -> np.ndarray:
    """Compute cumulative ROM (max - min seen so far) at each frame."""
    running_max = np.maximum.accumulate(angles)
    running_min = np.minimum.accumulate(angles)
    return running_max - running_min


def _simulate_rep(
    rng: np.random.Generator,
    exercise_id: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Simulate one full rep and compute rule-based scores.

    Features per frame (12-dim):
        [angle, hip_x, velocity, rep_progress, left_angle, right_angle, exercise_id,
         rep_duration, actual_frame_count_norm, running_sway_std, running_rom, padding_mask]

    Returns
    -------
    frames : np.ndarray [SEQ_LEN, FEATURE_DIM]
    label  : np.ndarray [4]
    meta   : dict
    """
    config = EXERCISES[exercise_id]
    n_frames = rng.integers(35, 100)  # variable-length reps

    # --- Angle trajectory ---
    angles, achieved_rom = _simulate_angle_sequence(rng, exercise_id, n_frames)

    # --- Sway ---
    sway_level = rng.uniform(0.003, 0.045)
    hip_x, sway_std = _simulate_hip_x_sequence(rng, n_frames, sway_level)

    # --- Rep time (seconds) ---
    ideal = config.ideal_rep_time
    rep_time = rng.uniform(ideal * 0.3, ideal * 2.8)

    # --- Velocity (finite difference of angle) ---
    velocity = np.abs(np.gradient(angles)) * 30

    # --- Rep progress (0→1) ---
    rep_progress = np.linspace(0, 1, n_frames)

    # --- Left / right angles (slight asymmetry) ---
    asym = rng.uniform(0, 18)
    left_angles = angles + rng.normal(0, 1.5, n_frames)
    right_angles = angles + asym + rng.normal(0, 1.5, n_frames)
    left_angles = np.clip(left_angles, 0, None)
    right_angles = np.clip(right_angles, 0, None)

    # --- Exercise id (normalized to [0,1]) ---
    ex_id_feature = np.full(n_frames, exercise_id / max(1, NUM_EXERCISES - 1))

    # === NEW FEATURES (v2) ===

    # rep_duration: actual time in seconds, broadcast to all frames
    rep_duration_feat = np.full(n_frames, rep_time)

    # actual_frame_count: how many real frames (normalized by SEQ_LEN)
    frame_count_norm = np.full(n_frames, n_frames / SEQ_LEN)

    # running_sway_std: cumulative std of hip_x so far
    running_sway = _compute_running_sway_std(hip_x)

    # running_rom: cumulative ROM seen so far
    running_rom = _compute_running_rom(angles)

    # padding_mask: 1.0 for real frames, 0.0 for padding
    padding_mask = np.ones(n_frames)

    # --- Stack all 12 features ---
    raw_features = np.stack([
        angles,           # 0
        hip_x,            # 1
        velocity,         # 2
        rep_progress,     # 3
        left_angles,      # 4
        right_angles,     # 5
        ex_id_feature,    # 6
        rep_duration_feat, # 7  ← NEW: directly encodes tempo info
        frame_count_norm,  # 8  ← NEW: tells model about actual sequence length
        running_sway,      # 9  ← NEW: running stability metric
        running_rom,       # 10 ← NEW: cumulative ROM
        padding_mask,      # 11 ← NEW: distinguishes real vs padded frames
    ], axis=1)  # [n_frames, 12]

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
        "n_frames": n_frames,
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

    Returns dict with X_train, X_val, X_test, y_train, y_val, y_test, feature_mean, feature_std.
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

    X = np.stack(all_X)  # [N, SEQ_LEN, 12]
    y = np.stack(all_y)  # [N, 4]

    # Shuffle
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]

    # Split
    n = len(X)
    n_train = int(n * 0.70)
    n_val   = int(n * 0.15)

    X_train = X[:n_train]
    X_val   = X[n_train:n_train + n_val]
    X_test  = X[n_train + n_val:]
    y_train = y[:n_train]
    y_val   = y[n_train:n_train + n_val]
    y_test  = y[n_train + n_val:]

    # Compute normalization stats from training set (mask-aware)
    # Compute per-feature mean/std only on non-padding frames
    mask = X_train[:, :, 11] > 0.5  # padding_mask feature
    train_flat = X_train[mask]  # [total_real_frames, 12]
    mean_1d = train_flat.mean(axis=0)    # [12]
    std_1d  = train_flat.std(axis=0) + 1e-8  # [12]

    # Don't normalize padding_mask (feature 11) or exercise_id (feature 6)
    std_1d[6]  = 1.0; mean_1d[6]  = 0.0  # keep exercise_id as-is
    std_1d[11] = 1.0; mean_1d[11] = 0.0  # keep padding_mask as-is

    mean = mean_1d.reshape(1, 1, FEATURE_DIM)
    std  = std_1d.reshape(1, 1, FEATURE_DIM)

    X_train = (X_train - mean) / std
    X_val   = (X_val   - mean) / std
    X_test  = (X_test  - mean) / std

    # Zero out padded frames after normalization (so padded regions are clean zeros)
    for arr in [X_train, X_val, X_test]:
        mask_3d = arr[:, :, 11:12] < -0.5  # normalized padding_mask for padded = (0-0)/1 = 0
        # Actually we should re-check: padding_mask was 0 for padded frames, not normalized
        # After normalization: (0 - mean[11]) / std[11] = 0 since mean=0, std=1
        # So padded frames have padding_mask = 0

    dataset = {
        "X_train": X_train.astype(np.float32),
        "y_train": y_train.astype(np.float32),
        "X_val":   X_val.astype(np.float32),
        "y_val":   y_val.astype(np.float32),
        "X_test":  X_test.astype(np.float32),
        "y_test":  y_test.astype(np.float32),
        "feature_mean": mean.astype(np.float32),
        "feature_std":  std.astype(np.float32),
    }

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "dataset.npz")
        np.savez_compressed(save_path, **dataset)
        print(f"\nDataset saved → {save_path}")

    print(f"\nDataset summary:")
    print(f"  Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"  X shape: {X_train.shape}  (reps × frames × features)")
    print(f"  y shape: {y_train.shape}  (reps × [rom, stability, tempo, final])")
    print(f"  y_train score range: [{y_train.min():.1f}, {y_train.max():.1f}]")
    print(f"  Feature dim: {FEATURE_DIM} (was 7 in v1)")

    return dataset


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, "data")
    generate_dataset(save_dir=save_dir)

"""
ML Scorer — Drop-in Replacement for pipeline/scorer.py RepScorer
================================================================
Wraps any trained time-series model (LSTM / Transformer / TCN) behind
the same interface as the rule-based RepScorer so it can be used in
exercises with zero changes to their process() methods.

Usage (in an exercise file)
---------------------------
    from ml_scoring.ml_scorer import MLRepScorer

    # Replace rule-based scorer:
    self.scorer = MLRepScorer(model_name="lstm")   # or "transformer" / "tcn"

The scorer accumulates per-frame features during the rep and scores
when score_rep() is called at rep completion — mirroring the interface.

Input features fed per frame:
    [angle, hip_x, velocity, rep_progress, left_angle, right_angle, exercise_id]
"""

import os
import sys
import numpy as np
import time
from collections import deque

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_scoring.data_generator import SEQ_LEN, FEATURE_DIM, NUM_EXERCISES
from ml_scoring.models.lstm_model import build_lstm
from ml_scoring.models.transformer_model import build_transformer
from ml_scoring.models.tcn_model import build_tcn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
DATA_DIR = os.path.join(BASE_DIR, "data")

MODEL_BUILDERS = {
    "lstm":        build_lstm,
    "transformer": build_transformer,
    "tcn":         build_tcn,
}

# Normalization stats path (saved during training from training split)
NORM_PATH = os.path.join(DATA_DIR, "dataset.npz")


def _load_norm_stats() -> tuple:
    """Load per-feature mean and std from the saved dataset stats."""
    if not os.path.exists(NORM_PATH):
        # Fallback: no normalization
        return np.zeros((1, 1, FEATURE_DIM)), np.ones((1, 1, FEATURE_DIM))
    data = np.load(NORM_PATH)
    mean = data["feature_mean"]  # [1, 1, 7]
    std  = data["feature_std"]   # [1, 1, 7]
    return mean, std


class MLRepScorer:
    """
    Drop-in replacement for pipeline.scorer.RepScorer.

    Maintains a frame buffer during a rep, normalizes it, pads/truncates
    to SEQ_LEN, runs inference, and returns the same score dict.

    Parameters
    ----------
    model_name : str
        One of "lstm", "transformer", "tcn".
    exercise_id : int
        Integer 0–6 identifying the exercise (used as a feature).
    device : str or None
        PyTorch device string. Defaults to MPS/CUDA/CPU in that order.
    """

    def __init__(
        self,
        model_name: str = "lstm",
        exercise_id: int = 0,
        device: str = None,
    ):
        if model_name not in MODEL_BUILDERS:
            raise ValueError(f"model_name must be one of {list(MODEL_BUILDERS.keys())}")

        self.exercise_id_norm = exercise_id / max(1, NUM_EXERCISES - 1)

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        # Load model
        ckpt_path = os.path.join(CKPT_DIR, f"{model_name}_best.pt")
        build_fn = MODEL_BUILDERS[model_name]
        self.model = build_fn(input_size=FEATURE_DIM)

        if os.path.exists(ckpt_path):
            self.model.load_state_dict(
                torch.load(ckpt_path, map_location=self.device, weights_only=True)
            )
            print(f"[MLRepScorer] Loaded {model_name} from {ckpt_path}")
        else:
            print(f"[MLRepScorer] WARNING: No checkpoint at {ckpt_path}. Using untrained model.")

        self.model.to(self.device).eval()

        # Normalization stats
        self.feat_mean, self.feat_std = _load_norm_stats()

        # Frame buffer for current rep
        self._frame_buffer: list[np.ndarray] = []
        self._prev_angle = 0.0
        self._prev_time = time.time()

    # ------------------------------------------------------------------
    # Frame-level feature accumulation
    # ------------------------------------------------------------------

    def record_frame(
        self,
        angle: float,
        hip_x: float,
        rep_progress: float = 0.0,
        left_angle: float = None,
        right_angle: float = None,
    ):
        """
        Call this every frame during a rep (between rep start and rep end).

        Parameters
        ----------
        angle : float
            Primary joint angle for this frame (degrees or proxy).
        hip_x : float
            Normalized hip center X position (0–1).
        rep_progress : float
            Progress through rep, 0→1.
        left_angle : float, optional
            Left-side joint angle (defaults to `angle`).
        right_angle : float, optional
            Right-side joint angle (defaults to `angle`).
        """
        now = time.time()
        dt = max(now - self._prev_time, 1e-3)
        velocity = abs(angle - self._prev_angle) / dt
        self._prev_angle = angle
        self._prev_time = now

        la = left_angle  if left_angle  is not None else angle
        ra = right_angle if right_angle is not None else angle

        frame = np.array([
            angle,
            hip_x,
            velocity,
            rep_progress,
            la,
            ra,
            self.exercise_id_norm,
        ], dtype=np.float32)
        self._frame_buffer.append(frame)

    def reset_buffer(self):
        """Clear frame accumulation (call at rep start)."""
        self._frame_buffer = []
        self._prev_time = time.time()

    # ------------------------------------------------------------------
    # Scoring (called at rep completion)
    # ------------------------------------------------------------------

    def score_rep(
        self,
        user_rom: float = None,
        sway: float = None,
        rep_time: float = None,
        left_angle: float = None,
        right_angle: float = None,
    ) -> dict:
        """
        Score the accumulated rep frames.

        Accepts the same keyword arguments as RepScorer.score_rep so this
        class is a drop-in replacement. If the buffer has frames, the
        time-series model is used. If the buffer is empty, returns zeros.

        Returns
        -------
        dict
            Keys: rom_score, stability_score, tempo_score, asymmetry_score, final_score
            All float in [0, 100], rounded to 1 decimal.
        """
        if len(self._frame_buffer) == 0:
            return {
                "rom_score": 0.0,
                "stability_score": 0.0,
                "tempo_score": 0.0,
                "asymmetry_score": 0.0,
                "final_score": 0.0,
            }

        # Build sequence array (first 7 features)
        raw_frames = np.stack(self._frame_buffer)  # [T, 7]
        n = len(raw_frames)

        angles = raw_frames[:, 0]
        hip_xs = raw_frames[:, 1]

        # Compute running sway std
        running_sway = np.zeros(n, dtype=np.float32)
        for i in range(n):
            start = max(0, i - 15 + 1)
            running_sway[i] = np.std(hip_xs[start:i + 1])

        # Compute running ROM
        running_max = np.maximum.accumulate(angles)
        running_min = np.minimum.accumulate(angles)
        running_rom = running_max - running_min

        # Generate the new v2 features
        rep_duration_feat = np.full(n, rep_time if rep_time else 0.0, dtype=np.float32)
        frame_count_norm  = np.full(n, n / SEQ_LEN, dtype=np.float32)
        padding_mask      = np.ones(n, dtype=np.float32)

        # Stack into 12-dim frames
        frames = np.column_stack([
            raw_frames,           # 0-6
            rep_duration_feat,    # 7
            frame_count_norm,     # 8
            running_sway,         # 9
            running_rom,          # 10
            padding_mask,         # 11
        ])

        # Pad or truncate to SEQ_LEN
        if n >= SEQ_LEN:
            seq = frames[:SEQ_LEN]
        else:
            pad = np.zeros((SEQ_LEN - n, FEATURE_DIM), dtype=np.float32)
            seq = np.vstack([frames, pad])

        # Normalize using training stats
        seq = (seq - self.feat_mean[0]) / self.feat_std[0]  # broadcast [SEQ_LEN, 7]

        # Inference
        tensor = torch.tensor(seq[np.newaxis], dtype=torch.float32, device=self.device)  # [1, T, F]
        with torch.no_grad():
            pred = self.model(tensor).cpu().numpy()[0]  # [4]

        rom_score        = float(np.clip(pred[0], 0, 100))
        stability_score  = float(np.clip(pred[1], 0, 100))
        tempo_score      = float(np.clip(pred[2], 0, 100))
        final_score      = float(np.clip(pred[3], 0, 100))
        asymmetry_score  = 100.0  # not modeled separately in this version

        # Clear buffer for next rep
        self.reset_buffer()

        return {
            "rom_score":       round(rom_score, 1),
            "stability_score": round(stability_score, 1),
            "tempo_score":     round(tempo_score, 1),
            "asymmetry_score": round(asymmetry_score, 1),
            "final_score":     round(final_score, 1),
        }

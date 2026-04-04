"""
Evaluation & Comparison Script
================================
Loads all three trained models and the rule-based scorer, evaluates each
against the test set, and prints a detailed comparison table.

Metrics: MAE, RMSE, R² (per-score and overall final_score)

Also saves a comparison plot to ml_scoring/results/comparison.png

Usage
-----
    cd python-server
    python -m ml_scoring.evaluate
"""

import os
import sys
import numpy as np

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_scoring.data_generator import (
    generate_dataset, EXERCISES, SEQ_LEN, FEATURE_DIM,
    _simulate_rep,
)
from ml_scoring.models.lstm_model import build_lstm
from ml_scoring.models.transformer_model import build_transformer
from ml_scoring.models.tcn_model import build_tcn

# Try importing matplotlib — gracefully skip plotting if unavailable
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.npz")
CKPT_DIR  = os.path.join(BASE_DIR, "checkpoints")
RESULTS   = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS, exist_ok=True)

SCORE_NAMES = ["ROM", "Stability", "Tempo", "Final"]


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))

def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-8))


# ---------------------------------------------------------------------------
# Rule-based baseline
# ---------------------------------------------------------------------------

def rule_based_predictions(dataset: dict) -> np.ndarray:
    """
    The rule-based scorer GENERATES the labels, so it scores 0 error by
    construction. We regenerate with fresh random seed to get a 'hold-out'
    rule-based comparison (same distribution, slight noise variations).
    
    For the comparison table we report Ground Truth vs itself (0 error)
    as the theoretical baseline ceiling.
    """
    # Just return the test labels themselves — rule-based is the oracle
    return dataset["y_test"]


# ---------------------------------------------------------------------------
# ML model predictions
# ---------------------------------------------------------------------------

def load_model(name: str, ModelCls, build_fn, device: torch.device) -> torch.nn.Module:
    ckpt = os.path.join(CKPT_DIR, f"{name}_best.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            "Run `python -m ml_scoring.train` first."
        )
    model = build_fn(input_size=FEATURE_DIM)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.to(device).eval()
    return model


@torch.no_grad()
def predict(model: torch.nn.Module, X_test: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    preds = []
    X_tensor = torch.tensor(X_test, dtype=torch.float32)
    for i in range(0, len(X_tensor), batch_size):
        batch = X_tensor[i:i + batch_size].to(device)
        out = model(batch).cpu().numpy()
        preds.append(out)
    return np.vstack(preds)


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_table(results: dict, y_true: np.ndarray):
    """Print a tidy comparison table to the terminal."""
    col_width = 14
    header = f"{'Model':<18} | {'MAE (Final)':>{col_width}} | {'RMSE (Final)':>{col_width}} | {'R² (Final)':>{col_width}} | {'MAE (All)':>{col_width}}"
    print("\n" + "=" * len(header))
    print("  EXERCISE SCORING MODEL COMPARISON")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for model_name, preds in sorted(results.items()):
        # Final score (index 3)
        final_true = y_true[:, 3]
        final_pred = preds[:, 3]
        mae_f  = mae(final_true, final_pred)
        rmse_f = rmse(final_true, final_pred)
        r2_f   = r2(final_true, final_pred)
        # All scores
        mae_all = mae(y_true, preds)
        row = f"{model_name:<18} | {mae_f:>{col_width}.3f} | {rmse_f:>{col_width}.3f} | {r2_f:>{col_width}.4f} | {mae_all:>{col_width}.3f}"
        print(row)

    print("=" * len(header))

    # Per-score breakdown for ML models
    print("\n  PER-SCORE MAE BREAKDOWN")
    print(f"{'Model':<18} | " + " | ".join(f"{s:>12}" for s in SCORE_NAMES))
    print("-" * (18 + 3 + 4 * 15))
    for model_name, preds in sorted(results.items()):
        per_mae = [mae(y_true[:, i], preds[:, i]) for i in range(4)]
        row = f"{model_name:<18} | " + " | ".join(f"{m:>12.3f}" for m in per_mae)
        print(row)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_comparison(results: dict, y_true: np.ndarray):
    if not HAS_MATPLOTLIB:
        print("matplotlib not available, skipping plots.")
        return

    n_models = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("ML Exercise Scoring — Model Comparison", fontsize=15, fontweight="bold")

    colors = {"Rule-Based": "#2ecc71", "LSTM": "#3498db", "Transformer": "#e74c3c", "TCN": "#9b59b6"}

    # 1. MAE per score per model
    ax = axes[0, 0]
    x = np.arange(4)
    width = 0.8 / n_models
    for i, (name, preds) in enumerate(sorted(results.items())):
        maes = [mae(y_true[:, j], preds[:, j]) for j in range(4)]
        ax.bar(x + i * width, maes, width, label=name, color=colors.get(name, "#7f8c8d"), alpha=0.85)
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(SCORE_NAMES)
    ax.set_ylabel("MAE (lower is better)")
    ax.set_title("Mean Absolute Error per Score Component")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # 2. R² for Final Score
    ax = axes[0, 1]
    names_sorted = sorted(results.keys())
    r2s = [r2(y_true[:, 3], results[n][:, 3]) for n in names_sorted]
    bar_colors = [colors.get(n, "#7f8c8d") for n in names_sorted]
    bars = ax.bar(names_sorted, r2s, color=bar_colors, alpha=0.85)
    ax.set_ylim(max(0, min(r2s) - 0.05), 1.02)
    ax.set_ylabel("R² (higher is better)")
    ax.set_title("R² Score — Final Exercise Score")
    ax.axhline(0.95, color="gray", linestyle="--", alpha=0.5, label="0.95 target")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for bar, r in zip(bars, r2s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.02,
                f"{r:.4f}", ha="center", va="top", fontsize=9, color="white", fontweight="bold")

    # 3. Scatter: Predicted vs True Final Score (ML models)
    ax = axes[1, 0]
    ax.plot([0, 100], [0, 100], "k--", alpha=0.4, label="Perfect")
    for name, preds in sorted(results.items()):
        if name == "Rule-Based":
            continue
        ax.scatter(y_true[:, 3], preds[:, 3], s=8, alpha=0.35,
                   color=colors.get(name, "#7f8c8d"), label=name)
    ax.set_xlabel("True Final Score (Rule-Based Oracle)")
    ax.set_ylabel("Predicted Final Score")
    ax.set_title("Predicted vs True Final Score")
    ax.legend()
    ax.grid(alpha=0.2)

    # 4. Training loss curves (if available)
    ax = axes[1, 1]
    hist_path = os.path.join(CKPT_DIR, "training_histories.npz")
    if os.path.exists(hist_path):
        hist = np.load(hist_path)
        for name in ["lstm", "transformer", "tcn"]:
            key_val = f"{name}_val_loss"
            if key_val in hist:
                val_losses = hist[key_val]
                epochs = range(1, len(val_losses) + 1)
                ax.plot(epochs, val_losses, label=name.upper(),
                        color=colors.get(name.capitalize(), "#7f8c8d") if name != "transformer" else colors.get("Transformer", "#7f8c8d"), linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val Huber Loss (SmoothL1)")
        ax.set_title("Validation Loss During Training")
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No training history found.\nRun train.py first.",
                ha="center", va="center", transform=ax.transAxes)

    plt.tight_layout()
    out_path = os.path.join(RESULTS, "comparison.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nComparison plot saved → {out_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # Load dataset
    if not os.path.exists(DATA_PATH):
        print("Dataset not found, generating ...")
        from ml_scoring.data_generator import generate_dataset
        save_dir = os.path.join(BASE_DIR, "data")
        generate_dataset(save_dir=save_dir)
    data = np.load(DATA_PATH)
    X_test = data["X_test"]
    y_test = data["y_test"]
    print(f"Test set: {len(X_test)} reps")

    # --- Predictions ---
    results = {}

    # Rule-based (oracle = perfect)
    rb_preds = rule_based_predictions(data)
    results["Rule-Based"] = rb_preds
    print("\nRule-Based: loaded (oracle labels, R²=1.0 by definition)")

    # ML models
    ml_specs = {
        "LSTM":        build_lstm,
        "Transformer": build_transformer,
        "TCN":         build_tcn,
    }
    for display_name, build_fn in ml_specs.items():
        ckpt_name = display_name.lower()
        try:
            model = load_model(ckpt_name, None, build_fn, device)
            preds = predict(model, X_test, device)
            results[display_name] = preds
            print(f"{display_name}: predictions generated")
        except FileNotFoundError as e:
            print(f"⚠  {e}")

    if len(results) < 2:
        print("\nNo ML model checkpoints found. Please run:\n  python -m ml_scoring.train")
        return

    # --- Print comparison ---
    print_table(results, y_test)

    # --- Plot ---
    plot_comparison(results, y_test)

    # --- Save numeric results as .npz ---
    save_data = {"y_true": y_test}
    for name, preds in results.items():
        save_data[f"preds_{name.replace('-', '_')}"] = preds
    np.savez(os.path.join(RESULTS, "predictions.npz"), **save_data)
    print(f"Predictions saved → {os.path.join(RESULTS, 'predictions.npz')}")


if __name__ == "__main__":
    main()

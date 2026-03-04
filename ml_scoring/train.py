"""
Training Script — ML Exercise Scoring Models
=============================================
Trains LSTM, Transformer, and TCN models on the synthetic dataset.

Usage
-----
    cd python-server
    python -m ml_scoring.train

Optional flags (via env vars or edit below):
    EPOCHS   = 80   (default)
    BATCH    = 64   (default)
    LR       = 1e-3 (default)
    PATIENCE = 12   (early stopping)
"""

import os
import sys
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Allow running as a module OR as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_scoring.data_generator import generate_dataset, SEQ_LEN, FEATURE_DIM
from ml_scoring.models.lstm_model import build_lstm
from ml_scoring.models.transformer_model import build_transformer
from ml_scoring.models.tcn_model import build_tcn

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
EPOCHS   = int(os.environ.get("EPOCHS",   80))
BATCH    = int(os.environ.get("BATCH",    64))
LR       = float(os.environ.get("LR",     1e-3))
PATIENCE = int(os.environ.get("PATIENCE", 12))

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE_DIR, "data", "dataset.npz")
CKPT_DIR   = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CKPT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_or_generate() -> dict:
    """Load dataset from disk if available, otherwise generate it."""
    if os.path.exists(DATA_PATH):
        print(f"Loading existing dataset from {DATA_PATH} ...")
        data = np.load(DATA_PATH)
        return {k: data[k] for k in data.files}
    print("No dataset found — generating now ...")
    save_dir = os.path.join(BASE_DIR, "data")
    return generate_dataset(save_dir=save_dir)


def to_tensors(dataset: dict, device: torch.device):
    """Convert numpy arrays to torch tensors on the given device."""
    def t(arr):
        return torch.tensor(arr, dtype=torch.float32, device=device)
    return (
        t(dataset["X_train"]), t(dataset["y_train"]),
        t(dataset["X_val"]),   t(dataset["y_val"]),
        t(dataset["X_test"]),  t(dataset["y_test"]),
    )


def make_loaders(X_train, y_train, X_val, y_val, batch_size: int):
    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val,   y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Train one model
# ---------------------------------------------------------------------------

def train_model(
    name: str,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = EPOCHS,
    lr: float = LR,
    patience: int = PATIENCE,
) -> dict:
    """
    Train the given model with MSE loss + Adam + cosine LR schedule.

    Returns history dict with train_losses and val_losses lists.
    """
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    history = {"train_losses": [], "val_losses": []}

    ckpt_path = os.path.join(CKPT_DIR, f"{name}_best.pt")
    print(f"\n{'='*55}")
    print(f"  Training {name}  |  {sum(p.numel() for p in model.parameters()):,} params")
    print(f"{'='*55}")
    print(f"{'Epoch':>6} | {'Train MSE':>10} | {'Val MSE':>10} | {'LR':>10}")
    print("-" * 45)

    t_start = time.time()
    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ---- Validate ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_losses.append(loss.item())

        train_mse = np.mean(train_losses)
        val_mse   = np.mean(val_losses)
        current_lr = optimizer.param_groups[0]["lr"]
        history["train_losses"].append(train_mse)
        history["val_losses"].append(val_mse)

        scheduler.step()

        # ---- Early stopping ----
        if val_mse < best_val_loss - 0.01:
            best_val_loss = val_mse
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1

        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            marker = " ★" if epoch == best_epoch else ""
            print(f"{epoch:>6} | {train_mse:>10.4f} | {val_mse:>10.4f} | {current_lr:>10.2e}{marker}")

        if no_improve >= patience:
            print(f"\n  Early stop at epoch {epoch}. Best val MSE: {best_val_loss:.4f} (epoch {best_epoch})")
            break

    elapsed = time.time() - t_start
    print(f"\n  Done in {elapsed:.1f}s | Best Val MSE: {best_val_loss:.4f} @ epoch {best_epoch}")
    print(f"  Saved checkpoint → {ckpt_path}")

    # Reload best weights
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    return history


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # --- Data ---
    dataset = load_or_generate()
    X_train, y_train, X_val, y_val, X_test, y_test = to_tensors(dataset, device)
    train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val, BATCH)

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # --- Models ---
    models = {
        "lstm":        build_lstm(input_size=FEATURE_DIM),
        "transformer": build_transformer(input_size=FEATURE_DIM),
        "tcn":         build_tcn(input_size=FEATURE_DIM),
    }

    histories = {}
    for name, model in models.items():
        history = train_model(name, model, train_loader, val_loader, device)
        histories[name] = history

    # Save training histories
    hist_path = os.path.join(CKPT_DIR, "training_histories.npz")
    np.savez(hist_path, **{f"{k}_train": v["train_losses"] for k, v in histories.items()},
                       **{f"{k}_val":   v["val_losses"]   for k, v in histories.items()})
    print(f"\nTraining histories saved → {hist_path}")
    print("\nAll models trained! Run `python -m ml_scoring.evaluate` for comparison.")


if __name__ == "__main__":
    main()

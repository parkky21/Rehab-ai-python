"""
Training Script — ML Exercise Scoring Models (v2 — Improved)
=============================================================
Trains LSTM, Transformer, and TCN with improved training recipe:
  - SmoothL1Loss (Huber) instead of MSE — robust to score outliers
  - Linear warmup + cosine decay LR schedule
  - Per-epoch logging of ALL metrics to CSV + terminal
  - Larger dataset (14K reps / 12 features)

Usage
-----
    cd python-server
    python -m ml_scoring.train
"""

import os
import sys
import csv
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_scoring.data_generator import generate_dataset, SEQ_LEN, FEATURE_DIM
from ml_scoring.models.lstm_model import build_lstm
from ml_scoring.models.transformer_model import build_transformer
from ml_scoring.models.tcn_model import build_tcn

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
EPOCHS       = int(os.environ.get("EPOCHS",   120))
BATCH        = int(os.environ.get("BATCH",    128))
LR           = float(os.environ.get("LR",     5e-4))
PATIENCE     = int(os.environ.get("PATIENCE", 18))
WARMUP_EPOCHS = 5  # Linear warmup period

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "dataset.npz")
CKPT_DIR  = os.path.join(BASE_DIR, "checkpoints")
LOGS_DIR  = os.path.join(BASE_DIR, "training_logs")
os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_or_generate() -> dict:
    if os.path.exists(DATA_PATH):
        print(f"Loading existing dataset from {DATA_PATH} ...")
        data = np.load(DATA_PATH)
        return {k: data[k] for k in data.files}
    print("No dataset found — generating now ...")
    save_dir = os.path.join(BASE_DIR, "data")
    return generate_dataset(save_dir=save_dir)


def to_tensors(dataset: dict, device: torch.device):
    def t(arr):
        return torch.tensor(arr, dtype=torch.float32, device=device)
    return (
        t(dataset["X_train"]), t(dataset["y_train"]),
        t(dataset["X_val"]),   t(dataset["y_val"]),
        t(dataset["X_test"]),  t(dataset["y_test"]),
    )


def make_loaders(X_train, y_train, X_val, y_val, batch_size: int):
    train_ds = TensorDataset(X_train, y_train)
    val_ds   = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Custom LR schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            factor = (self.last_epoch + 1) / max(1, self.warmup_epochs)
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            # Cosine decay
            import math
            progress = (self.last_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [self.min_lr + (base_lr - self.min_lr) * factor for base_lr in self.base_lrs]


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, R² for each score + overall."""
    score_names = ["rom", "stability", "tempo", "final"]
    metrics = {}
    for i, name in enumerate(score_names):
        true_i = y_true[:, i]
        pred_i = y_pred[:, i]
        mae = float(np.mean(np.abs(true_i - pred_i)))
        rmse = float(np.sqrt(np.mean((true_i - pred_i) ** 2)))
        ss_res = np.sum((true_i - pred_i) ** 2)
        ss_tot = np.sum((true_i - np.mean(true_i)) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-8))
        metrics[f"mae_{name}"] = mae
        metrics[f"rmse_{name}"] = rmse
        metrics[f"r2_{name}"] = r2

    # Overall
    mae_all = float(np.mean(np.abs(y_true - y_pred)))
    metrics["mae_all"] = mae_all
    return metrics


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
    Train with SmoothL1Loss (Huber) + AdamW + warmup+cosine LR.
    Logs every epoch to CSV and terminal.
    """
    model = model.to(device)
    criterion = nn.SmoothL1Loss(beta=5.0)  # Huber loss, transitions at delta=5
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=WARMUP_EPOCHS, total_epochs=epochs)

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0

    ckpt_path = os.path.join(CKPT_DIR, f"{name}_best.pt")
    csv_path  = os.path.join(LOGS_DIR, f"{name}_training_log.csv")

    # CSV header
    csv_fields = [
        "epoch", "lr", "train_loss", "val_loss",
        "val_mae_rom", "val_mae_stability", "val_mae_tempo", "val_mae_final",
        "val_rmse_final", "val_r2_final", "val_mae_all",
        "is_best",
    ]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n{'='*70}")
    print(f"  Training {name.upper()}  |  {param_count:,} params  |  Huber(β=5) + AdamW")
    print(f"{'='*70}")
    print(f"{'Ep':>4} | {'LR':>9} | {'TrainL':>8} | {'ValL':>8} | "
          f"{'MAE_R':>6} {'MAE_S':>6} {'MAE_T':>6} {'MAE_F':>6} | "
          f"{'R²_F':>7} | {'Best':>4}")
    print("-" * 86)

    t_start = time.time()
    all_epoch_data = []

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_losses.append(loss.item())

        # ---- Validate ----
        model.eval()
        val_losses = []
        all_preds, all_trues = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_losses.append(loss.item())
                all_preds.append(pred.cpu().numpy())
                all_trues.append(y_batch.cpu().numpy())

        train_loss = float(np.mean(train_losses))
        val_loss   = float(np.mean(val_losses))
        current_lr = optimizer.param_groups[0]["lr"]

        # Detailed validation metrics
        val_preds = np.vstack(all_preds)
        val_trues = np.vstack(all_trues)
        metrics = compute_metrics(val_trues, val_preds)

        scheduler.step()

        # ---- Early stopping ----
        is_best = val_loss < best_val_loss - 0.001
        if is_best:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            no_improve += 1

        # ---- Log to CSV ----
        row = {
            "epoch": epoch,
            "lr": f"{current_lr:.2e}",
            "train_loss": f"{train_loss:.4f}",
            "val_loss": f"{val_loss:.4f}",
            "val_mae_rom": f"{metrics['mae_rom']:.3f}",
            "val_mae_stability": f"{metrics['mae_stability']:.3f}",
            "val_mae_tempo": f"{metrics['mae_tempo']:.3f}",
            "val_mae_final": f"{metrics['mae_final']:.3f}",
            "val_rmse_final": f"{metrics['rmse_final']:.3f}",
            "val_r2_final": f"{metrics['r2_final']:.4f}",
            "val_mae_all": f"{metrics['mae_all']:.3f}",
            "is_best": "★" if is_best else "",
        }
        csv_writer.writerow(row)
        csv_file.flush()

        all_epoch_data.append({
            "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
            **metrics,
        })

        # ---- Print every epoch ----
        marker = "★" if is_best else ""
        print(
            f"{epoch:>4} | {current_lr:>9.2e} | {train_loss:>8.4f} | {val_loss:>8.4f} | "
            f"{metrics['mae_rom']:>6.2f} {metrics['mae_stability']:>6.2f} "
            f"{metrics['mae_tempo']:>6.2f} {metrics['mae_final']:>6.2f} | "
            f"{metrics['r2_final']:>7.4f} | {marker:>4}"
        )

        if no_improve >= patience:
            print(f"\n  Early stop at epoch {epoch}. Best val loss: {best_val_loss:.4f} (epoch {best_epoch})")
            break

    csv_file.close()
    elapsed = time.time() - t_start
    print(f"\n  Done in {elapsed:.1f}s | Best Val Loss: {best_val_loss:.4f} @ epoch {best_epoch}")
    print(f"  Checkpoint → {ckpt_path}")
    print(f"  Training log → {csv_path}")

    # Reload best weights
    model.load_state_dict(torch.load(ckpt_path, weights_only=True))

    return {
        "epochs": all_epoch_data,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\nDevice: {device}")
    print(f"Config: epochs={EPOCHS}, batch={BATCH}, lr={LR}, patience={PATIENCE}, warmup={WARMUP_EPOCHS}")

    # --- Data ---
    dataset = load_or_generate()
    X_train, y_train, X_val, y_val, X_test, y_test = to_tensors(dataset, device)
    train_loader, val_loader = make_loaders(X_train, y_train, X_val, y_val, BATCH)

    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"Features per frame: {FEATURE_DIM}")

    # --- Models ---
    models = {
        "lstm":        build_lstm(input_size=FEATURE_DIM),
        "transformer": build_transformer(input_size=FEATURE_DIM),
        "tcn":         build_tcn(input_size=FEATURE_DIM),
    }

    all_histories = {}
    for name, model in models.items():
        history = train_model(name, model, train_loader, val_loader, device)
        all_histories[name] = history

    # Save combined training metrics
    hist_path = os.path.join(CKPT_DIR, "training_histories.npz")
    save_dict = {}
    for name, hist in all_histories.items():
        for metric_key in ["train_loss", "val_loss", "mae_rom", "mae_stability",
                           "mae_tempo", "mae_final", "r2_final", "mae_all"]:
            vals = [ep.get(metric_key, 0.0) for ep in hist["epochs"]]
            save_dict[f"{name}_{metric_key}"] = vals
    np.savez(hist_path, **save_dict)

    print(f"\n{'='*70}")
    print("  TRAINING COMPLETE — SUMMARY")
    print(f"{'='*70}")
    for name, hist in all_histories.items():
        final_ep = hist["epochs"][-1] if hist["epochs"] else {}
        print(f"\n  {name.upper()}")
        print(f"    Best epoch: {hist['best_epoch']} | Best val loss: {hist['best_val_loss']:.4f}")
        print(f"    Final R²(final): {final_ep.get('r2_final', 0):.4f}")
        print(f"    Final MAE: ROM={final_ep.get('mae_rom', 0):.2f}  "
              f"Stab={final_ep.get('mae_stability', 0):.2f}  "
              f"Tempo={final_ep.get('mae_tempo', 0):.2f}  "
              f"Final={final_ep.get('mae_final', 0):.2f}")

    print(f"\nTraining histories → {hist_path}")
    print(f"Per-epoch logs → {LOGS_DIR}/")
    print("\nRun `python -m ml_scoring.evaluate` for full comparison.")


if __name__ == "__main__":
    main()

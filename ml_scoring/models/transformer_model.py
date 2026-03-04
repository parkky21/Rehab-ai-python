"""
Transformer-based Exercise Rep Scorer
=======================================
Lightweight Transformer encoder with sinusoidal positional encoding.
After encoding, applies mean pooling over the time dimension then
regresses to four score values: [rom, stability, tempo, final].

Architecture
------------
  Input [B, T, F]
  → Linear projection to d_model
  → Sinusoidal positional encoding
  → 3× TransformerEncoderLayer(d_model=64, nhead=4, dim_ff=256)
  → Mean pool over T
  → Linear(128) → GELU → dropout
  → Linear(4)
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al. 2017)."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, T, d_model]"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerScorer(nn.Module):
    """
    Transformer encoder with mean-pooling regression head.

    Parameters
    ----------
    input_size : int
        Number of input features per frame (default 7).
    d_model : int
        Transformer internal dimension (default 64).
    nhead : int
        Number of attention heads (default 4).
    num_layers : int
        Number of stacked TransformerEncoderLayers (default 3).
    dim_feedforward : int
        FFN intermediate dimension (default 256).
    dropout : float
        Dropout in encoder layers (default 0.2).
    num_scores : int
        Number of regression outputs (default 4).
    """

    def __init__(
        self,
        input_size: int = 7,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.2,
        num_scores: int = 4,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout * 0.5)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN: more stable training
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)

        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, num_scores),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor [B, T, F]

        Returns
        -------
        Tensor [B, num_scores]  — predicted scores in [0, 100]
        """
        x = self.input_proj(x)       # [B, T, d_model]
        x = self.pos_enc(x)
        x = self.encoder(x)          # [B, T, d_model]
        x = self.norm(x)
        x = x.mean(dim=1)            # mean pool over T → [B, d_model]
        scores = self.head(x)        # [B, 4]
        scores = torch.clamp(scores, 0.0, 100.0)
        return scores


def build_transformer(input_size: int = 7, num_scores: int = 4) -> TransformerScorer:
    """Factory with default hyperparameters."""
    return TransformerScorer(
        input_size=input_size,
        d_model=64,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.2,
        num_scores=num_scores,
    )

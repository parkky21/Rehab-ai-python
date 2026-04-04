"""
Transformer-based Exercise Rep Scorer (v2 — Improved)
======================================================
Deeper Transformer with 4 layers, d_model=96, CLS token pooling,
and a deeper regression head.

Architecture
------------
  Input [B, T, 12]
  → Linear(96) → positional encoding
  → [CLS] token prepended
  → 4× TransformerEncoder(d=96, h=6, ff=384)
  → CLS output + mean pool concat
  → LayerNorm → Linear(256) → GELU → Linear(128) → GELU
  → Linear(4)
"""

import math
import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
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
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerScorer(nn.Module):
    def __init__(
        self,
        input_size: int = 12,
        d_model: int = 96,
        nhead: int = 6,
        num_layers: int = 4,
        dim_feedforward: int = 384,
        dropout: float = 0.15,
        num_scores: int = 4,
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, dropout=dropout * 0.5)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model * 2)  # CLS + mean pool

        self.head = nn.Sequential(
            nn.Linear(d_model * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(128, num_scores),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        x = self.input_proj(x)                           # [B, T, d_model]
        x = self.pos_enc(x)

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)            # [B, 1, d_model]
        x = torch.cat([cls, x], dim=1)                    # [B, T+1, d_model]

        x = self.encoder(x)                                # [B, T+1, d_model]

        cls_out = x[:, 0]                                  # [B, d_model]
        mean_pool = x[:, 1:].mean(dim=1)                   # [B, d_model]

        combined = torch.cat([cls_out, mean_pool], dim=1)  # [B, 2*d_model]
        combined = self.norm(combined)

        scores = self.head(combined)
        scores = torch.clamp(scores, 0.0, 100.0)
        return scores


def build_transformer(input_size: int = 12, num_scores: int = 4) -> TransformerScorer:
    return TransformerScorer(
        input_size=input_size,
        d_model=96,
        nhead=6,
        num_layers=4,
        dim_feedforward=384,
        dropout=0.15,
        num_scores=num_scores,
    )

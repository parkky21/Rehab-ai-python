"""
LSTM-based Exercise Rep Scorer (v2 — Improved)
================================================
Stacked Bidirectional LSTM with larger capacity and better regularization.

Architecture
------------
  Input [B, T, 12]
  → BiLSTM(192) → dropout(0.25)
  → BiLSTM(96)  → concat(last_fwd, last_bwd, avg_pool, max_pool)
  → LayerNorm → Linear(256) → GELU → dropout(0.15)
  → Linear(128) → GELU
  → Linear(4)  (rom, stability, tempo, final scores)
"""

import torch
import torch.nn as nn


class LSTMScorer(nn.Module):
    def __init__(
        self,
        input_size: int = 12,
        hidden_size: int = 192,
        dropout: float = 0.25,
        num_scores: int = 4,
    ):
        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.lstm2 = nn.LSTM(
            input_size=hidden_size * 2,
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.drop2 = nn.Dropout(dropout * 0.6)

        # hidden_size//2 * 2 (bidirectional) for last hidden
        # + same for avg pool + same for max pool = 3 × hidden_size
        pool_size = hidden_size  # bidir → hidden_size//2 * 2 = hidden_size
        concat_size = pool_size * 3  # last_hidden + avg_pool + max_pool

        self.norm = nn.LayerNorm(concat_size)

        self.head = nn.Sequential(
            nn.Linear(concat_size, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(128, num_scores),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out1, _ = self.lstm1(x)
        out1 = self.drop1(out1)

        out2, (h_n, _) = self.lstm2(out1)  # out2: [B, T, hidden_size]

        # Last hidden from both directions
        last_hidden = torch.cat([h_n[0], h_n[1]], dim=1)  # [B, hidden_size]

        # Global pooling over time
        avg_pool = out2.mean(dim=1)   # [B, hidden_size]
        max_pool = out2.max(dim=1).values  # [B, hidden_size]

        # Concatenate all representations
        combined = torch.cat([last_hidden, avg_pool, max_pool], dim=1)
        combined = self.norm(combined)

        scores = self.head(combined)
        scores = torch.clamp(scores, 0.0, 100.0)
        return scores


def build_lstm(input_size: int = 12, num_scores: int = 4) -> LSTMScorer:
    return LSTMScorer(input_size=input_size, hidden_size=192, dropout=0.25, num_scores=num_scores)

"""
LSTM-based Exercise Rep Scorer
================================
Stacked Bidirectional LSTM that ingests a sequence of per-frame features
and regresses to four score values: [rom, stability, tempo, final].

Architecture
------------
  Input [B, T, F]
  → BiLSTM(128) → dropout(0.3)
  → BiLSTM(64)  → take last hidden
  → LayerNorm → Linear(128) → GELU → dropout(0.2)
  → Linear(4)  (each head predicts one score, clamped 0-100)
"""

import torch
import torch.nn as nn


class LSTMScorer(nn.Module):
    """
    Bidirectional stacked LSTM for per-rep exercise scoring.

    Parameters
    ----------
    input_size : int
        Number of features per frame (default 7).
    hidden_size : int
        LSTM hidden units per direction (default 128 for layer 1, 64 for layer 2).
    dropout : float
        Dropout applied between layers (default 0.3).
    num_scores : int
        Number of regression outputs (default 4: rom, stability, tempo, final).
    """

    def __init__(
        self,
        input_size: int = 7,
        hidden_size: int = 128,
        dropout: float = 0.3,
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
            input_size=hidden_size * 2,  # bidirectional
            hidden_size=hidden_size // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.drop2 = nn.Dropout(dropout * 0.7)

        lstm2_out_size = (hidden_size // 2) * 2  # bidirectional → hidden_size

        self.norm = nn.LayerNorm(lstm2_out_size)

        self.head = nn.Sequential(
            nn.Linear(lstm2_out_size, 128),
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
        out1, _ = self.lstm1(x)          # [B, T, 2*128]
        out1 = self.drop1(out1)

        out2, (h_n, _) = self.lstm2(out1)  # [B, T, hidden_size]

        # Concatenate final forward + backward hidden states
        # h_n: [2, B, hidden//2]  (2 because bidirectional)
        last = torch.cat([h_n[0], h_n[1]], dim=1)   # [B, hidden_size]

        last = self.norm(last)
        scores = self.head(last)           # [B, 4]
        scores = torch.clamp(scores, 0.0, 100.0)
        return scores


def build_lstm(input_size: int = 7, num_scores: int = 4) -> LSTMScorer:
    """Factory with default hyperparameters."""
    return LSTMScorer(input_size=input_size, hidden_size=128, dropout=0.3, num_scores=num_scores)

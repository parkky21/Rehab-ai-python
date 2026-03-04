"""
Temporal Convolutional Network (TCN) Exercise Rep Scorer
=========================================================
Uses stacked dilated causal 1-D convolutions with residual connections.
Captures local and long-range temporal patterns without recurrence.

Architecture
------------
  Input [B, T, F]  →  permute to [B, F, T]  (Conv1d convention)
  → TCN Block (dilation=1,  channels=64)
  → TCN Block (dilation=2,  channels=64)
  → TCN Block (dilation=4,  channels=128)
  → TCN Block (dilation=8,  channels=128)
  → GlobalAvgPool over T  →  [B, 128]
  → Linear(128) → GELU → dropout
  → Linear(4)

Each TCN Block:
  Causal Conv1d → WeightNorm → GELU → dropout
  Causal Conv1d → WeightNorm → GELU → dropout
  + residual (1×1 conv if channels change)
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class CausalConv1d(nn.Module):
    """1-D convolution with left-side (causal) zero-padding."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = weight_norm(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                dilation=dilation,
                padding=self.padding,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, T]"""
        out = self.conv(x)
        # Remove the future-looking padding on the right
        return out[:, :, : x.size(2)]


class TCNBlock(nn.Module):
    """
    Two-layer dilated causal residual block.

    Parameters
    ----------
    in_channels : int
    out_channels : int
    kernel_size : int  (default 3)
    dilation : int
    dropout : float
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)

        self.act1 = nn.GELU()
        self.act2 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.norm2 = nn.BatchNorm1d(out_channels)

        # Residual projection if channel dimensions differ
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_channels, T]"""
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.drop2(out)

        return out + residual


class TCNScorer(nn.Module):
    """
    Temporal Convolutional Network with global average pooling and regression head.

    Parameters
    ----------
    input_size : int
        Number of input features per frame (default 7).
    channels : list[int]
        Output channels for each TCN block (len = num blocks).
    kernel_size : int
        Convolutional kernel size (default 3).
    dropout : float
        Dropout in TCN blocks (default 0.2).
    num_scores : int
        Number of regression outputs (default 4).
    """

    def __init__(
        self,
        input_size: int = 7,
        channels: list = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        num_scores: int = 4,
    ):
        super().__init__()

        if channels is None:
            channels = [64, 64, 128, 128]

        # Build TCN blocks with exponentially increasing dilation
        layers = []
        in_ch = input_size
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(
                TCNBlock(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            in_ch = out_ch

        self.tcn = nn.Sequential(*layers)
        final_channels = channels[-1]

        self.head = nn.Sequential(
            nn.Linear(final_channels, 128),
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
        x = x.permute(0, 2, 1)      # [B, F, T] for Conv1d
        x = self.tcn(x)              # [B, C_last, T]
        x = x.mean(dim=2)            # GlobalAvgPool → [B, C_last]
        scores = self.head(x)
        scores = torch.clamp(scores, 0.0, 100.0)
        return scores


def build_tcn(input_size: int = 7, num_scores: int = 4) -> TCNScorer:
    """Factory with default hyperparameters."""
    return TCNScorer(
        input_size=input_size,
        channels=[64, 64, 128, 128],
        kernel_size=3,
        dropout=0.2,
        num_scores=num_scores,
    )

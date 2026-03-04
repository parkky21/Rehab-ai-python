"""
TCN Exercise Rep Scorer (v2 — Improved)
========================================
Deeper TCN with more channels and squeeze-and-excitation attention.

Architecture
------------
  Input [B, T, 12]  →  permute [B, 12, T]
  → TCNBlock(d=1,  ch=96)
  → TCNBlock(d=2,  ch=96)
  → TCNBlock(d=4,  ch=128)
  → TCNBlock(d=8,  ch=128)
  → TCNBlock(d=16, ch=192)
  → GlobalAvgPool + GlobalMaxPool concat → [B, 384]
  → Linear(256) → GELU → dropout → Linear(128) → GELU
  → Linear(4)
"""

import torch
import torch.nn as nn
from torch.nn.utils import parametrizations


class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.padding,
        )
        self.conv = parametrizations.weight_norm(conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return out[:, :, :x.size(2)]


class SqueezeExcite(nn.Module):
    """Channel attention (SE block)."""
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.GELU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        s = x.mean(dim=2)           # [B, C]
        s = self.fc(s).unsqueeze(2)  # [B, C, 1]
        return x * s


class TCNBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        dilation: int = 1,
        dropout: float = 0.15,
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

        self.se = SqueezeExcite(out_channels)

        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.drop2(out)

        out = self.se(out)  # Channel attention

        return out + residual


class TCNScorer(nn.Module):
    def __init__(
        self,
        input_size: int = 12,
        channels: list = None,
        kernel_size: int = 3,
        dropout: float = 0.15,
        num_scores: int = 4,
    ):
        super().__init__()

        if channels is None:
            channels = [96, 96, 128, 128, 192]

        layers = []
        in_ch = input_size
        for i, out_ch in enumerate(channels):
            dilation = 2 ** i
            layers.append(
                TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout)
            )
            in_ch = out_ch

        self.tcn = nn.Sequential(*layers)
        final_channels = channels[-1]

        # Dual pooling: avg + max
        self.head = nn.Sequential(
            nn.Linear(final_channels * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout * 0.3),
            nn.Linear(128, num_scores),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)         # [B, F, T]
        x = self.tcn(x)                 # [B, C, T]

        avg = x.mean(dim=2)             # [B, C]
        mx  = x.max(dim=2).values       # [B, C]
        pooled = torch.cat([avg, mx], dim=1)  # [B, 2*C]

        scores = self.head(pooled)
        scores = torch.clamp(scores, 0.0, 100.0)
        return scores


def build_tcn(input_size: int = 12, num_scores: int = 4) -> TCNScorer:
    return TCNScorer(
        input_size=input_size,
        channels=[96, 96, 128, 128, 192],
        kernel_size=3,
        dropout=0.15,
        num_scores=num_scores,
    )

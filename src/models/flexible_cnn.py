"""
src/models/flexible_cnn.py

FlexibleEmotionCNN — ablation-capable variant of Emotion1DCNN.

Supports comprehensive architectural and training parameter exploration:
  - Variable depth / width (channels list)
  - Pluggable normalization  (BatchNorm / GroupNorm / LayerNorm / none)
  - Pluggable activation     (ReLU / GELU / LeakyReLU)
  - Pluggable pooling        (AdaptiveAvg / AdaptiveMax / AdaptiveAvg4)
  - Configurable dropout, hidden_dim, weight_decay, init_scheme

Also exports:
  - FocalLoss          — imbalanced-class loss for handling class imbalance
  - make_loss_fn()     — factory that returns the correct loss for a config
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# N_MELS and N_CLASSES are resolved lazily to avoid import-time side effects.
# Callers should pass explicit values when constructing FlexibleEmotionCNN.
_N_MELS = 128
_N_CLASSES = 12  # len(EMOTION_CLASSES) — kept here as default


# ── Activation factory ────────────────────────────────────────────


def make_activation(name: str, slope: float = 0.01) -> nn.Module:
    """Return the activation module corresponding to *name*."""
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "leaky_relu":
        return nn.LeakyReLU(negative_slope=slope)
    raise ValueError(f"Unknown activation: {name!r}")


# ── Normalization factory ─────────────────────────────────────────


def make_norm(
    norm_type: str, channels: int, group_norm_groups: int = 32
) -> nn.Module | None:
    """Return the normalization module, or *None* for norm_type='none'."""
    if norm_type == "batch":
        return nn.BatchNorm1d(channels)
    if norm_type == "group":
        groups = min(group_norm_groups, channels)
        return nn.GroupNorm(groups, channels)
    if norm_type == "layer":
        return nn.GroupNorm(1, channels)  # LayerNorm equivalent for 1-D conv
    if norm_type == "none":
        return None
    raise ValueError(f"Unknown norm_type: {norm_type!r}")


# ── Pooling strategy factory ──────────────────────────────────────


def make_pool(pool_type: str) -> nn.Module:
    """Return the temporal pooling module corresponding to *pool_type*."""
    if pool_type == "adaptive_avg":
        return nn.AdaptiveAvgPool1d(1)
    if pool_type == "adaptive_max":
        return nn.AdaptiveMaxPool1d(1)
    if pool_type == "adaptive_avg4":
        return nn.AdaptiveAvgPool1d(4)  # retains 4 time steps (temporal structure)
    raise ValueError(f"Unknown pool_type: {pool_type!r}")


# ── FlexibleEmotionCNN ─────────────────────────────────────────────────────────


class FlexibleEmotionCNN(nn.Module):
    """
    Flexible 1-D CNN for audio emotion classification.

    Input:  (B, N_MELS, T) — mel-spectrogram frames
    Output: (B, num_classes) — class logits

    All structural hyper-parameters are driven by a CNNAblationConfig so that
    every design decision is reproducible from a single config JSON.
    """

    def __init__(self, cfg, num_classes: int = _N_CLASSES):
        """
        Args:
            cfg:         CNNAblationConfig (or any object with the same fields).
            num_classes: Number of output classes (default: 12 emotion classes).
        """
        super().__init__()
        in_ch = _N_MELS
        blocks: list[nn.Module] = []

        for i, out_ch in enumerate(cfg.channels):
            blocks.append(
                nn.Conv1d(
                    in_ch,
                    out_ch,
                    kernel_size=cfg.kernel_size,
                    padding=cfg.kernel_size // 2,
                )
            )
            norm = make_norm(cfg.norm_type, out_ch, cfg.group_norm_groups)
            if norm is not None:
                blocks.append(norm)
            blocks.append(make_activation(cfg.activation, cfg.leaky_slope))
            # Mid-network max-pool halves the time dimension (except last block)
            if i < len(cfg.channels) - 1:
                blocks.append(nn.MaxPool1d(kernel_size=2))
            in_ch = out_ch

        blocks.append(make_pool(cfg.pool_type))
        self.features = nn.Sequential(*blocks)

        # Flattened size: adaptive_avg/max → 1 frame, adaptive_avg4 → 4 frames
        flat_mult = 4 if cfg.pool_type == "adaptive_avg4" else 1
        flat_dim = in_ch * flat_mult

        self.classifier = nn.Sequential(
            nn.Dropout(p=cfg.dropout),
            nn.Linear(flat_dim, cfg.hidden_dim),
            make_activation(cfg.activation, cfg.leaky_slope),
            nn.Linear(cfg.hidden_dim, num_classes),
        )

        if cfg.init_scheme != "default":
            self._init_weights(cfg.init_scheme)

    # ── Weight initialization ──────────────────────────────────────────────────

    def _init_weights(self, scheme: str) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                if scheme == "kaiming":
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                elif scheme == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                if scheme == "kaiming":
                    nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                elif scheme == "xavier":
                    nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward passes ─────────────────────────────────────────────────────────

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Return flattened feature vector before the classifier head."""
        return torch.flatten(self.features(x), 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 128, T) mel spectrogram
        Returns:
            (B, num_classes) logits
        """
        return self.classifier(self.embed(x))


# ── Loss functions ────────────────────────────────────────────────


class FocalLoss(nn.Module):
    """
    Focal Loss [Lin et al., 2017] for imbalanced multi-class classification.

    Reduces the loss contribution from easy examples so the model focuses on
    hard, under-represented emotion classes.  *gamma* controls focus strength.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def make_loss_fn(cfg, class_weights: torch.Tensor | None) -> nn.Module:
    """
    Loss function factory.

    Args:
        cfg:           CNNAblationConfig (or compatible).
        class_weights: Per-class weights tensor, or None to disable weighting.
    """
    w = class_weights if cfg.use_class_weights else None
    if cfg.loss_fn == "focal":
        return FocalLoss(gamma=cfg.focal_gamma, weight=w)
    if cfg.loss_fn == "ce":
        return nn.CrossEntropyLoss(weight=w)
    if cfg.loss_fn == "ce_smooth":
        return nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smooth)
    raise ValueError(f"Unknown loss_fn: {cfg.loss_fn!r}")

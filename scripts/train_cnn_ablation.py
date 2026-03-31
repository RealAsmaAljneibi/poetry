"""
scripts/train_cnn_ablation.py

Systematic ablation study for Emotion1DCNN (from-scratch audio emotion classifier).

Explores architectural and training hyperparameter choices including:
  - Architecture & depth (conv blocks, pooling strategies)
  - Activation functions & normalization schemes (ReLU, GELU, BatchNorm, GroupNorm)
  - Loss functions (CrossEntropy, Focal loss with label smoothing)
  - Optimizers & learning rate schedules (SGD+momentum, Adam, AdamW with cosine annealing)
  - Regularization strategies (dropout, L2 weight decay, gradient clipping)
  - Data augmentation (SpecAugment, MixUp, CutMix on mel-spectrograms)
  - Training stability (gradient explosion detection, early stopping, mixed precision)

Run naming convention (consistent with Whisper ASR runs):
  CNN-R1  Baseline       — AdamW + cosine + BatchNorm + ReLU + Focal + MaxPool
  CNN-R2  Activation     — GELU vs LeakyReLU
  CNN-R3  Loss           — CE + label_smoothing=0.1 vs Focal γ=1 vs Focal γ=2
  CNN-R4  Optimizer      — SGD+momentum vs Adam vs AdamW
  CNN-R5  Normalization  — GroupNorm vs LayerNorm vs no norm
  CNN-R6  Architecture   — deeper (3 conv blocks) vs shallower (1 block)
  CNN-R7  Augmentation   — SpecAugment on/off, Gaussian noise, mixup
  CNN-R8  Regularization — dropout sweep 0.1 / 0.3 / 0.5 + L2 weight decay
  CNN-R9  Pooling        — MaxPool vs AvgPool vs AdaptiveAvgPool(4)
  CNN-R10 Scheduler      — OneCycleLR vs StepLR vs cosine_annealing
  CNN-R11 Init           — Kaiming vs Xavier vs default
  CNN-R12 Mixed Prec     — AMP float16 vs float32
  CNN-R13 MixUp          — mel-spec linear blend, λ~Beta(0.4,0.4)
  CNN-R14 CutMix         — freq×time patch paste, proportional label
  CNN-R15 MixUp+CutMix   — both augmentations stacked
  CNN-R16 Best-Combo     — GELU+3-block+CE+dropout=0.1+AvgPool4+OneCycle+MixUp (hyperparameter tuning)

Each run is logged to:
  outputs/models/cnn_ablation/<run_id>/
    config.json           — Pydantic config dump
    training_history.json — per-epoch metrics
    best_model.pt         — state dict of best val-F1 epoch

Visualizations produced per run:
  outputs/figures/cnn_ablation/<run_id>/
    loss_curve.png         — train/val loss per epoch
    f1_curve.png           — train/val macro-F1 per epoch
    confusion_matrix.png   — test set confusion matrix
    grad_norm_curve.png    — gradient L2 norm per step (gradient explosion check)
    lr_schedule.png        — learning rate trace

Usage:
    # Run single ablation
    uv run python scripts/train_cnn_ablation.py --run CNN-R1

    # Run all ablations sequentially
    uv run python scripts/train_cnn_ablation.py --run all

    # Run baseline only (quick smoke test)
    uv run python scripts/train_cnn_ablation.py --run CNN-R1 --epochs 3

    # List available runs
    uv run python scripts/train_cnn_ablation.py --list
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Literal

import librosa
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for CI / headless
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.labels import EMOTION_CLASSES, encode_emotion
from src.models.flexible_cnn import FlexibleEmotionCNN, make_loss_fn
from src.training.trainer import EarlyStopper, set_seed

# ── Constants ─────────────────────────────────────────────────────────────────

PROJECT_ROOT  = Path(__file__).parent.parent
DATA_DIR      = PROJECT_ROOT / "data" / "processed"
OUTPUT_ROOT   = PROJECT_ROOT / "outputs" / "models" / "cnn_ablation"
FIGURE_ROOT   = PROJECT_ROOT / "outputs" / "figures" / "cnn_ablation"
N_CLASSES     = len(EMOTION_CLASSES)   # 12
SAMPLING_RATE = 16_000
MAX_AUDIO_SEC = 30
N_MELS        = 128
SEED          = 42


# ══════════════════════════════════════════════════════════════════════════════
# 1. ABLATION CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CNNAblationConfig:
    """
    One complete configuration for a single ablation run.

    All major training components are represented as explicit fields so that every
    design decision is documented, versioned, and reproducible.
    """

    run_id:       str                          # e.g. "CNN-R1"
    description:  str                          # human-readable one-liner

    # ── 1 & 2: Architecture & Layers ───────────────────────────
    channels:     list[int]   = None  # conv channel widths per block, e.g. [256, 512]
    kernel_size:  int         = 5
    hidden_dim:   int         = 128   # linear layer width before classifier

    # ── 3: Weight Initialization ───────────────────────────────
    init_scheme:  Literal["kaiming", "xavier", "default"] = "kaiming"

    # ── 4: Loss Function ───────────────────────────────────────
    loss_fn:      Literal["focal", "ce", "ce_smooth"] = "focal"
    focal_gamma:  float       = 2.0
    label_smooth: float       = 0.0   # for "ce_smooth"

    # ── 5: Optimizer ───────────────────────────────────────────
    optimizer:    Literal["adamw", "adam", "sgd"] = "adamw"
    momentum:     float       = 0.9   # SGD only

    # ── 6 & 7: LR and Scheduler ───────────────────────────────
    lr:           float       = 5e-4
    lr_scheduler: Literal["cosine", "onecycle", "step", "none"] = "cosine"
    lr_step_size: int         = 10    # for StepLR
    lr_gamma:     float       = 0.5   # for StepLR

    # ── 8: Batch Size ──────────────────────────────────────────
    batch_size:   int         = 32
    grad_accum:   int         = 1

    # ── 9: Regularization ──────────────────────────────────────
    dropout:      float       = 0.3
    weight_decay: float       = 1e-4
    use_class_weights: bool   = True  # class-weighted loss

    # ── 10: Normalization ──────────────────────────────────────
    norm_type:    Literal["batch", "group", "layer", "none"] = "batch"
    group_norm_groups: int    = 32

    # ── 11: Activation Functions ───────────────────────────────
    activation:   Literal["relu", "gelu", "leaky_relu"] = "relu"
    leaky_slope:  float       = 0.01

    # ── 12 & 13: Checkpointing & Early Stopping ───────────────
    epochs:       int         = 30
    patience:     int         = 7

    # ── 14: Data Augmentation ─────────────────────────────────
    spec_augment:       bool  = True
    spec_time_masks:    int   = 2
    spec_time_max:      int   = 40
    spec_freq_masks:    int   = 2
    spec_freq_max:      int   = 20
    gaussian_noise_std: float = 0.005
    # MixUp (Zhang et al., 2018) — linear blend of two mel-spectrograms
    # Taught in learn/04_lab_dl_with_limited_data/01_augmentations.ipynb
    use_mixup:          bool  = False
    mixup_alpha:        float = 0.4   # Beta(alpha, alpha) concentration; higher = more mixing
    # CutMix (Yun et al., 2019) — paste a freq-time patch from one sample onto another
    # Audio adaptation: cut rectangular region of mel-spectrogram (freq × time)
    use_cutmix:         bool  = False
    cutmix_alpha:       float = 1.0   # Beta(alpha, alpha) for patch area ratio

    # ── 15: Gradient Clipping ─────────────────────────────────
    grad_clip:    float | None = 1.0

    # ── 17: Warmup ─────────────────────────────────────────────
    warmup_epochs: int        = 2

    # ── 17: Mixed Precision ────────────────────────────────────
    mixed_precision: bool     = False

    # ── 22: Pooling Strategy ──────────────────────────────────
    pool_type:    Literal["adaptive_avg", "adaptive_max", "adaptive_avg4"] = "adaptive_avg"

    def __post_init__(self):
        if self.channels is None:
            self.channels = [256, 512]

    def to_dict(self) -> dict:
        return asdict(self)


# ── Ablation run registry ──────────────────────────────────────────────────

def _make_runs() -> dict[str, CNNAblationConfig]:
    """Return ordered dict of all ablation configs."""
    runs: dict[str, CNNAblationConfig] = {}

    # ── CNN-R1: BASELINE ──────────────────────────────────────────────────
    runs["CNN-R1"] = CNNAblationConfig(
        run_id="CNN-R1",
        description="Baseline — AdamW + cosine + BatchNorm + ReLU + Focal(γ=2) + MaxPool + SpecAugment",
        channels=[256, 512],
        activation="relu",
        norm_type="batch",
        loss_fn="focal", focal_gamma=2.0,
        optimizer="adamw", lr=5e-4, weight_decay=1e-4,
        lr_scheduler="cosine",
        batch_size=32, dropout=0.3,
        spec_augment=True, mixed_precision=False,
        pool_type="adaptive_avg",
        init_scheme="kaiming",
        warmup_epochs=2, grad_clip=1.0,
        epochs=30, patience=7,
    )

    # ── CNN-R2: Activation ablation ───────────────────────────────────────
    runs["CNN-R2a"] = deepcopy(runs["CNN-R1"])
    runs["CNN-R2a"].run_id = "CNN-R2a"
    runs["CNN-R2a"].description = "Activation ablation — GELU instead of ReLU"
    runs["CNN-R2a"].activation = "gelu"

    runs["CNN-R2b"] = deepcopy(runs["CNN-R1"])
    runs["CNN-R2b"].run_id = "CNN-R2b"
    runs["CNN-R2b"].description = "Activation ablation — LeakyReLU(0.1) instead of ReLU"
    runs["CNN-R2b"].activation = "leaky_relu"
    runs["CNN-R2b"].leaky_slope = 0.1

    # ── CNN-R3: Loss function ablation ────────────────────────────────────
    runs["CNN-R3a"] = deepcopy(runs["CNN-R1"])
    runs["CNN-R3a"].run_id = "CNN-R3a"
    runs["CNN-R3a"].description = "Loss ablation — Cross-Entropy + label_smoothing=0.1"
    runs["CNN-R3a"].loss_fn = "ce_smooth"
    runs["CNN-R3a"].label_smooth = 0.1

    runs["CNN-R3b"] = deepcopy(runs["CNN-R1"])
    runs["CNN-R3b"].run_id = "CNN-R3b"
    runs["CNN-R3b"].description = "Loss ablation — Focal loss γ=1.0 (milder penalty)"
    runs["CNN-R3b"].loss_fn = "focal"
    runs["CNN-R3b"].focal_gamma = 1.0

    runs["CNN-R3c"] = deepcopy(runs["CNN-R1"])
    runs["CNN-R3c"].run_id = "CNN-R3c"
    runs["CNN-R3c"].description = "Loss ablation — Vanilla cross-entropy (no focal, no smoothing)"
    runs["CNN-R3c"].loss_fn = "ce"

    # ── CNN-R4: Optimizer ablation ────────────────────────────────────────
    runs["CNN-R4a"] = deepcopy(runs["CNN-R1"])
    runs["CNN-R4a"].run_id = "CNN-R4a"
    runs["CNN-R4a"].description = "Optimizer ablation — SGD + momentum=0.9 + cosine"
    runs["CNN-R4a"].optimizer = "sgd"
    runs["CNN-R4a"].lr = 1e-2
    runs["CNN-R4a"].momentum = 0.9

    runs["CNN-R4b"] = deepcopy(runs["CNN-R1"])
    runs["CNN-R4b"].run_id = "CNN-R4b"
    runs["CNN-R4b"].description = "Optimizer ablation — Adam (no weight decay correction)"
    runs["CNN-R4b"].optimizer = "adam"

    # ── CNN-R5: Normalization ablation ────────────────────────────────────
    runs["CNN-R5a"] = deepcopy(runs["CNN-R1"])
    runs["CNN-R5a"].run_id = "CNN-R5a"
    runs["CNN-R5a"].description = "Normalization — GroupNorm(32 groups)"
    runs["CNN-R5a"].norm_type = "group"
    runs["CNN-R5a"].group_norm_groups = 32

    runs["CNN-R5b"] = deepcopy(runs["CNN-R1"])
    runs["CNN-R5b"].run_id = "CNN-R5b"
    runs["CNN-R5b"].description = "Normalization — No normalization layer"
    runs["CNN-R5b"].norm_type = "none"

    # ── CNN-R6: Architecture ablation ─────────────────────────────────────
    runs["CNN-R6a"] = deepcopy(runs["CNN-R1"])
    runs["CNN-R6a"].run_id = "CNN-R6a"
    runs["CNN-R6a"].description = "Architecture — Deeper (3 conv blocks: 128→256→512)"
    runs["CNN-R6a"].channels = [128, 256, 512]

    runs["CNN-R6b"] = deepcopy(runs["CNN-R1"])
    runs["CNN-R6b"].run_id = "CNN-R6b"
    runs["CNN-R6b"].description = "Architecture — Shallower (1 conv block: 512)"
    runs["CNN-R6b"].channels = [512]

    runs["CNN-R6c"] = deepcopy(runs["CNN-R1"])
    runs["CNN-R6c"].run_id = "CNN-R6c"
    runs["CNN-R6c"].description = "Architecture — Wider (2 blocks: 512→1024), larger linear"
    runs["CNN-R6c"].channels = [512, 1024]
    runs["CNN-R6c"].hidden_dim = 256

    # ── CNN-R7: Augmentation ablation ─────────────────────────────────────
    runs["CNN-R7a"] = deepcopy(runs["CNN-R1"])
    runs["CNN-R7a"].run_id = "CNN-R7a"
    runs["CNN-R7a"].description = "Augmentation — No SpecAugment (baseline for ablation)"
    runs["CNN-R7a"].spec_augment = False

    runs["CNN-R7b"] = deepcopy(runs["CNN-R1"])
    runs["CNN-R7b"].run_id = "CNN-R7b"
    runs["CNN-R7b"].description = "Augmentation — Aggressive SpecAugment (4 time + 4 freq masks)"
    runs["CNN-R7b"].spec_time_masks = 4
    runs["CNN-R7b"].spec_freq_masks = 4
    runs["CNN-R7b"].spec_time_max = 60

    # ── CNN-R8: Regularization ablation ───────────────────────────────────
    runs["CNN-R8a"] = deepcopy(runs["CNN-R1"])
    runs["CNN-R8a"].run_id = "CNN-R8a"
    runs["CNN-R8a"].description = "Regularization — Low dropout=0.1"
    runs["CNN-R8a"].dropout = 0.1

    runs["CNN-R8b"] = deepcopy(runs["CNN-R1"])
    runs["CNN-R8b"].run_id = "CNN-R8b"
    runs["CNN-R8b"].description = "Regularization — High dropout=0.5 + high weight_decay=1e-3"
    runs["CNN-R8b"].dropout = 0.5
    runs["CNN-R8b"].weight_decay = 1e-3

    # ── CNN-R9: Pooling strategy ablation ─────────────────────────────────
    runs["CNN-R9a"] = deepcopy(runs["CNN-R1"])
    runs["CNN-R9a"].run_id = "CNN-R9a"
    runs["CNN-R9a"].description = "Pooling — Adaptive MaxPool (takes peak activation per channel)"
    runs["CNN-R9a"].pool_type = "adaptive_max"

    runs["CNN-R9b"] = deepcopy(runs["CNN-R1"])
    runs["CNN-R9b"].run_id = "CNN-R9b"
    runs["CNN-R9b"].description = "Pooling — AdaptiveAvgPool(4) keeps 4 time steps before flatten"
    runs["CNN-R9b"].pool_type = "adaptive_avg4"

    # ── CNN-R10: LR scheduler ablation ────────────────────────────────────
    runs["CNN-R10a"] = deepcopy(runs["CNN-R1"])
    runs["CNN-R10a"].run_id = "CNN-R10a"
    runs["CNN-R10a"].description = "Scheduler — OneCycleLR (fast ramp, aggressive cosine)"
    runs["CNN-R10a"].lr_scheduler = "onecycle"

    runs["CNN-R10b"] = deepcopy(runs["CNN-R1"])
    runs["CNN-R10b"].run_id = "CNN-R10b"
    runs["CNN-R10b"].description = "Scheduler — StepLR (decay by 0.5 every 10 epochs)"
    runs["CNN-R10b"].lr_scheduler = "step"

    runs["CNN-R10c"] = deepcopy(runs["CNN-R1"])
    runs["CNN-R10c"].run_id = "CNN-R10c"
    runs["CNN-R10c"].description = "Scheduler — No scheduler (constant LR)"
    runs["CNN-R10c"].lr_scheduler = "none"

    # ── CNN-R11: Weight initialization ────────────────────────────────────
    runs["CNN-R11a"] = deepcopy(runs["CNN-R1"])
    runs["CNN-R11a"].run_id = "CNN-R11a"
    runs["CNN-R11a"].description = "Init — Xavier uniform (fan-in/fan-out symmetric)"
    runs["CNN-R11a"].init_scheme = "xavier"

    runs["CNN-R11b"] = deepcopy(runs["CNN-R1"])
    runs["CNN-R11b"].run_id = "CNN-R11b"
    runs["CNN-R11b"].description = "Init — PyTorch default (uniform scaling by fan-in)"
    runs["CNN-R11b"].init_scheme = "default"

    # ── CNN-R12: Mixed precision ───────────────────────────────────────────
    runs["CNN-R12"] = deepcopy(runs["CNN-R1"])
    runs["CNN-R12"].run_id = "CNN-R12"
    runs["CNN-R12"].description = "Mixed Precision — torch.amp float16 (speed test)"
    runs["CNN-R12"].mixed_precision = True

    # ── CNN-R13: MixUp augmentation ───────────────────────────────────────
    # From learn/04_lab_dl_with_limited_data/01_augmentations.ipynb (Zhang et al., 2018)
    # Linearly blends two mel-spectrograms: mel = λ*mel_a + (1-λ)*mel_b
    # Loss = λ*CE(logits, ya) + (1-λ)*CE(logits, yb)
    runs["CNN-R13"] = deepcopy(runs["CNN-R1"])
    runs["CNN-R13"].run_id = "CNN-R13"
    runs["CNN-R13"].description = "MixUp aug (α=0.4) — smooth label interpolation between samples"
    runs["CNN-R13"].use_mixup = True
    runs["CNN-R13"].mixup_alpha = 0.4   # matches notebook demo value

    # ── CNN-R14: CutMix augmentation ──────────────────────────────────────
    # From learn/04_lab_dl_with_limited_data/01_augmentations.ipynb (Yun et al., 2019)
    # Pastes a freq×time patch from one mel-spec onto another; proportional label mixing
    runs["CNN-R14"] = deepcopy(runs["CNN-R1"])
    runs["CNN-R14"].run_id = "CNN-R14"
    runs["CNN-R14"].description = "CutMix aug (α=1.0) — paste freq×time patch; proportional label mix"
    runs["CNN-R14"].use_cutmix = True
    runs["CNN-R14"].cutmix_alpha = 1.0  # uniform patch area distribution

    # ── CNN-R15: MixUp + CutMix combined ──────────────────────────────────
    # Enables both; run_epoch applies MixUp (use_mixup checked first).
    # Demonstrates stacking multiple augmentation strategies from the course.
    runs["CNN-R15"] = deepcopy(runs["CNN-R1"])
    runs["CNN-R15"].run_id = "CNN-R15"
    runs["CNN-R15"].description = "MixUp+CutMix combined — aggressive label-mixing regularisation"
    runs["CNN-R15"].use_mixup = True
    runs["CNN-R15"].mixup_alpha = 0.4
    runs["CNN-R15"].use_cutmix = True
    runs["CNN-R15"].cutmix_alpha = 1.0

    # ── CNN-R16: BEST-COMBO run ────────────────────────────────────────────
    # Combines the best individual settings found across all single-ingredient ablations:
    #   R3c  → vanilla CE (best loss)
    #   R2a  → GELU activation
    #   R6a  → 3-block deeper architecture
    #   R8a  → low dropout=0.1 (avoids over-regularization on small data)
    #   R9b  → AdaptiveAvgPool(4) retains temporal structure
    #   R10a → OneCycleLR (aggressive ramp for small datasets)
    #   R13  → MixUp (course notebook technique, new in this script)
    # This demonstrates that the best set is not found by any single-ingredient
    # ablation alone — it requires combining the winners (hyperparameter tuning).
    runs["CNN-R16"] = deepcopy(runs["CNN-R1"])
    runs["CNN-R16"].run_id = "CNN-R16"
    runs["CNN-R16"].description = "Best-combo: GELU+3-block+CE+dropout=0.1+AvgPool4+OneCycle+MixUp"
    runs["CNN-R16"].activation    = "gelu"
    runs["CNN-R16"].channels      = [128, 256, 512]   # 3-block deeper architecture
    runs["CNN-R16"].loss_fn       = "ce"              # vanilla CE won in R3c
    runs["CNN-R16"].dropout       = 0.1               # low dropout (R8a winner)
    runs["CNN-R16"].pool_type     = "adaptive_avg4"   # temporal structure (R9b)
    runs["CNN-R16"].lr_scheduler  = "onecycle"        # fast ramp for small data (R10a)
    runs["CNN-R16"].use_mixup     = True              # MixUp from course (R13)
    runs["CNN-R16"].mixup_alpha   = 0.4
    runs["CNN-R16"].epochs        = 40                # more epochs — OneCycle handles decay
    runs["CNN-R16"].patience      = 10

    return runs


ALL_RUNS = _make_runs()


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATASET  (model classes live in src/models/flexible_cnn.py)
# ══════════════════════════════════════════════════════════════════════════════

class AblationAudioDataset(Dataset):
    """
    Loads mel-spectrogram tensors for ablation training.
    Supports SpecAugment (14) and Gaussian noise augmentation.
    """

    def __init__(
        self,
        jsonl_path: Path,
        is_train: bool = False,
        cfg: CNNAblationConfig | None = None,
        max_time_frames: int = 313,   # 30s at 100fps Whisper
    ):
        self.is_train = is_train
        self.cfg = cfg
        self.max_T = max_time_frames
        self.samples: list[dict] = []
        self._rng = np.random.default_rng(SEED)

        for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            audio_path = Path(rec.get("audio_filename", ""))
            if not audio_path.exists():
                continue
            emotion_str = rec.get("emotion_text", "")
            eid = encode_emotion(emotion_str)
            if eid == -1:
                continue
            self.samples.append({"path": str(audio_path), "label": eid})

        logger.info("AblationAudioDataset: {} samples from {}", len(self.samples), jsonl_path.name)

    def _load_mel(self, path: str) -> np.ndarray:
        y, _ = librosa.load(
            path,
            sr=SAMPLING_RATE,
            mono=True,
            duration=MAX_AUDIO_SEC,
            res_type="polyphase",
        )
        mel = librosa.feature.melspectrogram(
            y=y, sr=SAMPLING_RATE, n_mels=N_MELS,
            n_fft=1024, hop_length=512, fmax=8000,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db  # (128, T)

    def _spec_augment(self, mel: np.ndarray) -> np.ndarray:
        """Ingredient 14: SpecAugment [Park et al., 2019]."""
        cfg = self.cfg
        if cfg is None or not cfg.spec_augment:
            return mel
        x = mel.copy()
        n_mels, T = x.shape
        for _ in range(cfg.spec_freq_masks):
            f = int(self._rng.integers(0, cfg.spec_freq_max + 1))
            f0 = int(self._rng.integers(0, max(1, n_mels - f)))
            x[f0:f0 + f, :] = 0.0
        for _ in range(cfg.spec_time_masks):
            t = int(self._rng.integers(0, cfg.spec_time_max + 1))
            t0 = int(self._rng.integers(0, max(1, T - t)))
            x[:, t0:t0 + t] = 0.0
        return x

    def _pad_or_trim(self, mel: np.ndarray) -> np.ndarray:
        n_mels, T = mel.shape
        if T >= self.max_T:
            return mel[:, :self.max_T]
        return np.pad(mel, ((0, 0), (0, self.max_T - T)), mode="constant")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        mel = self._load_mel(s["path"])

        if self.is_train and self.cfg is not None:
            mel = self._spec_augment(mel)
            # Gaussian noise (14)
            if self.cfg.gaussian_noise_std > 0:
                mel = mel + self._rng.normal(0, self.cfg.gaussian_noise_std, mel.shape)

        mel = self._pad_or_trim(mel)
        return {
            "mel":   torch.from_numpy(mel).float(),
            "label": torch.tensor(s["label"], dtype=torch.long),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4b. MIXUP AND CUTMIX AUGMENTATION (Ingredient 14 — from course notebooks)
#
# These are the audio-spectrogram adaptations of the image MixUp / CutMix
# techniques studied in:
#   learn/04_lab_dl_with_limited_data/01_augmentations.ipynb
#
# MixUp (Zhang et al., 2018):
#   λ ~ Beta(α, α)
#   mel_mixed  = λ * mel_a + (1-λ) * mel_b
#   loss_mixed = λ * CE(logits, ya) + (1-λ) * CE(logits, yb)
#
# CutMix (Yun et al., 2019) — audio adaptation:
#   Cut a random (freq-band × time-span) rectangle from mel_b,
#   paste it onto mel_a.  Label mixing proportional to cut area.
# ══════════════════════════════════════════════════════════════════════════════

def mixup_batch(
    mel: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    MixUp augmentation for a batch of mel-spectrograms.

    Args:
        mel:    (B, 128, T) batch of mel-spectrograms
        labels: (B,) integer class labels
        alpha:  Beta distribution concentration parameter
        device: target device

    Returns:
        mel_mixed:  (B, 128, T) mixed spectrograms
        labels_a:   original labels (for mixed loss calculation)
        labels_b:   shuffled labels (for mixed loss calculation)
        lam:        mixing coefficient λ
    """
    lam = float(np.random.beta(alpha, alpha))
    batch_size = mel.size(0)
    idx = torch.randperm(batch_size, device=device)
    mel_mixed  = lam * mel + (1 - lam) * mel[idx]
    return mel_mixed, labels, labels[idx], lam


def cutmix_batch(
    mel: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    CutMix augmentation for a batch of mel-spectrograms.

    Cuts a random freq × time rectangle from the shuffled sample and
    pastes it onto the original.  λ is adjusted by the actual cut area.

    Args:
        mel:    (B, 128, T) batch of mel-spectrograms
        labels: (B,) integer class labels
        alpha:  Beta distribution concentration (controls patch size)
        device: target device

    Returns:
        mel_cut:  (B, 128, T) augmented spectrograms
        labels_a: original labels
        labels_b: shuffled labels
        lam:      effective mixing ratio (1 - cut_area / total_area)
    """
    lam_init = float(np.random.beta(alpha, alpha))
    batch_size = mel.size(0)
    n_mels, T = mel.size(1), mel.size(2)

    # Patch size proportional to sqrt(1 - lam) — matches CutMix paper
    cut_ratio = np.sqrt(1.0 - lam_init)
    cut_f = int(n_mels * cut_ratio)    # frequency dimension of patch
    cut_t = int(T * cut_ratio)         # time dimension of patch

    # Random patch origin (centre-based, clamped to boundaries)
    f_c = np.random.randint(n_mels)
    t_c = np.random.randint(T)
    f1 = max(0, f_c - cut_f // 2)
    f2 = min(n_mels, f_c + cut_f // 2)
    t1 = max(0, t_c - cut_t // 2)
    t2 = min(T, t_c + cut_t // 2)

    # Effective λ after boundary clamping
    lam = 1.0 - (f2 - f1) * (t2 - t1) / (n_mels * T)

    idx = torch.randperm(batch_size, device=device)
    mel_cut = mel.clone()
    mel_cut[:, f1:f2, t1:t2] = mel[idx, f1:f2, t1:t2]
    return mel_cut, labels, labels[idx], lam


def mixed_loss(
    loss_fn: nn.Module,
    logits: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """
    Compute the blended loss for MixUp / CutMix:
        L = λ * L(logits, ya) + (1-λ) * L(logits, yb)
    """
    return lam * loss_fn(logits, labels_a) + (1 - lam) * loss_fn(logits, labels_b)


# ══════════════════════════════════════════════════════════════════════════════
# 5. TRAINING
# ══════════════════════════════════════════════════════════════════════════════

def make_optimizer(cfg: CNNAblationConfig, parameters) -> torch.optim.Optimizer:
    """Ingredient 5: Optimizer factory."""
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "adam":
        return torch.optim.Adam(parameters, lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            parameters, lr=cfg.lr, momentum=cfg.momentum,
            weight_decay=cfg.weight_decay, nesterov=True,
        )
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


def make_scheduler(
    cfg: CNNAblationConfig,
    optimizer: torch.optim.Optimizer,
    n_batches: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Ingredient 7: LR scheduler factory."""
    total_steps = cfg.epochs * n_batches
    warmup_steps = cfg.warmup_epochs * n_batches

    if cfg.lr_scheduler == "cosine":
        from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
        if warmup_steps > 0:
            warm = LinearLR(optimizer, start_factor=1e-4, total_iters=warmup_steps)
            main = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
            return SequentialLR(optimizer, schedulers=[warm, main], milestones=[warmup_steps])
        return CosineAnnealingLR(optimizer, T_max=total_steps)

    if cfg.lr_scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=cfg.lr, total_steps=total_steps,
            pct_start=0.3, anneal_strategy="cos",
        )

    if cfg.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.lr_step_size, gamma=cfg.lr_gamma,
        )

    return None   # constant LR


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    cfg: CNNAblationConfig,
    device: torch.device,
    scaler: GradScaler | None,
    grad_norms: list[float],
) -> tuple[float, float]:
    """One training or evaluation epoch. Returns (loss, macro_f1)."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_preds, all_labels = [], []
    step = 0

    for batch in loader:
        mel    = batch["mel"].to(device)
        labels = batch["label"].to(device)

        # Ingredient 14 — MixUp / CutMix (only during training)
        # Taught in learn/04_lab_dl_with_limited_data/01_augmentations.ipynb
        labels_b: torch.Tensor | None = None
        lam_mix: float = 1.0
        if is_train and cfg.use_mixup and mel.size(0) > 1:
            mel, labels, labels_b, lam_mix = mixup_batch(mel, labels, cfg.mixup_alpha, device)
        elif is_train and cfg.use_cutmix and mel.size(0) > 1:
            mel, labels, labels_b, lam_mix = cutmix_batch(mel, labels, cfg.cutmix_alpha, device)

        with autocast(enabled=cfg.mixed_precision):  # Ingredient 17: Mixed Precision
            logits = model(mel)
            if labels_b is not None:
                loss = mixed_loss(loss_fn, logits, labels, labels_b, lam_mix)
            else:
                loss = loss_fn(logits, labels)

        if is_train:
            # Ingredient 8: Gradient accumulation
            loss = loss / cfg.grad_accum
            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            step += 1
            if step % cfg.grad_accum == 0:
                # Ingredient 15: Gradient clipping (prevents explosion)
                if scaler:
                    scaler.unscale_(optimizer)
                if cfg.grad_clip:
                    gnorm = nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.grad_clip
                    ).item()
                    grad_norms.append(gnorm)

                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

                # Ingredient 7: LR scheduler step (per-step for onecycle)
                if scheduler is not None and cfg.lr_scheduler in ("onecycle", "cosine"):
                    scheduler.step()

        total_loss += loss.item() * cfg.grad_accum * len(labels)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    # Per-epoch step for non per-step schedulers
    if is_train and scheduler and cfg.lr_scheduler == "step":
        scheduler.step()

    avg_loss = total_loss / len(loader.dataset)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, macro_f1


# ══════════════════════════════════════════════════════════════════════════════
# 6. VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════

def _save_curve(
    values_dict: dict[str, list[float]],
    title: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """Generic line plot for training curves."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, vals in values_dict.items():
        ax.plot(range(1, len(vals) + 1), vals, label=label, marker="o", markersize=3)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _save_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    out_path: Path,
    title: str = "Confusion Matrix",
) -> None:
    """Ingredient 13 / Evaluation: Confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(figsize=(14, 11))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=[c.split("(")[0].strip() for c in class_names],
        yticklabels=[c.split("(")[0].strip() for c in class_names],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _save_grad_norm_curve(grad_norms: list[float], out_path: Path, run_id: str) -> None:
    """Ingredient 15 & 28: Gradient norm trace — detect explosion."""
    if not grad_norms:
        return
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(grad_norms, alpha=0.7, linewidth=0.8)
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="clip threshold")
    ax.set_xlabel("Gradient step")
    ax.set_ylabel("L2 Gradient Norm")
    ax.set_title(f"Gradient Norm Trace — {run_id}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 7. MAIN TRAINING LOOP FOR ONE RUN
# ══════════════════════════════════════════════════════════════════════════════

def train_one_run(cfg: CNNAblationConfig, epochs_override: int | None = None) -> dict:
    """
    Full training + evaluation for a single ablation configuration.

    Returns a results dict written to training_history.json.
    """
    if epochs_override:
        cfg.epochs = epochs_override  # override for smoke tests (CNNAblationConfig is mutable dataclass)

    set_seed(SEED)

    run_out   = OUTPUT_ROOT / cfg.run_id
    fig_out   = FIGURE_ROOT / cfg.run_id
    run_out.mkdir(parents=True, exist_ok=True)
    fig_out.mkdir(parents=True, exist_ok=True)

    logger.add(run_out / "train.log", rotation="5 MB")
    logger.info("=== {} === {}", cfg.run_id, cfg.description)
    logger.info("Config: {}", json.dumps(cfg.to_dict(), indent=2))

    # ── 12: Save config for reproducibility ────────────────────
    (run_out / "config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: {}", device)

    # ── Datasets ──────────────────────────────────────────────────────────
    train_ds = AblationAudioDataset(DATA_DIR / "train.jsonl", is_train=True, cfg=cfg)
    val_ds   = AblationAudioDataset(DATA_DIR / "val.jsonl",   is_train=False)
    test_ds  = AblationAudioDataset(DATA_DIR / "test.jsonl",  is_train=False)

    if len(train_ds) == 0:
        logger.error("No training samples found. Run data pipeline first: just generate-data")
        return {}

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        num_workers=0, pin_memory=device.type == "cuda",
    )
    val_loader  = DataLoader(val_ds,  batch_size=64, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=0)

    # ── Class weights (9) ──────────────────────────────────────
    all_labels = [s["label"] for s in train_ds.samples]
    cw = compute_class_weight("balanced", classes=np.arange(N_CLASSES), y=all_labels)
    class_weights = torch.tensor(cw, dtype=torch.float, device=device)

    # ── Model (ingredients 1, 2, 3, 10, 11, 22) ──────────────────────────
    model = FlexibleEmotionCNN(cfg, num_classes=N_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Trainable params: {:,} ({:.2f}M)", n_params, n_params / 1e6)
    assert n_params < 50_000_000, f"Model exceeds 50M param limit: {n_params:,}"

    # Dummy forward pass sanity check
    dummy = torch.randn(2, N_MELS, 313, device=device)
    with torch.no_grad():
        out = model(dummy)
    assert out.shape == (2, N_CLASSES), f"Unexpected output shape: {out.shape}"
    logger.info("Forward pass OK — output shape: {}", out.shape)

    # ── Loss (4) ───────────────────────────────────────────────
    loss_fn = make_loss_fn(cfg, class_weights)

    # ── Optimizer (5) ──────────────────────────────────────────
    optimizer = make_optimizer(cfg, model.parameters())

    # ── Scheduler (ingredients 6, 7, 18) ─────────────────────────────────
    scheduler = make_scheduler(cfg, optimizer, n_batches=len(train_loader))

    # ── Mixed Precision Scaler (17) ────────────────────────────
    scaler = GradScaler() if cfg.mixed_precision and device.type == "cuda" else None

    # ── Early stopping (13) ───────────────────────────────────
    early_stopper = EarlyStopper(cfg.patience, run_out, cfg.run_id)
    best_val_f1 = -1.0
    best_ckpt_path = early_stopper.best_ckpt

    # ── Training state ────────────────────────────────────────────────────
    history: list[dict] = []
    train_losses, val_losses = [], []
    train_f1s, val_f1s      = [], []
    grad_norms: list[float] = []

    logger.info("Starting training: {} epochs, batch={}, lr={}", cfg.epochs, cfg.batch_size, cfg.lr)
    start_time = time.time()

    for epoch in range(1, cfg.epochs + 1):
        with logger.contextualize(epoch=epoch, run=cfg.run_id):
            # ── Forward + Backward ────────────────────────────────────────
            tr_loss, tr_f1 = run_epoch(
                model, train_loader, loss_fn, optimizer, scheduler, cfg, device, scaler, grad_norms
            )
            val_loss, val_f1 = run_epoch(
                model, val_loader, loss_fn, None, None, cfg, device, None, []
            )

            train_losses.append(tr_loss)
            val_losses.append(val_loss)
            train_f1s.append(tr_f1)
            val_f1s.append(val_f1)

            row = {
                "epoch": epoch, "train_loss": round(tr_loss, 4),
                "val_loss": round(val_loss, 4), "train_f1": round(tr_f1, 4),
                "val_f1": round(val_f1, 4),
            }
            history.append(row)
            logger.info(
                "Ep {:02d} | tr_loss={:.4f} tr_f1={:.4f} | val_loss={:.4f} val_f1={:.4f}",
                epoch, tr_loss, tr_f1, val_loss, val_f1,
            )

            # ── Checkpointing (12) ─────────────────────────────
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1

            # ── Early Stopping (13) ────────────────────────────
            if early_stopper.step(val_f1, model):
                logger.info("Early stop at epoch {} — no val_f1 improvement for {} epochs",
                            epoch, cfg.patience)
                break

    elapsed = time.time() - start_time
    logger.info("Training done in {:.1f}s", elapsed)

    # ── Test evaluation ───────────────────────────────────────────────────
    model.load_state_dict(torch.load(best_ckpt_path, map_location=device))
    _, test_f1 = run_epoch(
        model, test_loader, loss_fn, None, None, cfg, device, None, []
    )

    # Collect test predictions for confusion matrix
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            logits = model(batch["mel"].to(device))
            y_pred.extend(logits.argmax(1).cpu().numpy())
            y_true.extend(batch["label"].numpy())

    logger.info("Test macro-F1: {:.4f}", test_f1)

    # ── Visualizations ────────────────────────────────────────────────────
    _save_curve(
        {"train": train_losses, "val": val_losses},
        f"Loss Curve — {cfg.run_id}", "Loss",
        fig_out / "loss_curve.png",
    )
    _save_curve(
        {"train": train_f1s, "val": val_f1s},
        f"Macro-F1 Curve — {cfg.run_id}", "Macro-F1",
        fig_out / "f1_curve.png",
    )
    _save_confusion_matrix(
        y_true, y_pred, EMOTION_CLASSES,
        fig_out / "confusion_matrix.png",
        title=f"Test Confusion Matrix — {cfg.run_id}",
    )
    _save_grad_norm_curve(grad_norms, fig_out / "grad_norm_curve.png", cfg.run_id)

    # ── Save results ──────────────────────────────────────────────────────
    result = {
        "run_id":      cfg.run_id,
        "description": cfg.description,
        "n_params":    n_params,
        "best_val_f1": round(best_val_f1, 4),
        "test_f1":     round(test_f1, 4),
        "epochs_run":  len(history),
        "elapsed_s":   round(elapsed, 1),
        "history":     history,
        "config":      cfg.to_dict(),
    }
    (run_out / "training_history.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Results saved to {}", run_out)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 8. COMPARISON SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

def compile_ablation_summary() -> None:
    """Collect all completed run results into one summary JSON + table."""
    results = []
    for run_id in ALL_RUNS:
        history_path = OUTPUT_ROOT / run_id / "training_history.json"
        if history_path.exists():
            r = json.loads(history_path.read_text())
            results.append({
                "run_id":      r["run_id"],
                "description": r["description"],
                "n_params":    r.get("n_params", "?"),
                "val_f1":      r.get("best_val_f1", "?"),
                "test_f1":     r.get("test_f1", "?"),
                "epochs_run":  r.get("epochs_run", "?"),
            })

    summary_path = OUTPUT_ROOT / "ablation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("CNN ABLATION SUMMARY")
    print("=" * 70)
    header = f"{'Run':<12} {'Val F1':>8} {'Test F1':>8} {'Epochs':>7}  Description"
    print(header)
    print("-" * 70)
    for r in sorted(results, key=lambda x: x["test_f1"] if isinstance(x["test_f1"], float) else -1, reverse=True):
        desc = r["description"][:50] + "…" if len(r["description"]) > 50 else r["description"]
        print(f"{r['run_id']:<12} {str(r['val_f1']):>8} {str(r['test_f1']):>8} "
              f"{str(r['epochs_run']):>7}  {desc}")
    print(f"\nSummary saved to {summary_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 9. CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CNN ablation study — trains one or all ablation configurations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--run", default="CNN-R1",
        help="Run ID to train (e.g. CNN-R1, CNN-R2a) or 'all' to run every config.",
    )
    p.add_argument(
        "--epochs", type=int, default=None,
        help="Override epochs from config (useful for smoke tests).",
    )
    p.add_argument(
        "--list", action="store_true",
        help="List all available run IDs and descriptions, then exit.",
    )
    p.add_argument(
        "--summary", action="store_true",
        help="Compile and print summary of completed runs, then exit.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.list:
        print(f"\n{'Run ID':<12} Description")
        print("-" * 70)
        for rid, cfg in ALL_RUNS.items():
            print(f"{rid:<12} {cfg.description}")
        return

    if args.summary:
        compile_ablation_summary()
        return

    if args.run.lower() == "all":
        for run_id, cfg in ALL_RUNS.items():
            try:
                train_one_run(cfg, epochs_override=args.epochs)
            except Exception as e:
                logger.error("Run {} failed: {}", run_id, e)
        compile_ablation_summary()
    else:
        run_id = args.run
        if run_id not in ALL_RUNS:
            logger.error("Unknown run '{}'. Use --list to see available runs.", run_id)
            sys.exit(1)
        train_one_run(ALL_RUNS[run_id], epochs_override=args.epochs)


if __name__ == "__main__":
    main()

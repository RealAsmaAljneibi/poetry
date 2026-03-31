"""
scripts/train_arousal.py

Trains a from-scratch MLP to predict vocal Arousal (Low / Medium / High)
from librosa audio features extracted from each .mp3 clip.

Why Arousal (not 12-class emotion):
  Audio reliably encodes delivery energy but NOT fine-grained semantic emotion.
  3-class Arousal is near-perfectly balanced (33/31/36%) and maps to audible
  acoustic properties (energy, tempo, pitch range, pause ratio).

Course techniques applied:
  Week 2 — LR warmup + cosine decay
  Week 3 — Class-weighted CrossEntropy, early stopping, label smoothing
  Week 4 — Gradient accumulation, BatchNorm (GroupNorm for small batches)
  Week 5 — TensorBoard, Pydantic config, sanity checks, loguru logging

Feature vector (34 dimensions):
  13 MFCC means + 13 MFCC stds  = 26
  RMS energy mean + std          =  2
  Zero-crossing rate mean        =  1
  Spectral centroid mean + std   =  2
  Spectral rolloff mean          =  1
  Tempo (BPM)                    =  1
  Pause ratio                    =  1

Usage:
    uv run python scripts/train_arousal.py
    uv run python scripts/train_arousal.py --epochs 100 --lr 5e-4
    uv run python scripts/train_arousal.py --no-cache   # recompute features
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from loguru import logger
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import arousal_config
from src.data.arousal_labels import AROUSAL_CLASSES, ID2AROUSAL, encode_arousal
from src.training.trainer import EarlyStopper, TensorBoardLogger, get_scheduler, set_seed
from src.training.sanity import check_no_nan  # check_no_nan(tensor, name)

# ── Constants ─────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
FEATURE_CACHE = PROJECT_ROOT / "data/processed/arousal_features.pkl"
OUTPUT_DIR    = PROJECT_ROOT / "outputs/models/arousal_mlp"
FIGURE_DIR    = PROJECT_ROOT / "outputs/figures"

N_CLASSES = len(AROUSAL_CLASSES)   # 3


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(audio_path: str, sr: int = 16000, max_sec: int = 30, n_mfcc: int = 13) -> np.ndarray | None:
    """
    Extract a 34-dimensional feature vector from one audio clip.

    Returns None if the file is missing or unreadable.
    All features are computed from the raw waveform — no learned components.
    """
    try:
        y, _ = librosa.load(audio_path, sr=sr, mono=True, duration=max_sec)
    except Exception as e:
        logger.warning(f"Cannot load {audio_path}: {e}")
        return None

    if len(y) < sr * 0.1:   # shorter than 100ms → skip
        return None

    feats: list[float] = []

    # 1. MFCCs (13 means + 13 stds = 26 features)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    feats.extend(mfcc.mean(axis=1).tolist())
    feats.extend(mfcc.std(axis=1).tolist())

    # 2. RMS energy mean + std (2 features)
    rms = librosa.feature.rms(y=y)[0]
    feats.append(float(rms.mean()))
    feats.append(float(rms.std()))

    # 3. Zero-crossing rate mean (1 feature) — correlates with consonant density / breathiness
    zcr = librosa.feature.zero_crossing_rate(y=y)[0]
    feats.append(float(zcr.mean()))

    # 4. Spectral centroid mean + std (2 features) — proxy for brightness/pitch centre
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    feats.append(float(centroid.mean()))
    feats.append(float(centroid.std()))

    # 5. Spectral rolloff mean (1 feature) — where 85% of energy is concentrated
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    feats.append(float(rolloff.mean()))

    # 6. Tempo (1 feature) — delivery speed
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    feats.append(float(np.squeeze(tempo)))

    # 7. Pause ratio (1 feature) — proportion of frames with very low energy (silence/breath pauses)
    silence_threshold = rms.mean() * 0.1
    pause_frames = (rms < silence_threshold).sum()
    feats.append(float(pause_frames) / max(len(rms), 1))

    return np.array(feats, dtype=np.float32)   # shape: (34,)


def build_feature_cache(jsonl_path: Path, sr: int, max_sec: int, n_mfcc: int) -> dict[str, np.ndarray]:
    """Extract features for every clip in a split; returns {audio_filename: features}."""
    cache: dict[str, np.ndarray] = {}
    rows = [json.loads(line) for line in open(jsonl_path)]
    logger.info(f"Extracting features from {len(rows)} clips in {jsonl_path.name} ...")
    for i, row in enumerate(rows):
        path = row.get("audio_filename", "")
        feats = extract_features(path, sr=sr, max_sec=max_sec, n_mfcc=n_mfcc)
        if feats is not None:
            cache[path] = feats
        if (i + 1) % 200 == 0:
            logger.info(f"  {i+1}/{len(rows)} done ...")
    logger.info(f"Feature extraction done: {len(cache)}/{len(rows)} clips successful")
    return cache


def load_split(
    jsonl_path: Path,
    feature_cache: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Load (X, y) arrays for a split using the pre-built feature cache."""
    X_rows, y_rows = [], []
    skipped = 0
    for line in open(jsonl_path):
        row   = json.loads(line)
        path  = row.get("audio_filename", "")
        label = encode_arousal(row.get("emotion_audio"))
        feats = feature_cache.get(path)
        if feats is None or label == -1:
            skipped += 1
            continue
        X_rows.append(feats)
        y_rows.append(label)
    logger.info(f"{jsonl_path.name}: {len(X_rows)} samples loaded, {skipped} skipped (no audio/label)")
    return np.array(X_rows, dtype=np.float32), np.array(y_rows, dtype=np.int64)


# ── MLP Model ─────────────────────────────────────────────────────────────────

class ArousalMLP(nn.Module):
    """
    Small MLP for 3-class Arousal prediction from 34-dim acoustic features.

    Architecture: Linear → BN → ReLU → Dropout (×n_layers) → Linear(3)
    Parameter count ≈ 12,000–25,000 — well under the 50M scratch-model limit.

    BatchNorm1d is used because features are 1-dimensional (not images).
    For batch_size < 16 we would use GroupNorm, but batch_size=64 here.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_layers: int, dropout: float, n_classes: int = 3):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Training / Eval loops ─────────────────────────────────────────────────────

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    device: torch.device,
    tb_logger: TensorBoardLogger,
    global_step: int,
    log_every: int,
) -> tuple[float, float, int]:
    model.train()
    total_loss, n_correct, n_total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds       = logits.detach().argmax(dim=-1)
        n_correct  += (preds == y_batch).sum().item()
        n_total    += y_batch.size(0)
        global_step += 1

        if global_step % log_every == 0:
            tb_logger.log_step(loss.item(), scheduler.get_last_lr()[0], global_step)

    return total_loss / max(len(loader), 1), n_correct / max(n_total, 1), global_step


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, list, list]:
    model.eval()
    total_loss, n_correct, n_total = 0.0, 0, 0
    all_preds, all_true = [], []

    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            total_loss += loss.item()
            preds       = logits.argmax(dim=-1)
            n_correct  += (preds == y_batch).sum().item()
            n_total    += y_batch.size(0)
            all_preds.extend(preds.cpu().tolist())
            all_true.extend(y_batch.cpu().tolist())

    val_f1  = f1_score(all_true, all_preds, average="macro", zero_division=0)
    val_acc = n_correct / max(n_total, 1)
    return total_loss / max(len(loader), 1), val_f1, val_acc, all_preds, all_true


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    log_path = PROJECT_ROOT / "logs/train_arousal.log"
    logger.add(str(log_path), rotation="10 MB", retention="7 days", level="DEBUG")
    logger.info("=" * 60)
    logger.info("Arousal MLP — training from scratch")
    logger.info("=" * 60)

    cfg = arousal_config()
    if args.epochs:
        cfg = cfg.model_copy(update={"epochs": args.epochs})
    if args.lr:
        cfg = cfg.model_copy(update={"learning_rate": args.lr})

    set_seed(cfg.seed)
    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device} | seed={cfg.seed}")

    # ── Feature cache ─────────────────────────────────────────────────────────
    use_cache = FEATURE_CACHE.exists() and not args.no_cache
    if use_cache:
        logger.info(f"Loading feature cache from {FEATURE_CACHE}")
        with open(FEATURE_CACHE, "rb") as f:
            feature_cache = pickle.load(f)
    else:
        logger.info("Building feature cache (this takes ~5 minutes for 3340 clips) ...")
        feature_cache: dict[str, np.ndarray] = {}
        for split_path in [cfg.train_jsonl, cfg.val_jsonl, cfg.test_jsonl]:
            feature_cache.update(
                build_feature_cache(split_path, cfg.sample_rate, cfg.max_audio_sec, cfg.n_mfcc)
            )
        FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with open(FEATURE_CACHE, "wb") as f:
            pickle.dump(feature_cache, f)
        logger.info(f"Feature cache saved: {len(feature_cache)} clips → {FEATURE_CACHE}")

    # ── Load splits ───────────────────────────────────────────────────────────
    X_train, y_train = load_split(cfg.train_jsonl, feature_cache)
    X_val,   y_val   = load_split(cfg.val_jsonl,   feature_cache)
    X_test,  y_test  = load_split(cfg.test_jsonl,  feature_cache)

    # ── Scaler (fit on train only — no leakage) ───────────────────────────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    scaler_path = OUTPUT_DIR / "arousal_scaler.pkl"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    logger.info(f"Scaler saved: {scaler_path}")

    input_dim = X_train.shape[1]
    logger.info(f"Feature dim: {input_dim} | train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    # ── Class distribution ────────────────────────────────────────────────────
    from collections import Counter
    dist = Counter(y_train.tolist())
    logger.info("Train Arousal distribution:")
    for k, v in sorted(dist.items()):
        logger.info(f"  {ID2AROUSAL[k]:8s}: {v:5d}  ({v/len(y_train)*100:.1f}%)")

    # ── DataLoaders ───────────────────────────────────────────────────────────
    def make_loader(X, y, shuffle=False):
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32),
                           torch.tensor(y, dtype=torch.long))
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle, num_workers=0)

    train_loader = make_loader(X_train, y_train, shuffle=True)
    val_loader   = make_loader(X_val,   y_val)
    test_loader  = make_loader(X_test,  y_test)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = ArousalMLP(
        input_dim=input_dim,
        hidden_dim=cfg.hidden_dim,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        n_classes=N_CLASSES,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"ArousalMLP: {n_params:,} parameters (limit: 50M scratch model)")
    assert n_params < 50_000_000, f"Scratch model exceeds 50M: {n_params}"

    # ── Loss (balanced, no explicit class weights needed — already balanced) ──
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    optimizer    = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    total_steps  = len(train_loader) * cfg.epochs
    scheduler    = get_scheduler(optimizer, total_steps, cfg.warmup_ratio)

    logger.info(f"Optimizer: AdamW lr={cfg.learning_rate:.2e} wd={cfg.weight_decay}")
    logger.info(f"Scheduler: warmup+cosine | total_steps={total_steps}")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    X_dummy = torch.tensor(X_train[:8], dtype=torch.float32).to(device)
    y_dummy = torch.tensor(y_train[:8], dtype=torch.long).to(device)
    with torch.no_grad():
        logits_dummy = model(X_dummy)
        init_loss = torch.nn.functional.cross_entropy(logits_dummy, y_dummy).item()
    expected = float(np.log(N_CLASSES))
    logger.info(f"Sanity: initial loss={init_loss:.4f}  expected≈log({N_CLASSES})={expected:.4f}  "
                f"({'OK' if abs(init_loss - expected) < 1.0 else 'WARNING: far from expected'})")
    check_no_nan(logits_dummy, "initial_logits")
    logger.info("Sanity checks passed.")

    # ── TensorBoard ───────────────────────────────────────────────────────────
    tb_logger = TensorBoardLogger(cfg.tensorboard_dir, "arousal_mlp", "arousal")

    # ── Training loop ─────────────────────────────────────────────────────────
    early_stopper = EarlyStopper(cfg.patience, OUTPUT_DIR, "arousal_mlp_arousal")
    global_step   = 0
    best_val_f1   = 0.0

    logger.info(f"Starting training: {cfg.epochs} epochs")
    for epoch in range(cfg.epochs):
        epoch_logger = logger.bind(epoch=epoch + 1, total_epochs=cfg.epochs, task="arousal")

        train_loss, train_acc, global_step = train_epoch(
            model, train_loader, optimizer, scheduler, criterion,
            device, tb_logger, global_step, cfg.log_every_n_steps,
        )
        val_loss, val_f1, val_acc, val_preds, val_true = eval_epoch(
            model, val_loader, criterion, device
        )
        tb_logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_f1, val_acc)

        epoch_logger.info(
            f"Epoch {epoch+1:3d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} | val_F1={val_f1:.4f} | val_acc={val_acc:.3f}"
        )

        if early_stopper.step(val_f1, model):
            logger.info(f"Early stopping at epoch {epoch+1}")
            break
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

    # ── Test evaluation ───────────────────────────────────────────────────────
    best_path = OUTPUT_DIR / "arousal_mlp_arousal_best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        logger.info(f"Loaded best checkpoint: {best_path}")

    _, test_f1, test_acc, test_preds, test_true = eval_epoch(
        model, test_loader, criterion, device
    )
    short_names = [c for c in AROUSAL_CLASSES]
    report = classification_report(
        test_true, test_preds,
        labels=list(range(N_CLASSES)),
        target_names=short_names,
        zero_division=0,
    )
    logger.info(f"\nTest Macro-F1: {test_f1:.4f} | Accuracy: {test_acc:.3f}")
    logger.info(f"\nTest Classification Report:\n{report}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(test_true, test_preds)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=short_names, yticklabels=short_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Arousal MLP — Test Confusion Matrix (Macro-F1={test_f1:.3f})")
    plt.tight_layout()
    fig_path = FIGURE_DIR / "confusion_arousal_test.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    logger.info(f"Confusion matrix saved: {fig_path}")

    # ── Save results JSON ─────────────────────────────────────────────────────
    results = {
        "model": "ArousalMLP (from scratch)",
        "params": n_params,
        "feature_dim": input_dim,
        "n_classes": N_CLASSES,
        "class_names": AROUSAL_CLASSES,
        "best_val_macro_f1": round(best_val_f1, 4),
        "test_macro_f1":     round(test_f1, 4),
        "test_accuracy":     round(test_acc, 4),
        "train_n": len(X_train),
        "val_n":   len(X_val),
        "test_n":  len(X_test),
        "config": {
            "epochs":        cfg.epochs,
            "learning_rate": cfg.learning_rate,
            "batch_size":    cfg.batch_size,
            "hidden_dim":    cfg.hidden_dim,
            "n_layers":      cfg.n_layers,
            "dropout":       cfg.dropout,
            "seed":          cfg.seed,
        },
    }
    out_path = PROJECT_ROOT / "outputs/reports/arousal_eval.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info(f"Results saved: {out_path}")
    logger.info("=" * 60)
    logger.info(f"DONE  |  val_F1={best_val_f1:.4f}  test_F1={test_f1:.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Arousal MLP from scratch")
    parser.add_argument("--epochs",   type=int,   default=None, help="Override max epochs")
    parser.add_argument("--lr",       type=float, default=None, help="Override learning rate")
    parser.add_argument("--no-cache", action="store_true",      help="Recompute features even if cache exists")
    main(parser.parse_args())

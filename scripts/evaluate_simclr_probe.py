"""
scripts/evaluate_simclr_probe.py

Linear probe evaluation of the pre-trained SimCLR audio encoder.

A linear probe is a standard evaluation technique in self-supervised learning:
  1. Load the pre-trained encoder (freeze all weights)
  2. Add a small linear classifier head on top
  3. Train ONLY the head on labeled data
  4. Compare performance against: a randomly initialized encoder (frozen) + same head

This isolates the quality of the learned representations. High linear probe performance
indicates that the encoder has learned useful features without labels.

Usage:
    uv run python scripts/evaluate_simclr_probe.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from loguru import logger
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.audio_cnn import Emotion1DCNN
from src.data.dataset import NabatiDataset
from src.training.trainer import set_seed


# ── Hyper-parameters ──────────────────────────────────────────────────────────

SAMPLE_RATE = 16_000
MAX_AUDIO_SEC = 8
N_MELS = 128
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.01
WEIGHT_DECAY = 1e-4
PATIENCE = 5  # early stopping
SEED = 42
ENCODER_DIM = 512
NUM_CLASSES = 12


# ── Linear Probe Head ──────────────────────────────────────────────────────────


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int = 512, num_classes: int = 12):
        super().__init__()
        self.head = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(x)


# ── Collate function ──────────────────────────────────────────────────────────


def collate_probe(batch: list[dict]) -> dict:
    max_T = max(item["audio_tensor"].shape[-1] for item in batch)

    audio_batch = []
    emotion_batch = []

    for item in batch:
        audio = item["audio_tensor"]
        if audio.shape[-1] < max_T:
            audio = F.pad(audio, (0, max_T - audio.shape[-1]))
        audio_batch.append(audio)
        emotion_batch.append(item["emotion_id"])

    return {
        "audio_tensor": torch.stack(audio_batch),
        "emotion_id": torch.stack(emotion_batch),
    }


# ── Training loop ──────────────────────────────────────────────────────────────


def train_probe(
    model: nn.Module,
    encoder: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = EPOCHS,
    lr: float = LR,
    patience: int = PATIENCE,
) -> dict[str, float]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    encoder.eval()  # encoder is frozen
    for epoch in range(epochs):
        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        total_train_loss = 0.0
        train_steps = 0

        for batch in train_loader:
            audio = batch["audio_tensor"].to(device)
            emotion_ids = batch["emotion_id"].to(device)

            with torch.no_grad():
                embeddings = encoder.embed(audio)

            logits = model(embeddings)
            loss = criterion(logits, emotion_ids)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            train_steps += 1

        avg_train_loss = total_train_loss / train_steps

        # ── Validate ─────────────────────────────────────────────────────────
        model.eval()
        total_val_loss = 0.0
        total_val_correct = 0
        total_val_samples = 0
        val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                audio = batch["audio_tensor"].to(device)
                emotion_ids = batch["emotion_id"].to(device)

                embeddings = encoder.embed(audio)
                logits = model(embeddings)
                loss = criterion(logits, emotion_ids)

                total_val_loss += loss.item()
                val_steps += 1

                preds = logits.argmax(dim=1)
                total_val_correct += (preds == emotion_ids).sum().item()
                total_val_samples += emotion_ids.shape[0]

        avg_val_loss = total_val_loss / val_steps
        val_acc = (
            total_val_correct / total_val_samples if total_val_samples > 0 else 0.0
        )

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)

        # ── Early stopping ───────────────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"train_loss={avg_train_loss:.4f} | "
                f"val_loss={avg_val_loss:.4f} | "
                f"val_acc={val_acc:.4f}"
            )

        if epochs_no_improve >= patience:
            logger.info(f"Early stopping at epoch {epoch + 1} (patience={patience})")
            break

    logger.info(f"Best val_loss={best_val_loss:.4f} at epoch {best_epoch + 1}")

    return {
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "history": history,
    }


def evaluate_probe(
    model: nn.Module,
    encoder: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    encoder.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            audio = batch["audio_tensor"].to(device)
            emotion_ids = batch["emotion_id"].to(device)

            embeddings = encoder.embed(audio)
            logits = model(embeddings)
            preds = logits.argmax(dim=1)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(emotion_ids.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    accuracy = np.mean(all_preds == all_targets)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return {
        "clip_accuracy": accuracy,
        "clip_macro_f1": macro_f1,
    }


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    logger.add("logs/evaluate_simclr_probe.log", rotation="10 MB")
    set_seed(SEED)

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    logger.info(
        f"SimCLR Linear Probe Evaluation | device={device} | "
        f"epochs={EPOCHS} | batch={BATCH_SIZE} | lr={LR}"
    )

    # ── Data ───────────────────────────────────────────────────────────────
    data_dir = Path("data/processed")
    train_ds = NabatiDataset(
        data_dir / "train.jsonl",
        max_audio_sec=MAX_AUDIO_SEC,
        is_train=True,
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
    )
    val_ds = NabatiDataset(
        data_dir / "val.jsonl",
        max_audio_sec=MAX_AUDIO_SEC,
        is_train=False,
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
    )
    test_ds = NabatiDataset(
        data_dir / "test.jsonl",
        max_audio_sec=MAX_AUDIO_SEC,
        is_train=False,
        sample_rate=SAMPLE_RATE,
        n_mels=N_MELS,
    )

    if len(train_ds) == 0 or len(test_ds) == 0:
        logger.error("No training or test data found. Check data/processed/*.jsonl")
        return

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_probe
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_probe
    )
    test_loader = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_probe
    )

    logger.info(
        f"Data loaded: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}"
    )

    # ── Experiment 1: Pre-trained SimCLR encoder ───────────────────────────
    logger.info("=" * 70)
    logger.info("Experiment 1: Pre-trained SimCLR encoder + linear probe")
    logger.info("=" * 70)

    encoder_path = Path("outputs/models/simclr_encoder.pt")
    if not encoder_path.exists():
        logger.error(f"SimCLR encoder not found at {encoder_path}")
        logger.info("Please run: uv run python scripts/pretrain_audio_simclr.py")
        return

    simclr_encoder = Emotion1DCNN(num_classes=12).to(device)
    simclr_encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    for param in simclr_encoder.parameters():
        param.requires_grad = False
    logger.info(f"Loaded pre-trained encoder from {encoder_path}")

    simclr_probe = LinearProbe(ENCODER_DIM, NUM_CLASSES).to(device)
    logger.info("Training linear probe on frozen SimCLR encoder...")
    simclr_train_info = train_probe(
        simclr_probe, simclr_encoder, train_loader, val_loader, device
    )

    simclr_results = evaluate_probe(simclr_probe, simclr_encoder, test_loader, device)
    simclr_results.update(simclr_train_info)
    logger.success(
        f"SimCLR Probe Results: "
        f"clip_accuracy={simclr_results['clip_accuracy']:.4f}, "
        f"clip_macro_f1={simclr_results['clip_macro_f1']:.4f}"
    )

    # ── Experiment 2: Random encoder baseline ──────────────────────────────
    logger.info("=" * 70)
    logger.info("Experiment 2: Random encoder baseline + linear probe")
    logger.info("=" * 70)

    random_encoder = Emotion1DCNN(num_classes=12).to(device)
    for param in random_encoder.parameters():
        param.requires_grad = False
    logger.info("Created random encoder (all weights initialized, frozen)")

    random_probe = LinearProbe(ENCODER_DIM, NUM_CLASSES).to(device)
    logger.info("Training linear probe on frozen random encoder...")
    random_train_info = train_probe(
        random_probe, random_encoder, train_loader, val_loader, device
    )

    random_results = evaluate_probe(random_probe, random_encoder, test_loader, device)
    random_results.update(random_train_info)
    logger.success(
        f"Random Baseline Results: "
        f"clip_accuracy={random_results['clip_accuracy']:.4f}, "
        f"clip_macro_f1={random_results['clip_macro_f1']:.4f}"
    )

    # ── Summary and Comparison ─────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("SUMMARY: Pre-trained SimCLR vs Random Baseline")
    logger.info("=" * 70)

    simclr_acc = simclr_results["clip_accuracy"]
    simclr_f1 = simclr_results["clip_macro_f1"]
    random_acc = random_results["clip_accuracy"]
    random_f1 = random_results["clip_macro_f1"]

    acc_gain = simclr_acc - random_acc
    f1_gain = simclr_f1 - random_f1

    logger.info(f"SimCLR Probe  | Accuracy={simclr_acc:.4f}, Macro F1={simclr_f1:.4f}")
    logger.info(
        f"Random Baseline | Accuracy={random_acc:.4f}, Macro F1={random_f1:.4f}"
    )
    logger.info(f"Gain (SimCLR - Random) | Accuracy={acc_gain:+.4f}, F1={f1_gain:+.4f}")

    if acc_gain > 0 and f1_gain > 0:
        logger.success(
            "✓ SimCLR encoder learned useful representations "
            "(positive gain over random baseline)"
        )
    else:
        logger.warning(
            "✗ SimCLR encoder did NOT learn useful representations "
            "(no gain over random baseline)"
        )

    # ── Save results to JSON ───────────────────────────────────────────────
    output_dir = Path("outputs/reports")
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "method": "linear_probe_evaluation",
        "timestamp": str(Path(__file__).stat().st_mtime),
        "config": {
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "weight_decay": WEIGHT_DECAY,
            "patience": PATIENCE,
            "seed": SEED,
        },
        "simclr_encoder": {
            "path": str(encoder_path),
            "clip_accuracy": float(simclr_acc),
            "clip_macro_f1": float(simclr_f1),
            "best_val_loss": float(simclr_results["best_val_loss"]),
            "best_epoch": int(simclr_results["best_epoch"]),
        },
        "random_baseline": {
            "clip_accuracy": float(random_acc),
            "clip_macro_f1": float(random_f1),
            "best_val_loss": float(random_results["best_val_loss"]),
            "best_epoch": int(random_results["best_epoch"]),
        },
        "comparison": {
            "accuracy_gain": float(acc_gain),
            "f1_gain": float(f1_gain),
            "simclr_better": bool(acc_gain > 0 and f1_gain > 0),
        },
    }

    output_file = output_dir / "simclr_probe_eval.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    logger.success(f"Results saved to {output_file}")


if __name__ == "__main__":
    main()

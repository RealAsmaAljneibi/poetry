"""
scripts/train_audio_cnn.py

Trains Emotion1DCNN from scratch on mel-spectrograms for audio emotion classification.

SpecAugment reference: Park et al. 2019 — standard augmentation for audio CNNs.

Usage:
    uv run python scripts/train_audio_cnn.py
"""

import argparse
import json
import sys
from pathlib import Path

import librosa
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import audio_cnn_config
from src.data.labels import EMOTION_CLASSES, encode_emotion
from src.models.audio_cnn import Emotion1DCNN
from src.training.trainer import (
    TensorBoardLogger,
    EarlyStopper,
    get_scheduler,
    get_one_cycle_scheduler,
    set_seed,
)
from src.training.sanity import run_all_checks


# ── Focal Loss ────────────────────────────────────────────────────────────────


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ── SpecAugment ───────────────────────────────────────────────────────────────


def spec_augment(
    mel: torch.Tensor,
    freq_mask_param: int = 20,
    time_mask_param: int = 40,
    num_freq_masks: int = 2,
    num_time_masks: int = 2,
) -> torch.Tensor:
    """
    SpecAugment (Park et al. 2019) — randomly mask frequency and time strips.
    Applied only during training. Strongly regularises audio CNNs on small datasets.

    Args:
        mel: (n_mels, T) float tensor
        freq_mask_param: max width of frequency mask in mel bins
        time_mask_param: max width of time mask in frames
    """
    mel = mel.clone()
    n_mels, T = mel.shape

    # Frequency masking — mask F consecutive mel bins
    for _ in range(num_freq_masks):
        f = np.random.randint(0, freq_mask_param + 1)
        f0 = np.random.randint(0, max(1, n_mels - f))
        mel[f0 : f0 + f, :] = mel.mean()

    # Time masking — mask T consecutive frames
    for _ in range(num_time_masks):
        t = np.random.randint(0, min(time_mask_param + 1, T))
        t0 = np.random.randint(0, max(1, T - t))
        mel[:, t0 : t0 + t] = mel.mean()

    return mel


# ── Audio Dataset ─────────────────────────────────────────────────────────────


class AudioEmotionDataset(Dataset):
    """
    Loads audio clips and returns mel-spectrograms + emotion_audio label.

    Only includes samples where emotion_audio is annotated (not None).
    Applies Gaussian noise + SpecAugment during training.
    """

    def __init__(
        self,
        jsonl_path: Path,
        max_audio_sec: int = 8,
        sample_rate: int = 16000,
        n_mels: int = 128,
        is_train: bool = False,
        noise_amplitude: float = 0.005,
    ):
        self.sr = sample_rate
        self.max_len = max_audio_sec * sample_rate
        self.n_mels = n_mels
        self.is_train = is_train
        self.noise_amp = noise_amplitude
        self.samples: list[dict] = []

        skipped_no_label = 0
        skipped_no_audio = 0

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                label_str = rec.get("emotion_audio")
                if not label_str:
                    skipped_no_label += 1
                    continue
                label_id = encode_emotion(label_str)
                if label_id == -1:
                    skipped_no_label += 1
                    continue
                audio_path = Path(rec["audio_filename"])
                if not audio_path.exists():
                    skipped_no_audio += 1
                    continue
                self.samples.append({"path": audio_path, "label": label_id})

        logger.info(
            f"AudioEmotionDataset: {len(self.samples)} clips from {jsonl_path.name} "
            f"(skipped {skipped_no_label} no-label, {skipped_no_audio} missing audio)"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_mel(self, path: Path) -> torch.Tensor:
        try:
            wav, _ = librosa.load(str(path), sr=self.sr, mono=True)
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e} — using silence")
            wav = np.zeros(self.max_len, dtype=np.float32)

        if len(wav) >= self.max_len:
            wav = wav[: self.max_len]
        else:
            wav = np.pad(wav, (0, self.max_len - len(wav)))

        if self.is_train and self.noise_amp > 0:
            amp = self.noise_amp * np.random.uniform() * max(np.amax(np.abs(wav)), 1e-6)
            wav = wav + amp * np.random.normal(size=wav.shape).astype(np.float32)

        mel = librosa.feature.melspectrogram(y=wav, sr=self.sr, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        return torch.tensor(mel_db)  # (n_mels, T)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        mel = self._load_mel(s["path"])

        if self.is_train:
            mel = spec_augment(mel)

        return {
            "mel": mel,
            "label": torch.tensor(s["label"], dtype=torch.long),
        }


# ── Collate (variable time length → pad to batch max) ────────────────────────


def collate_fn(batch: list[dict]) -> dict:
    """Pad mel spectrograms to the longest in the batch along the time axis."""
    max_T = max(b["mel"].shape[-1] for b in batch)
    mels = torch.stack(
        [F.pad(b["mel"], (0, max_T - b["mel"].shape[-1])) for b in batch]
    )
    labels = torch.stack([b["label"] for b in batch])
    return {"mel": mels, "label": labels}


# ── Confusion matrix ──────────────────────────────────────────────────────────


def make_confusion_figure(y_true, y_pred, class_names, title) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(
        figsize=(max(8, len(class_names)), max(6, len(class_names) - 2))
    )
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    fig.tight_layout()
    return fig


# ── Train / eval loops ────────────────────────────────────────────────────────


def train_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    criterion,
    device,
    tb_logger,
    global_step,
    log_every,
    accum_steps=1,
) -> tuple[float, int]:
    model.train()
    total_loss, n_batches = 0.0, 0
    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        mel = batch["mel"].to(device)
        labels = batch["label"].to(device)

        loss = criterion(model(mel), labels) / accum_steps
        loss.backward()
        total_loss += loss.item() * accum_steps
        n_batches += 1

        is_last = (i + 1) == len(loader)
        is_boundary = (i + 1) % accum_steps == 0

        if is_boundary or is_last:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            if global_step % log_every == 0:
                tb_logger.log_step(
                    loss.item() * accum_steps, scheduler.get_last_lr()[0], global_step
                )

    return total_loss / max(n_batches, 1), global_step


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_preds, all_true = [], []

    with torch.no_grad():
        for batch in loader:
            mel = batch["mel"].to(device)
            labels = batch["label"].to(device)
            logits = model(mel)
            total_loss += criterion(logits, labels).item()
            n_batches += 1
            all_preds.extend(logits.argmax(-1).cpu().tolist())
            all_true.extend(labels.cpu().tolist())

    val_loss = total_loss / max(n_batches, 1)
    val_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
    val_acc = sum(p == t for p, t in zip(all_preds, all_true)) / max(len(all_true), 1)
    return val_loss, val_f1, val_acc, all_preds, all_true


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained-encoder",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to SimCLR pre-trained encoder weights (outputs/models/simclr_encoder.pt). "
        "When provided, the encoder is frozen for the first 5 epochs so only the "
        "classifier head adapts, then all layers are fine-tuned together.",
    )
    args = parser.parse_args()

    logger.add("logs/train_audio.log", rotation="10 MB")

    cfg = audio_cnn_config()
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    set_seed(cfg.seed)
    ssl_mode = args.pretrained_encoder is not None
    logger.info(
        f"Config: AudioCNN | task={cfg.task} | epochs={cfg.epochs} | "
        f"lr={cfg.learning_rate:.2e} | batch={cfg.batch_size} | device={device} | "
        f"ssl_pretrained={'YES' if ssl_mode else 'NO'}"
    )

    class_names = EMOTION_CLASSES
    num_classes = len(class_names)

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = AudioEmotionDataset(
        cfg.train_jsonl,
        cfg.max_audio_sec,
        cfg.sample_rate,
        cfg.n_mels,
        is_train=True,
        noise_amplitude=cfg.noise_amplitude,
    )
    val_ds = AudioEmotionDataset(
        cfg.val_jsonl, cfg.max_audio_sec, cfg.sample_rate, cfg.n_mels, is_train=False
    )
    test_ds = AudioEmotionDataset(
        cfg.test_jsonl, cfg.max_audio_sec, cfg.sample_rate, cfg.n_mels, is_train=False
    )

    if len(train_ds) == 0:
        logger.error(
            "No training samples with emotion_audio labels found. "
            "Ensure Pass-A annotation is complete."
        )
        return

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = Emotion1DCNN(num_classes=num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Emotion1DCNN: {total_params:,} params")
    assert total_params <= 50_000_000, f"CNN exceeds 50M constraint: {total_params:,}"

    # ── Load SimCLR pre-trained encoder ────────────────────────────────────────
    if ssl_mode:
        ckpt_path = args.pretrained_encoder
        if not ckpt_path.exists():
            logger.error(f"Pre-trained encoder not found: {ckpt_path}")
            return
        state = torch.load(ckpt_path, map_location=device)
        # The SimCLR checkpoint contains the full Emotion1DCNN state dict.
        # We load everything (encoder + old classifier head), then the classifier
        # head weights get overwritten by random init — only encoder matters.
        missing, unexpected = model.load_state_dict(state, strict=False)
        logger.info(
            f"Loaded SimCLR encoder from {ckpt_path.name} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
        # Freeze the encoder for the first HEAD_FREEZE_EPOCHS epochs so the
        # randomly-initialised classifier head can stabilise before end-to-end
        # fine-tuning begins (same strategy as gradual unfreeze).
        HEAD_FREEZE_EPOCHS = 10  # more head-only epochs before touching encoder
        for param in model.features.parameters():
            param.requires_grad = False
        logger.info(
            f"Encoder frozen for first {HEAD_FREEZE_EPOCHS} epochs — "
            "only classifier head trains initially."
        )
    else:
        HEAD_FREEZE_EPOCHS = 0

    # ── Class weights ─────────────────────────────────────────────────────────
    train_labels = [s["label"] for s in train_ds.samples]
    present_classes = np.unique(train_labels)
    class_w_present = compute_class_weight(
        "balanced", classes=present_classes, y=train_labels
    )
    class_w = np.ones(num_classes, dtype=np.float32)
    for cls_id, w in zip(present_classes, class_w_present):
        class_w[cls_id] = w
    class_weights = torch.tensor(class_w, dtype=torch.float32).to(device)
    logger.info(f"Class weights: min={class_w.min():.2f}, max={class_w.max():.2f}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    # CNN is trained from scratch: use CE + class weights (no focal loss).
    # Focal loss dampens gradients for uncertain predictions — harmful for random init
    # where ALL predictions are uncertain. Let the model learn first, then class
    # weights alone handle imbalance.
    criterion = nn.CrossEntropyLoss(
        weight=class_weights, label_smoothing=cfg.label_smoothing
    )
    logger.info(f"Loss: CrossEntropyLoss(label_smoothing={cfg.label_smoothing})")

    # ── Optimizer (simple AdamW — CNN is trained from scratch, no disc LR) ───
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    # ── Scheduler ─────────────────────────────────────────────────────────────
    total_steps = (len(train_loader) // cfg.grad_accum_steps) * cfg.epochs
    if cfg.scheduler == "one_cycle":
        scheduler = get_one_cycle_scheduler(
            optimizer, cfg.learning_rate, total_steps, pct_start=cfg.warmup_ratio
        )
        logger.info(
            f"Scheduler: OneCycleLR (max_lr={cfg.learning_rate}, steps={total_steps})"
        )
    else:
        scheduler = get_scheduler(optimizer, total_steps, cfg.warmup_ratio)
        logger.info(f"Scheduler: warmup+cosine (steps={total_steps})")
    logger.info(
        f"Grad accumulation: {cfg.grad_accum_steps} → "
        f"effective batch={cfg.batch_size * cfg.grad_accum_steps}, "
        f"optimizer steps/epoch={len(train_loader) // cfg.grad_accum_steps}"
    )

    # ── TensorBoard ───────────────────────────────────────────────────────────
    tb_logger = TensorBoardLogger(cfg.tensorboard_dir, "audio_cnn", cfg.task)
    tb_logger.log_hparams(
        hparam_dict={
            "lr": cfg.learning_rate,
            "batch": cfg.batch_size,
            "epochs": cfg.epochs,
            "n_mels": cfg.n_mels,
            "max_audio_sec": cfg.max_audio_sec,
        },
        metric_dict={"hparam/val_macro_f1": 0.0},
    )

    # ── Sanity checks ─────────────────────────────────────────────────────────
    # Use a SEPARATE optimizer with the real LR for the sanity check.
    # Reason: get_scheduler (LambdaLR) immediately calls step() during __init__,
    # which sets the training optimizer's LR → 0 (warmup start). Passing the
    # scheduled optimizer to run_all_checks would cause the overfit test to
    # fail spuriously because all gradient steps have LR=0.
    logger.info("Running sanity checks...")
    sanity_optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    run_all_checks(model, train_loader, criterion, sanity_optim, num_classes, device)
    del sanity_optim  # discard — training uses its own fresh optimizer below

    # ── Training loop ─────────────────────────────────────────────────────────
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    early_stopper = EarlyStopper(cfg.patience, cfg.output_dir, "audio_cnn_emotion")
    global_step = 0
    best_val_f1 = 0.0

    logger.info("=" * 60)
    logger.info(
        f"Starting training: {cfg.epochs} epochs, {len(train_loader)} steps/epoch"
    )
    logger.info("=" * 60)

    for epoch in range(cfg.epochs):
        # Unfreeze encoder after HEAD_FREEZE_EPOCHS (SSL fine-tuning strategy).
        # IMPORTANT: encoder LR must be very small (1e-5) to avoid catastrophic
        # forgetting of SimCLR features. The run with 1e-4 crashed val F1 from
        # 0.104 → 0.022. Use 0.01× of head LR, not 0.1×.
        if ssl_mode and epoch == HEAD_FREEZE_EPOCHS:
            for param in model.features.parameters():
                param.requires_grad = True
            encoder_lr = cfg.learning_rate * 0.01  # 1e-5 (was 1e-4 — too aggressive)
            # Discriminative LRs: tiny for pre-trained encoder, normal for head
            optimizer = torch.optim.AdamW(
                [
                    {"params": model.features.parameters(), "lr": encoder_lr},
                    {
                        "params": model.classifier.parameters(),
                        "lr": cfg.learning_rate * 0.1,
                    },
                ],
                weight_decay=cfg.weight_decay,
            )
            remaining = (cfg.epochs - HEAD_FREEZE_EPOCHS) * (
                len(train_loader) // cfg.grad_accum_steps
            )
            scheduler = get_scheduler(optimizer, remaining, cfg.warmup_ratio)
            logger.info(
                f"Epoch {epoch + 1}: encoder unfrozen — discriminative LR: "
                f"encoder={encoder_lr:.1e}, head={cfg.learning_rate * 0.1:.1e}"
            )

        train_loss, global_step = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            tb_logger,
            global_step,
            cfg.log_every_n_steps,
            cfg.grad_accum_steps,
        )
        tb_logger.log_histograms(model, epoch)

        val_loss, val_f1, val_acc, val_preds, val_true = eval_epoch(
            model, val_loader, criterion, device
        )
        tb_logger.log_epoch(epoch, train_loss, val_loss, val_f1, val_acc)

        logger.info(
            f"Epoch {epoch + 1:3d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_F1={val_f1:.4f} | val_acc={val_acc:.3f}"
        )

        if (epoch + 1) % 5 == 0:
            short_names = [c.split("(")[0].strip() for c in class_names]
            fig = make_confusion_figure(
                val_true, val_preds, short_names, f"Val Confusion — Epoch {epoch + 1}"
            )
            tb_logger.log_confusion_matrix(fig, "confusion/emotion_audio/val", epoch)
            plt.close(fig)

        if early_stopper.step(val_f1, model):
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

    # ── Test evaluation ───────────────────────────────────────────────────────
    logger.info(f"Loading best checkpoint: {early_stopper.best_ckpt}")
    model.load_state_dict(torch.load(early_stopper.best_ckpt, map_location=device))

    test_loss, test_f1, test_acc, test_preds, test_true = eval_epoch(
        model, test_loader, criterion, device
    )
    logger.success(
        f"TEST RESULTS | loss={test_loss:.4f} | "
        f"Macro-F1={test_f1:.4f} | accuracy={test_acc:.4f}"
    )

    # ── Reports ───────────────────────────────────────────────────────────────
    report_dir = Path("outputs/reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    short_names = [c.split("(")[0].strip() for c in class_names]
    present_ids = sorted(set(test_true))
    present_names = [short_names[i] for i in present_ids]
    report_str = classification_report(
        test_true,
        test_preds,
        labels=present_ids,
        target_names=present_names,
        zero_division=0,
    )
    logger.info(f"\n{report_str}")
    (report_dir / "audio_cnn_emotion_report.txt").write_text(report_str)

    fig = make_confusion_figure(
        test_true,
        test_preds,
        short_names,
        f"Test Confusion — AudioCNN | emotion_audio | Macro-F1={test_f1:.3f}",
    )
    fig_path = report_dir / "audio_cnn_emotion_confusion.png"
    fig.savefig(fig_path, bbox_inches="tight", dpi=150)
    logger.info(f"Confusion matrix → {fig_path}")
    plt.close(fig)

    tb_logger.log_hparams(
        hparam_dict={
            "lr": cfg.learning_rate,
            "batch": cfg.batch_size,
            "epochs": cfg.epochs,
        },
        metric_dict={"hparam/test_macro_f1": test_f1, "hparam/test_acc": test_acc},
    )
    tb_logger.close()
    logger.success(
        f"Done. Best val Macro-F1={best_val_f1:.4f}, test Macro-F1={test_f1:.4f}"
    )


if __name__ == "__main__":
    main()

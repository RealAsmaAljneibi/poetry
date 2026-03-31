"""
Train a multimodal genre fusion head over text and audio embeddings.

Supports:
  - fusion strategies: concat, gated, cross_attn
  - ablation modes: fusion, text_only, audio_only
  - frozen encoders by default, with optional end-to-end fine-tuning
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import seaborn as sns

from src.data.labels import GENRE_CLASSES, encode_genre
from src.models.audio_cnn import Emotion1DCNN
from src.models.fusion import NabatiMultimodalFusion
from src.training.trainer import EarlyStopper, TensorBoardLogger, get_scheduler, set_seed


class FusionGenreDataset(Dataset):
    """Loads corrected text + mel spectrogram + genre label for fusion training."""

    def __init__(
        self,
        jsonl_path: Path,
        tokenizer,
        max_seq_len: int = 32,
        max_audio_sec: int = 8,
        sample_rate: int = 16_000,
        n_mels: int = 128,
        is_train: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.max_audio_len = max_audio_sec * sample_rate
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.is_train = is_train
        self.samples: list[dict[str, object]] = []

        skipped = 0
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            text = (record.get("text_corrected") or "").strip()
            label = encode_genre(record.get("genre_en", ""))
            audio_path = Path(record.get("audio_filename", ""))
            if not text or label == -1 or not audio_path.exists():
                skipped += 1
                continue
            self.samples.append({"text": text, "label": label, "audio_path": audio_path})

        logger.info(
            f"FusionGenreDataset: {len(self.samples)} samples from {jsonl_path.name} "
            f"(skipped {skipped})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_mel(self, audio_path: Path) -> torch.Tensor:
        wav, _ = librosa.load(str(audio_path), sr=self.sample_rate, mono=True)
        if len(wav) >= self.max_audio_len:
            wav = wav[: self.max_audio_len]
        else:
            wav = np.pad(wav, (0, self.max_audio_len - len(wav)))

        if self.is_train:
            peak = max(float(np.max(np.abs(wav))), 1e-6)
            noise_amp = 0.005 * float(np.random.uniform()) * peak
            wav = wav + noise_amp * np.random.normal(size=wav.shape).astype(np.float32)

        mel = librosa.feature.melspectrogram(y=wav, sr=self.sample_rate, n_mels=self.n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        return torch.tensor(mel_db, dtype=torch.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]
        enc = self.tokenizer(
            sample["text"],
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "mel": self._load_mel(sample["audio_path"]),
            "label": torch.tensor(sample["label"], dtype=torch.long),
        }


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    max_t = max(item["mel"].shape[-1] for item in batch)
    mel = torch.stack([F.pad(item["mel"], (0, max_t - item["mel"].shape[-1])) for item in batch])
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "mel": mel,
        "label": torch.stack([item["label"] for item in batch]),
    }


def compute_ece(probabilities: list[list[float]], labels: list[int], n_bins: int = 10) -> float:
    if not probabilities:
        return 0.0

    confidences = np.array([max(row) for row in probabilities], dtype=np.float32)
    predictions = np.array([int(np.argmax(row)) for row in probabilities], dtype=np.int64)
    truth = np.array(labels, dtype=np.int64)
    correctness = (predictions == truth).astype(np.float32)
    boundaries = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0
    for lower, upper in zip(boundaries[:-1], boundaries[1:]):
        if upper == 1.0:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if mask.any():
            ece += float(mask.mean() * abs(correctness[mask].mean() - confidences[mask].mean()))
    return float(ece)


def make_confusion_figure(y_true: list[int], y_pred: list[int], title: str) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(GENRE_CLASSES))))
    short_names = [label.split("(")[0].strip() for label in GENRE_CLASSES]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=short_names, yticklabels=short_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    fig.tight_layout()
    return fig


def load_text_encoder(model_name: str, checkpoint_path: Path | None, num_classes: int, device: torch.device):
    config = AutoConfig.from_pretrained(model_name, local_files_only=True)
    config.num_labels = num_classes
    text_classifier = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    ).to(device)
    if checkpoint_path is not None and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        missing, unexpected = text_classifier.load_state_dict(state, strict=False)
        logger.info(
            f"Loaded text checkpoint {checkpoint_path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    elif checkpoint_path is not None:
        logger.warning(f"Text checkpoint not found: {checkpoint_path} — using base model weights")
    return text_classifier.base_model


def load_audio_encoder(checkpoint_path: Path | None, device: torch.device) -> Emotion1DCNN:
    audio_encoder = Emotion1DCNN(num_classes=12).to(device)
    if checkpoint_path is not None and checkpoint_path.exists():
        state = torch.load(checkpoint_path, map_location=device)
        missing, unexpected = audio_encoder.load_state_dict(state, strict=False)
        logger.info(
            f"Loaded audio checkpoint {checkpoint_path} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )
    elif checkpoint_path is not None:
        logger.warning(f"Audio checkpoint not found: {checkpoint_path} — using randomly initialized CNN")
    return audio_encoder


def run_epoch(
    model: NabatiMultimodalFusion,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    mode: str,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    tb_logger: TensorBoardLogger | None = None,
    global_step: int = 0,
    log_every: int = 20,
) -> tuple[float, float, float, list[int], list[int], list[list[float]], int]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_preds: list[int] = []
    all_true: list[int] = []
    all_probs: list[list[float]] = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        mel = batch["mel"].to(device)
        labels = batch["label"].to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(input_ids=input_ids, attention_mask=attention_mask, mel_spec=mel, mode=mode)
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            global_step += 1
            if tb_logger is not None and global_step % log_every == 0:
                tb_logger.log_step(loss.item(), scheduler.get_last_lr()[0], global_step)

        probs = torch.softmax(logits, dim=-1).detach().cpu().tolist()
        preds = [int(np.argmax(row)) for row in probs]
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_true.extend(labels.cpu().tolist())
        total_loss += loss.item()

    avg_loss = total_loss / max(len(loader), 1)
    macro_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
    accuracy = sum(int(p == t) for p, t in zip(all_preds, all_true)) / max(len(all_true), 1)
    return avg_loss, macro_f1, accuracy, all_preds, all_true, all_probs, global_step


def evaluate(
    model: NabatiMultimodalFusion,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    mode: str,
):
    with torch.no_grad():
        return run_epoch(model, loader, criterion, device, mode)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fusion-strategy", choices=["concat", "gated", "cross_attn"], default="gated")
    parser.add_argument("--mode", choices=["fusion", "text_only", "audio_only"], default="fusion")
    parser.add_argument("--text-model-name", default="faisalq/bert-base-arapoembert")
    parser.add_argument("--text-checkpoint", type=Path, default=Path("outputs/models/arapoem_genre/arapoem_genre_best.pt"))
    parser.add_argument("--audio-checkpoint", type=Path, default=Path("outputs/models/audio_cnn/audio_cnn_emotion_best.pt"))
    parser.add_argument("--fusion-checkpoint", type=Path, default=None)
    parser.add_argument("--train-jsonl", type=Path, default=Path("data/processed/train.jsonl"))
    parser.add_argument("--val-jsonl", type=Path, default=Path("data/processed/val.jsonl"))
    parser.add_argument("--test-jsonl", type=Path, default=Path("data/processed/test.jsonl"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-seq-len", type=int, default=32)
    parser.add_argument("--max-audio-sec", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--unfreeze-encoders", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/models/fusion"))
    parser.add_argument("--report-path", type=Path, default=None)
    return parser


def main(args: argparse.Namespace) -> float:
    logger.add("logs/train_fusion.log", rotation="10 MB")
    set_seed(args.seed)
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.text_model_name, local_files_only=True)
    train_ds = FusionGenreDataset(
        args.train_jsonl,
        tokenizer,
        max_seq_len=args.max_seq_len,
        max_audio_sec=args.max_audio_sec,
        is_train=True,
    )
    val_ds = FusionGenreDataset(
        args.val_jsonl,
        tokenizer,
        max_seq_len=args.max_seq_len,
        max_audio_sec=args.max_audio_sec,
        is_train=False,
    )
    test_ds = FusionGenreDataset(
        args.test_jsonl,
        tokenizer,
        max_seq_len=args.max_seq_len,
        max_audio_sec=args.max_audio_sec,
        is_train=False,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    text_encoder = load_text_encoder(args.text_model_name, args.text_checkpoint, len(GENRE_CLASSES), device)
    audio_encoder = load_audio_encoder(args.audio_checkpoint, device)
    model_tag = f"fusion_{args.fusion_strategy}_{args.mode}"
    model = NabatiMultimodalFusion(
        text_encoder=text_encoder,
        audio_encoder=audio_encoder,
        fusion_strategy=args.fusion_strategy,
        task="genre",
        num_classes=len(GENRE_CLASSES),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        freeze_encoders=not args.unfreeze_encoders,
    ).to(device)

    train_labels = np.array([sample["label"] for sample in train_ds.samples], dtype=np.int64)
    present_classes = np.unique(train_labels)
    class_weights_present = compute_class_weight("balanced", classes=present_classes, y=train_labels)
    class_weights = np.ones(len(GENRE_CLASSES), dtype=np.float32)
    for label_id, weight in zip(present_classes, class_weights_present):
        class_weights[int(label_id)] = float(weight)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fusion_ckpt = args.fusion_checkpoint or args.output_dir / f"{model_tag}_best.pt"

    if args.eval_only:
        if not fusion_ckpt.exists():
            raise FileNotFoundError(f"Fusion checkpoint not found: {fusion_ckpt}")
        model.load_state_dict(torch.load(fusion_ckpt, map_location=device))
        logger.info(f"Loaded fusion checkpoint for evaluation: {fusion_ckpt}")
        best_val_f1 = float("nan")
        tb_logger = None
    else:
        optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        total_steps = max(len(train_loader) * args.epochs, 1)
        scheduler = get_scheduler(optimizer, total_steps=total_steps, warmup_ratio=args.warmup_ratio)
        tb_logger = TensorBoardLogger(Path("outputs/runs"), "fusion", f"{args.fusion_strategy}_{args.mode}")
        early_stopper = EarlyStopper(args.patience, args.output_dir, model_tag)
        global_step = 0
        best_val_f1 = 0.0

        logger.info(
            f"Starting fusion training | strategy={args.fusion_strategy} | mode={args.mode} | "
            f"epochs={args.epochs} | batch={args.batch_size} | lr={args.learning_rate:.2e}"
        )
        for epoch in range(args.epochs):
            train_loss, train_f1, train_acc, _, _, _, global_step = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                args.mode,
                optimizer=optimizer,
                scheduler=scheduler,
                tb_logger=tb_logger,
                global_step=global_step,
            )
            val_loss, val_f1, val_acc, val_preds, val_true, _, _ = evaluate(
                model, val_loader, criterion, device, args.mode
            )

            tb_logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_f1, val_acc)
            logger.info(
                f"Epoch {epoch + 1:02d}/{args.epochs} | train_loss={train_loss:.4f} | "
                f"train_F1={train_f1:.4f} | val_loss={val_loss:.4f} | val_F1={val_f1:.4f} | "
                f"val_acc={val_acc:.4f}"
            )

            if (epoch + 1) % 5 == 0:
                fig = make_confusion_figure(val_true, val_preds, f"Fusion Val Confusion — Epoch {epoch + 1}")
                tb_logger.log_confusion_matrix(fig, f"confusion/fusion/{args.mode}", epoch)
                plt.close(fig)

            if early_stopper.step(val_f1, model):
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break
            best_val_f1 = max(best_val_f1, val_f1)

        model.load_state_dict(torch.load(early_stopper.best_ckpt, map_location=device))
        logger.info(f"Loaded best checkpoint: {early_stopper.best_ckpt}")

    test_loss, test_f1, test_acc, test_preds, test_true, test_probs, _ = evaluate(
        model, test_loader, criterion, device, args.mode
    )
    ece = compute_ece(test_probs, test_true)

    report_dir = Path("outputs/reports")
    figure_dir = Path("outputs/figures")
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    figure_path = figure_dir / f"{model_tag}_confusion.png"
    fig = make_confusion_figure(
        test_true,
        test_preds,
        f"Fusion Test Confusion — {args.fusion_strategy} | {args.mode} | F1={test_f1:.3f}",
    )
    fig.savefig(figure_path, bbox_inches="tight", dpi=150)
    plt.close(fig)

    short_names = [label.split("(")[0].strip() for label in GENRE_CLASSES]
    present_ids = sorted(set(test_true))
    report_text = classification_report(
        test_true,
        test_preds,
        labels=present_ids,
        target_names=[short_names[index] for index in present_ids],
        zero_division=0,
    )
    report_txt_path = report_dir / f"{model_tag}_report.txt"
    report_txt_path.write_text(report_text, encoding="utf-8")

    summary: dict[str, object] = {
        "task": "genre",
        "fusion_strategy": args.fusion_strategy,
        "mode": args.mode,
        "freeze_encoders": not args.unfreeze_encoders,
        "best_val_macro_f1": None if np.isnan(best_val_f1) else round(float(best_val_f1), 4),
        "test_macro_f1": round(float(test_f1), 4),
        "test_accuracy": round(float(test_acc), 4),
        "test_ece": round(float(ece), 4),
        "test_loss": round(float(test_loss), 4),
        "figure_path": str(figure_path),
        "report_txt_path": str(report_txt_path),
        "fusion_checkpoint": str(fusion_ckpt),
    }

    if args.fusion_strategy == "gated" and args.mode == "fusion":
        gate_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
        batch = next(iter(gate_loader))
        with torch.no_grad():
            gate_weights = model.get_gate_weights(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["mel"].to(device),
            )
        summary["val_gate_mean_text"] = round(float(gate_weights[:, 0].mean().item()), 4)
        summary["val_gate_mean_audio"] = round(float(gate_weights[:, 1].mean().item()), 4)

    json_path = args.report_path or report_dir / f"{model_tag}.json"
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.success(
        f"Fusion complete | strategy={args.fusion_strategy} | mode={args.mode} | "
        f"test_F1={test_f1:.4f} | acc={test_acc:.4f} | ECE={ece:.4f}"
    )
    logger.info(f"JSON report → {json_path}")

    if tb_logger is not None:
        tb_logger.log_hparams(
            hparam_dict={
                "strategy": args.fusion_strategy,
                "mode": args.mode,
                "lr": args.learning_rate,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
            },
            metric_dict={"hparam/test_macro_f1": float(test_f1), "hparam/test_acc": float(test_acc)},
        )
        tb_logger.close()

    return float(test_f1)


if __name__ == "__main__":
    parser = build_arg_parser()
    main(parser.parse_args())

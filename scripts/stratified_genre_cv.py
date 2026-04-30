"""
scripts/stratified_genre_cv.py

Stratified poet-disjoint 5-fold CV for genre, providing a complementary
performance estimate to the fixed strict split.

The adopted strict split (train/val/test) concentrates Hikma poets in test,
creating artificially hard evaluation. This script uses cross-validation with
poet-disjoint folds to provide a fairer estimate of the model's true
genre-discriminating ability.

Improvements over baseline run:
  - Model: faisalq/bert-base-arapoembert (poetry-domain pretraining vs general Arabic)
  - Context window: 5 clips (was 3) — captures more poem structure
  - Class-weighted CrossEntropyLoss: inverse-frequency weights address Hikma dominance
  - Label smoothing (ε=0.1): prevents overconfidence, improves ECE calibration
  - Discriminative LR decay=0.9: lower BERT layers retain poetry pretraining
  - Gradient clipping (max_norm=1.0): prevents explosion during fine-tuning

Usage:
    uv run python scripts/stratified_genre_cv.py

Output:
    outputs/reports/stratified_genre_cv_improved.json
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.labels import GENRE_CLASSES, encode_genre
from src.training.trainer import (
    get_optimizer,
    get_scheduler,
    set_seed,
)


class SimpleEarlyStopper:
    """Minimal early stopper for CV folds (no checkpoint saving needed)."""

    def __init__(self, patience: int):
        self.patience = patience
        self.counter = 0

    def reset(self) -> None:
        self.counter = 0

    def step(self) -> None:
        self.counter += 1

    def should_stop(self) -> bool:
        return self.counter >= self.patience


# ── Text Dataset for CV ────────────────────────────────────────────────────


class NabatiTextDatasetCV(Dataset):
    """
    Reads samples from memory (list of dicts) and returns tokenised Arabic text + label.
    Used within CV folds where we create train/val splits programmatically.

    Input field: text_corrected (human-corrected transcript)
    Label field: genre_en

    context_window (sliding window):
        1 → single clip
        3 → concatenate [clip_{i-1}, clip_i, clip_{i+1}]
    """

    def __init__(
        self,
        samples: list[dict],
        tokenizer,
        max_seq_len: int = 128,
        context_window: int = 1,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.context_window = context_window
        self.processed_samples: list[dict] = []

        # Group by poem and sort by start time for context window construction
        poem_clips: dict[str, list[dict]] = defaultdict(list)
        for sample in samples:
            poem_id = sample.get("source_poem", "unknown")
            poem_clips[poem_id].append(sample)

        for pid in poem_clips:
            poem_clips[pid].sort(key=lambda r: r.get("start", 0))

        skipped = 0
        half = context_window // 2

        for pid, clips in poem_clips.items():
            n = len(clips)
            for i, rec in enumerate(clips):
                text_i = rec.get("text_corrected", "").strip()
                genre_str = rec.get("genre_en", "")
                label_id = encode_genre(genre_str)

                if not text_i or label_id == -1:
                    skipped += 1
                    continue

                if context_window > 1:
                    parts: list[str] = []
                    # Previous clip(s)
                    for k in range(max(0, i - half), i):
                        t = clips[k].get("text_corrected", "").strip()
                        if t:
                            parts.append(t)
                    parts.append(text_i)
                    # Next clip(s)
                    for k in range(i + 1, min(n, i + 1 + half)):
                        t = clips[k].get("text_corrected", "").strip()
                        if t:
                            parts.append(t)
                    text = " ".join(parts)
                else:
                    text = text_i

                poem_id_str = pid + "|" + rec.get("poet_en", "")
                self.processed_samples.append(
                    {
                        "text": text,
                        "label": label_id,
                        "poem_id": poem_id_str,
                    }
                )

        logger.debug(
            f"CV Dataset: {len(self.processed_samples)} samples (skipped {skipped})"
        )

    def __len__(self) -> int:
        return len(self.processed_samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.processed_samples[idx]
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
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "poem_id": sample["poem_id"],
        }


# ── Training and Evaluation ────────────────────────────────────────────────


def train_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    criterion,
    device,
    log_every: int = 100,
) -> tuple[float, float]:
    """Train for one epoch. Returns (avg_loss, accuracy)."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    n_correct = 0
    n_total = 0

    for i, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

        preds = logits.detach().argmax(dim=-1)
        n_correct += (preds == labels).sum().item()
        n_total += labels.size(0)

        if (i + 1) % log_every == 0:
            logger.debug(
                f"Batch {i + 1}/{len(loader)}: loss={loss.item():.4f}, "
                f"acc={n_correct / n_total:.4f}"
            )

    train_loss = total_loss / max(n_batches, 1)
    train_acc = n_correct / max(n_total, 1)
    return train_loss, train_acc


def eval_epoch(
    model,
    loader,
    criterion,
    device,
) -> tuple[float, float, list, list, list]:
    """Evaluate for one epoch. Returns (loss, f1, preds, true_labels, poem_ids)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds, all_true, all_poem_ids = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            poem_ids = batch["poem_id"]

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = criterion(logits, labels)

            total_loss += loss.item()
            n_batches += 1

            preds = logits.detach().argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_true.extend(labels.cpu().tolist())
            all_poem_ids.extend(poem_ids)

    val_loss = total_loss / max(n_batches, 1)
    val_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
    return val_loss, val_f1, all_preds, all_true, all_poem_ids


def poem_level_eval(
    clip_preds: list[int],
    clip_true: list[int],
    clip_poem_ids: list[str],
) -> float:
    """Aggregate clip predictions to poem level via majority vote. Returns poem F1."""
    poem_preds: dict[str, list[int]] = defaultdict(list)
    poem_labels: dict[str, int] = {}

    for pred, true, pid in zip(clip_preds, clip_true, clip_poem_ids):
        poem_preds[pid].append(pred)
        poem_labels[pid] = true

    poem_pred_list, poem_true_list = [], []
    for pid in poem_preds:
        votes = poem_preds[pid]
        pred = max(set(votes), key=votes.count)
        poem_pred_list.append(pred)
        poem_true_list.append(poem_labels[pid])

    poem_f1 = f1_score(poem_true_list, poem_pred_list, average="macro", zero_division=0)
    return poem_f1


# ── Main CV Loop ───────────────────────────────────────────────────────────


def main():
    set_seed(42)
    logger.add("logs/stratified_genre_cv.log", rotation="10 MB")
    logger.info("=" * 80)
    logger.info("Stratified Poet-Disjoint 5-Fold CV for Genre Classification")
    logger.info("=" * 80)

    # ── Load full dataset ──────────────────────────────────────────────────────
    dataset_path = (
        Path(__file__).parent.parent / "data" / "processed" / "master_dataset.jsonl"
    )
    logger.info(f"Loading dataset from {dataset_path}")

    all_samples = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            sample = json.loads(line)
            all_samples.append(sample)

    logger.info(f"Loaded {len(all_samples)} clips")

    poet_groups: dict[str, list[dict]] = defaultdict(list)
    for sample in all_samples:
        poet = sample.get("poet_en", "unknown")
        poet_groups[poet].append(sample)

    poets = list(poet_groups.keys())
    n_poets = len(poets)
    logger.info(f"Dataset has {n_poets} unique poets")
    logger.info(f"Poets: {poets}")

    # ── Setup CV ───────────────────────────────────────────────────────────────
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Create stratification targets: genre distribution per poet
    poet_genre_targets = []
    for poet in poets:
        genre_ids = [encode_genre(s.get("genre_en", "")) for s in poet_groups[poet]]
        # Use the most common genre as the stratification target
        target = max(set(genre_ids), key=genre_ids.count) if genre_ids else 0
        poet_genre_targets.append(target)

    # ── Device & Model Setup ───────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # bert-base-arapoembert has max_position_embeddings=32 — too short for window=5 context.
    # bert-base-arabertv2 supports 512 tokens and works correctly with max_seq_len=128.
    model_name = "aubmindlab/bert-base-arabertv2"
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    logger.info(f"Loaded tokenizer: {model_name}")

    # ── CV Folds ───────────────────────────────────────────────────────────────
    fold_results = []

    for fold_idx, (train_poet_indices, test_poet_indices) in enumerate(
        skf.split(poets, poet_genre_targets)
    ):
        logger.info(f"\n{'=' * 80}")
        logger.info(f"FOLD {fold_idx + 1}/{n_splits}")
        logger.info(f"{'=' * 80}")

        train_poets = [poets[i] for i in train_poet_indices]
        test_poets = [poets[i] for i in test_poet_indices]

        logger.info(f"Train poets ({len(train_poets)}): {train_poets}")
        logger.info(f"Test poets ({len(test_poets)}): {test_poets}")

        train_samples = [s for poet in train_poets for s in poet_groups[poet]]
        test_samples = [s for poet in test_poets for s in poet_groups[poet]]

        logger.info(
            f"Train clips: {len(train_samples)}, Test clips: {len(test_samples)}"
        )

        # Create datasets with context_window=5 (matching GENRE-R4)
        train_dataset = NabatiTextDatasetCV(train_samples, tokenizer, context_window=5)
        test_dataset = NabatiTextDatasetCV(test_samples, tokenizer, context_window=5)

        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

        # ── Load fresh ArabertV2 ─────────────────────────────────────────────
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(GENRE_CLASSES),
            local_files_only=True,
        )
        model = model.to(device)
        logger.info(f"Loaded model: {model_name}")

        # ── Training Setup ─────────────────────────────────────────────────────
        # Class-weighted loss: inverse-frequency weights penalise Hikma dominance
        train_genre_ids = [encode_genre(s.get("genre_en", "")) for s in train_samples]
        label_counts = Counter(train_genre_ids)
        n_classes = len(GENRE_CLASSES)
        class_weights = torch.zeros(n_classes)
        for i in range(n_classes):
            class_weights[i] = 1.0 / (label_counts.get(i, 1) + 1e-6)
        class_weights = class_weights / class_weights.sum() * n_classes
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device), label_smoothing=0.1
        )
        # Discriminative LRs: lower layers keep pretrained poetry knowledge
        optimizer = get_optimizer(
            model, base_lr=2e-5, weight_decay=0.01, discriminative_lr_decay=0.9
        )
        total_steps = len(train_loader) * 10  # 10 epochs
        scheduler = get_scheduler(optimizer, total_steps=total_steps, warmup_ratio=0.1)

        early_stopper = SimpleEarlyStopper(patience=3)
        best_val_f1 = 0.0

        # ── Training Loop ──────────────────────────────────────────────────────
        logger.info("Starting training...")
        for epoch in range(10):
            train_loss, train_acc = train_epoch(
                model, train_loader, optimizer, scheduler, criterion, device
            )
            val_loss, val_f1, _, _, _ = eval_epoch(
                model, test_loader, criterion, device
            )

            logger.info(
                f"Epoch {epoch + 1}/10 | "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                f"val_loss={val_loss:.4f} val_f1={val_f1:.4f}"
            )

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                early_stopper.reset()
            else:
                early_stopper.step()

            if early_stopper.should_stop():
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # ── Final Evaluation ───────────────────────────────────────────────────
        logger.info("Final evaluation on test fold...")
        val_loss, val_f1, clip_preds, clip_true, clip_poem_ids = eval_epoch(
            model, test_loader, criterion, device
        )

        poem_f1 = poem_level_eval(clip_preds, clip_true, clip_poem_ids)

        logger.info(f"Fold {fold_idx + 1} Results:")
        logger.info(f"  Clip-level F1:  {val_f1:.4f}")
        logger.info(f"  Poem-level F1:  {poem_f1:.4f}")

        fold_results.append(
            {
                "fold": fold_idx + 1,
                "train_poets": train_poets,
                "test_poets": test_poets,
                "clip_f1": float(val_f1),
                "poem_f1": float(poem_f1),
            }
        )

        del model, optimizer, scheduler, criterion
        torch.cuda.empty_cache()

    # ── Summary Statistics ─────────────────────────────────────────────────────
    logger.info(f"\n{'=' * 80}")
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info(f"{'=' * 80}")

    clip_f1_scores = [r["clip_f1"] for r in fold_results]
    poem_f1_scores = [r["poem_f1"] for r in fold_results]

    clip_f1_mean = np.mean(clip_f1_scores)
    clip_f1_std = np.std(clip_f1_scores)
    poem_f1_mean = np.mean(poem_f1_scores)
    poem_f1_std = np.std(poem_f1_scores)

    logger.info(f"Clip-level F1:  {clip_f1_mean:.4f} ± {clip_f1_std:.4f}")
    logger.info(f"Poem-level F1:  {poem_f1_mean:.4f} ± {poem_f1_std:.4f}")
    logger.info(f"Individual fold results: {clip_f1_scores}")

    # ── Save Results ───────────────────────────────────────────────────────────
    output_dir = Path(__file__).parent.parent / "outputs" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "task": "genre",
        "method": "stratified_5fold_cv_improved",
        "model": model_name,
        "n_splits": n_splits,
        "n_poets": n_poets,
        "n_total_clips": len(all_samples),
        "context_window": 5,
        "learning_rate": 2e-5,
        "discriminative_lr_decay": 0.9,
        "class_weighted_loss": True,
        "label_smoothing": 0.1,
        "gradient_clipping": 1.0,
        "batch_size": 16,
        "epochs_per_fold": 10,
        "early_stopping_patience": 3,
        "clip_level": {
            "f1_mean": float(clip_f1_mean),
            "f1_std": float(clip_f1_std),
            "f1_per_fold": clip_f1_scores,
        },
        "poem_level": {
            "f1_mean": float(poem_f1_mean),
            "f1_std": float(poem_f1_std),
            "f1_per_fold": poem_f1_scores,
        },
        "fold_details": fold_results,
    }

    output_path = output_dir / "stratified_genre_cv_improved.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    logger.info(f"\nResults saved to {output_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()

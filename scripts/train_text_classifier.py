"""
scripts/train_text_classifier.py

Fine-tunes AraPoemBERT or mBERT on genre/emotion_text classification.

Usage:
    uv run python scripts/train_text_classifier.py --model arapoem --task genre
    uv run python scripts/train_text_classifier.py --model mbert   --task genre
    uv run python scripts/train_text_classifier.py --model arapoem --task emotion_text

    # Transfer-learning: freeze encoder, train head only
    uv run python scripts/train_text_classifier.py --freeze-encoder
    uv run python scripts/train_text_classifier.py --freeze-encoder --unfreeze-after 5
    uv run python scripts/train_text_classifier.py --epochs 10 --patience 5
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from loguru import logger
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import (
    mbert_genre_config,
    arapoem_genre_config,
    arapoem_emotion_config,
)
from src.data.labels import (
    GENRE_CLASSES,
    EMOTION_CLASSES,
    encode_genre,
    encode_emotion,
    EMOTION_MERGE_PROFILES,
    get_merged_emotion_classes,
    encode_emotion_with_profile,
)
from src.training.trainer import (
    TensorBoardLogger,
    EarlyStopper,
    get_optimizer,
    get_scheduler,
    get_one_cycle_scheduler,
    unfreeze_next_layer_group,
    set_seed,
)
from src.training.sanity import run_all_checks


# ── Focal Loss (handles Ghazal>>Tareef imbalance) ────────────────────────────


class FocalLoss(nn.Module):
    """
    Focal loss down-weights easy majority-class samples (Ghazal = 27% of data)
    so the model is forced to learn minority classes (Tareef = 0.8%).
    γ=2 is the standard value; higher γ = more focus on hard examples.
    """

    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight  # per-class inverse-frequency weights (optional)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)  # probability of correct class
        loss = ((1 - pt) ** self.gamma) * ce  # downweight easy samples
        return loss.mean()


# ── Text Dataset ─────────────────────────────────────────────────────────────


class NabatiTextDataset(Dataset):
    """
    Reads a JSONL split and returns tokenised Arabic text + integer label.
    Input field: text_corrected (human-corrected ground-truth transcript)
    Label field: genre_en  (for task='genre')
                 emotion_text (for task='emotion_text')

    context_window (Strategy 1 — sliding window):
        1 → single clip (original behaviour)
        3 → concatenate [clip_{i-1}, clip_i, clip_{i+1}] into one input
        Clips are ordered by their 'start' timestamp within each poem.
        For edge clips the missing neighbour is omitted (no padding text).
        This triples the usable context while staying within AraPoemBERT's
        32-token limit at the median clip length (5 words × 3 = 15 words).
    """

    def __init__(
        self,
        jsonl_path: Path,
        tokenizer,
        task: str,
        max_seq_len: int = 128,
        context_window: int = 1,
        emotion_merge_profile: str = "none",
    ):
        self.tokenizer = tokenizer
        self.task = task
        self.max_seq_len = max_seq_len
        self.emotion_merge_profile = emotion_merge_profile
        self.samples: list[dict] = []

        if task == "genre":
            encode_fn = encode_genre
        elif emotion_merge_profile != "none":
            encode_fn = lambda s: encode_emotion_with_profile(s, emotion_merge_profile)  # noqa: E731
        else:
            encode_fn = encode_emotion
        label_key = "genre_en" if task == "genre" else "emotion_text"
        half = context_window // 2  # window=3 → half=1

        # ── Two-pass: group by poem, sort by start time ───────────────────
        poem_records: dict[str, list] = defaultdict(list)
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                poem_records[rec.get("source_poem", "unknown")].append(rec)

        for pid in poem_records:
            poem_records[pid].sort(key=lambda r: r.get("start", 0))

        skipped = 0
        window_sizes: list[int] = []  # for logging

        for pid, clips in poem_records.items():
            n = len(clips)
            for i, rec in enumerate(clips):
                text_i = rec.get("text_corrected", "").strip()
                label_str = rec.get(label_key, "")
                label_id = encode_fn(label_str)

                if not text_i or label_id == -1:
                    skipped += 1
                    continue

                if context_window > 1:
                    parts: list[str] = []
                    # Previous clip(s): ordered earliest → latest
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
                    window_sizes.append(len(parts))
                else:
                    text = text_i

                poem_id = pid + "|" + rec.get("poet_en", "")
                self.samples.append(
                    {"text": text, "label": label_id, "poem_id": poem_id}
                )

        if context_window > 1:
            logger.info(
                f"Sliding window={context_window}: "
                f"avg_clips_per_window={np.mean(window_sizes):.2f}  "
                f"full_window_rate={sum(w == context_window for w in window_sizes) / len(window_sizes) * 100:.1f}%"
            )
        logger.info(
            f"Loaded {len(self.samples)} samples from {jsonl_path.name} "
            f"(skipped {skipped} with missing label)"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
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
            "label": torch.tensor(sample["label"], dtype=torch.long),
            "poem_id": sample["poem_id"],  # string — DataLoader collates as list
        }


# ── Confusion Matrix Figure ───────────────────────────────────────────────────


def make_confusion_figure(
    y_true: list, y_pred: list, class_names: list, title: str
) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    fig, ax = plt.subplots(
        figsize=(max(8, len(class_names)), max(6, len(class_names) - 2))
    )
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[c.split("(")[0].strip() for c in class_names],
        yticklabels=[c.split("(")[0].strip() for c in class_names],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    fig.tight_layout()
    return fig


# ── One epoch of training ─────────────────────────────────────────────────────


def train_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    criterion,
    device,
    tb_logger,
    global_step: int,
    log_every: int,
    accum_steps: int = 1,
) -> tuple[float, float, int]:
    """
    Gradient accumulation: accumulate gradients over
    `accum_steps` micro-batches, then step.  Effective batch = batch_size × accum_steps.
    This stabilises BERT fine-tuning on small datasets without requiring larger GPU memory.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    n_correct = 0
    n_total = 0

    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        # Divide loss so accumulated gradients equal a single full-batch gradient
        loss = criterion(logits, labels) / accum_steps
        loss.backward()

        total_loss += loss.item() * accum_steps  # log the un-scaled loss
        n_batches += 1

        preds = logits.detach().argmax(dim=-1)
        n_correct += (preds == labels).sum().item()
        n_total += labels.size(0)

        is_last_batch = (i + 1) == len(loader)
        is_accum_boundary = (i + 1) % accum_steps == 0

        if is_accum_boundary or is_last_batch:
            # Gradient clipping — prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % log_every == 0:
                lr = scheduler.get_last_lr()[0]
                tb_logger.log_step(loss.item() * accum_steps, lr, global_step)

    train_acc = n_correct / max(n_total, 1)
    return total_loss / max(n_batches, 1), train_acc, global_step


# ── One epoch of evaluation ───────────────────────────────────────────────────


def eval_epoch(
    model,
    loader,
    criterion,
    device,
) -> tuple[float, float, float, list, list, list, list]:
    """Returns (loss, macro_f1, acc, preds, true_labels, softmax_probs, poem_ids)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds, all_true, all_probs, all_poem_ids = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            poem_ids = batch["poem_id"]  # list[str]

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            loss = criterion(logits, labels)

            total_loss += loss.item()
            n_batches += 1

            probs = torch.softmax(logits, dim=-1).cpu().tolist()
            preds = [max(range(len(p)), key=lambda i: p[i]) for p in probs]
            all_preds.extend(preds)
            all_true.extend(labels.cpu().tolist())
            all_probs.extend(probs)
            all_poem_ids.extend(poem_ids)

    val_loss = total_loss / max(n_batches, 1)
    val_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
    val_acc = sum(p == t for p, t in zip(all_preds, all_true)) / len(all_true)
    return val_loss, val_f1, val_acc, all_preds, all_true, all_probs, all_poem_ids


def poem_aggregate_eval(
    all_probs: list[list[float]],
    all_true: list[int],
    all_poem_ids: list[str],
    class_names: list[str],
) -> tuple[float, float]:
    """
    Average softmax probabilities across all clips of the same poem, then argmax.
    Genre is a poem-level property; averaging 24 noisy 8-token predictions cancels
    clip-level noise and reveals the genre signal.
    Returns (poem_macro_f1, poem_accuracy).
    """
    poem_probs: dict[str, list] = defaultdict(list)
    poem_labels: dict[str, int] = {}
    for prob, label, pid in zip(all_probs, all_true, all_poem_ids):
        poem_probs[pid].append(prob)
        poem_labels[pid] = label  # all clips of same poem share the same label

    poem_preds, poem_true = [], []
    for pid, probs_list in poem_probs.items():
        avg = [
            sum(p[i] for p in probs_list) / len(probs_list)
            for i in range(len(probs_list[0]))
        ]
        poem_preds.append(max(range(len(avg)), key=lambda i: avg[i]))
        poem_true.append(poem_labels[pid])

    f1 = f1_score(poem_true, poem_preds, average="macro", zero_division=0)
    acc = sum(p == t for p, t in zip(poem_preds, poem_true)) / max(len(poem_true), 1)

    short = [c.split("(")[0].strip() for c in class_names]
    present_ids = sorted(set(poem_true))
    report = classification_report(
        poem_true,
        poem_preds,
        labels=present_ids,
        target_names=[short[i] for i in present_ids],
        zero_division=0,
    )
    logger.info(
        f"\nPoem-level classification report ({len(poem_probs)} poems):\n{report}"
    )
    return f1, acc


# ── Main ──────────────────────────────────────────────────────────────────────


def main(args):
    # ── Config ──────────────────────────────────────────────────────────────
    if args.model == "mbert":
        cfg = mbert_genre_config()
    elif args.model == "arapoem" and args.task == "genre":
        cfg = arapoem_genre_config()
    elif args.model == "arapoem" and args.task == "emotion_text":
        cfg = arapoem_emotion_config()
    else:
        raise ValueError(f"Unknown --model={args.model} --task={args.task}")

    # CLI overrides (allow quick experiments without touching config.py)
    if args.epochs is not None:
        cfg = cfg.model_copy(update={"epochs": args.epochs})
    if args.patience is not None:
        cfg = cfg.model_copy(update={"patience": args.patience})
    if args.lr is not None:
        cfg = cfg.model_copy(update={"learning_rate": args.lr})
    if args.seed is not None:
        cfg = cfg.model_copy(update={"seed": args.seed})

    freeze_encoder = args.freeze_encoder
    unfreeze_after = (
        args.unfreeze_after
    )  # epoch index (0-based) to unfreeze last 2 layers

    # ── CLI overrides for knob sweep ─────────────────────────────────────────
    if args.dropout is not None:
        cfg = cfg.model_copy(update={"dropout": args.dropout})
    if args.weight_decay is not None:
        cfg = cfg.model_copy(update={"weight_decay": args.weight_decay})
    if args.loss == "focal":
        cfg = cfg.model_copy(
            update={"use_focal_loss": True, "focal_gamma": args.focal_gamma}
        )
    elif args.loss == "ce":
        cfg = cfg.model_copy(update={"use_focal_loss": False})

    # Emotion merge profile
    emotion_merge_profile: str = (
        args.emotion_merge_profile if args.task == "emotion_text" else "none"
    )

    # Pydantic already validated all field constraints at construction time
    logger.info(
        f"Config: {cfg.model_name} | task={cfg.task} | epochs={cfg.epochs} | "
        f"lr={cfg.learning_rate:.2e} | batch={cfg.batch_size} | "
        f"freeze_encoder={freeze_encoder} | unfreeze_after={unfreeze_after} | "
        f"emotion_merge_profile={emotion_merge_profile} | "
        f"dropout={cfg.dropout:.2f} | weight_decay={cfg.weight_decay:.4f} | "
        f"loss={'focal' if cfg.use_focal_loss else 'ce'} | "
        f"seed={cfg.seed}"
    )

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Label setup ──────────────────────────────────────────────────────────
    if cfg.task == "genre":
        class_names = GENRE_CLASSES
    elif emotion_merge_profile != "none":
        class_names = get_merged_emotion_classes(emotion_merge_profile)
        logger.info(
            f"Emotion merge profile '{emotion_merge_profile}': "
            f"{len(EMOTION_CLASSES)} → {len(class_names)} classes"
        )
    else:
        class_names = EMOTION_CLASSES
    num_classes = len(class_names)

    # ── Tokenizer + Model ────────────────────────────────────────────────────
    logger.info(f"Loading tokenizer + model: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, local_files_only=True)
    model_cfg = AutoConfig.from_pretrained(cfg.model_name, local_files_only=True)
    model_cfg.num_labels = (
        num_classes  # must be set in config when passing config= explicitly
    )
    if cfg.dropout > 0.0:
        model_cfg.hidden_dropout_prob = cfg.dropout
        model_cfg.attention_probs_dropout_prob = cfg.dropout
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        config=model_cfg,
        ignore_mismatched_sizes=True,  # head will be replaced
        local_files_only=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded: {total_params:,} parameters")
    assert total_params <= 200_000_000, (
        f"Model exceeds 200M param constraint: {total_params:,}"
    )

    unfreeze_n = args.unfreeze_last_n
    if unfreeze_n is not None and unfreeze_n != "full":
        n_layers = int(unfreeze_n)
        for name, param in model.named_parameters():
            param.requires_grad = False  # freeze everything first
        # Always unfreeze the classifier head
        for name, param in model.named_parameters():
            if "classifier" in name or "pooler" in name:
                param.requires_grad = True
        # Unfreeze last N transformer layers
        if n_layers > 0:
            encoder_layers = model.bert.encoder.layer
            for layer in list(encoder_layers)[-n_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_p = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Static partial fine-tune: unfreeze_last_n={n_layers} | "
            f"trainable={trainable:,}/{total_p:,} ({trainable / total_p * 100:.1f}%)"
        )
        # Disable gradual unfreezing and freeze_encoder mode for this path
        freeze_encoder = False
        cfg = cfg.model_copy(update={"gradual_unfreeze_epochs": 0})

    unfreeze_params: list = []  # params pre-registered at lr=0 for progressive unfreeze
    if freeze_encoder:
        for name, param in model.named_parameters():
            if "classifier" not in name and "pooler" not in name:
                param.requires_grad = False

        # Replace the classification head with a deeper head + dropout
        hidden = model.config.hidden_size  # 768 for BERT-base
        model.classifier = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        ).to(device)

        # Pre-collect last 2 BERT encoder layer params for progressive unfreezing.
        # They stay requires_grad=False until the unfreeze epoch; we pre-register
        # them in the optimizer at lr=0.0 so the scheduler never sees a group-count
        # change (which would cause ValueError: zip() argument 2 is shorter).
        if unfreeze_after is not None:
            encoder_layers = model.bert.encoder.layer
            for layer in list(encoder_layers)[-2:]:
                for p in layer.parameters():
                    unfreeze_params.append(p)
            # Keep frozen (already False from loop above), just collect references

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_p = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Frozen encoder: {total_p - trainable:,} frozen, "
            f"{trainable:,} trainable ({trainable / total_p * 100:.1f}%)"
        )
        if unfreeze_params:
            logger.info(
                f"  {len(unfreeze_params)} params pre-registered at lr=0 "
                f"for progressive unfreeze at epoch {unfreeze_after}"
            )
        # Disable gradual unfreezing and discriminative LR in frozen mode
        cfg = cfg.model_copy(update={"gradual_unfreeze_epochs": 0})

    # Verify max_seq_len does not exceed model's max_position_embeddings.
    # AraPoemBERT: max_position_embeddings=32 (designed for single poetry verses).
    model_max_pos = model.config.max_position_embeddings
    if cfg.max_seq_len > model_max_pos:
        raise ValueError(
            f"max_seq_len={cfg.max_seq_len} exceeds model's max_position_embeddings={model_max_pos}. "
            f"Set max_seq_len<={model_max_pos} in the config."
        )
    logger.info(f"Sequence length: {cfg.max_seq_len} (model max: {model_max_pos})")

    # ── Datasets + Loaders ───────────────────────────────────────────────────
    cw = args.context_window
    if cw > 1:
        logger.info(f"Strategy 1: sliding context window = {cw} clips per input")
    train_ds = NabatiTextDataset(
        cfg.train_jsonl,
        tokenizer,
        cfg.task,
        cfg.max_seq_len,
        cw,
        emotion_merge_profile=emotion_merge_profile,
    )
    val_ds = NabatiTextDataset(
        cfg.val_jsonl,
        tokenizer,
        cfg.task,
        cfg.max_seq_len,
        cw,
        emotion_merge_profile=emotion_merge_profile,
    )
    test_ds = NabatiTextDataset(
        cfg.test_jsonl,
        tokenizer,
        cfg.task,
        cfg.max_seq_len,
        cw,
        emotion_merge_profile=emotion_merge_profile,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0
    )

    # Ghazal=27% of data but minority classes like Nabd=0.8% → balanced weights
    # force the model to attend equally to all classes regardless of frequency.
    # Only compute weights for classes present in training; absent classes get 1.0.
    train_labels = [s["label"] for s in train_ds.samples]
    present_classes = np.unique(train_labels)
    class_w_present = compute_class_weight(
        "balanced",
        classes=present_classes,
        y=train_labels,
    )
    # If max/min ratio > 20x (extreme imbalance), use sqrt to moderate weights.
    # emotion_text has 98x ratio (8 Longing vs 789 Contemplation); genre has 8x.
    raw_ratio = class_w_present.max() / class_w_present.min()
    if raw_ratio > 20.0:
        class_w_present = np.sqrt(class_w_present)
        logger.info(
            f"Class weight ratio {raw_ratio:.1f}x > 20x — applying sqrt to moderate imbalance"
        )
    class_w = np.ones(num_classes, dtype=np.float32)
    for cls_id, w in zip(present_classes, class_w_present):
        class_w[cls_id] = w
    class_weights = torch.tensor(class_w, dtype=torch.float32).to(device)
    logger.info(
        f"Class weights: min={class_w.min():.2f}, max={class_w.max():.2f}, ratio={class_w.max() / class_w.min():.1f}x"
    )

    # ── Loss function ─────────────────────────────────────────────────────────
    if cfg.use_focal_loss:
        criterion = FocalLoss(gamma=cfg.focal_gamma, weight=class_weights).to(device)
        logger.info(
            f"Using FocalLoss(γ={cfg.focal_gamma}) + balanced class weights + "
            f"label_smoothing={cfg.label_smoothing}"
        )
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights, label_smoothing=cfg.label_smoothing
        )
        logger.info(
            f"Using CrossEntropyLoss + balanced class weights + "
            f"label_smoothing={cfg.label_smoothing}"
        )

    # ── Optimizer (discriminative LR) ────────────────────────────────────────
    if freeze_encoder:
        # Head-only optimizer: flat lr=1e-3, weight_decay=0.01
        head_lr = args.lr if args.lr is not None else 1e-3
        head_params = [p for p in model.parameters() if p.requires_grad]
        if unfreeze_params:
            # Pre-register the to-be-unfrozen params at lr=0 so the scheduler
            # is created with the correct final group count. Their lr will be
            # updated in-place at the unfreeze epoch (no add_param_group needed).
            optimizer = torch.optim.AdamW(
                [
                    {"params": head_params, "lr": head_lr, "weight_decay": 0.01},
                    {"params": unfreeze_params, "lr": 0.0, "weight_decay": 0.01},
                ]
            )
            logger.info(
                f"Head-only optimizer: AdamW lr={head_lr:.2e} | "
                f"encoder-group pre-registered at lr=0.0 (will activate epoch {unfreeze_after})"
            )
        else:
            optimizer = torch.optim.AdamW(head_params, lr=head_lr, weight_decay=0.01)
            logger.info(
                f"Head-only optimizer: AdamW lr={head_lr:.2e} weight_decay=0.01"
            )
    else:
        optimizer = get_optimizer(
            model,
            base_lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            discriminative_lr_decay=cfg.discriminative_lr_decay,
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

    # ── TensorBoard ───────────────────────────────────────────────────────────
    model_tag = "mbert" if "multilingual" in cfg.model_name else "arapoem"
    tb_logger = TensorBoardLogger(cfg.tensorboard_dir, model_tag, cfg.task)

    # Log hyperparameters table (visible in TensorBoard HPARAMS tab)
    tb_logger.log_hparams(
        hparam_dict={
            "model": cfg.model_name,
            "task": cfg.task,
            "lr": cfg.learning_rate,
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
            "focal_loss": cfg.use_focal_loss,
            "focal_gamma": cfg.focal_gamma,
            "disc_lr_decay": cfg.discriminative_lr_decay,
        },
        metric_dict={"hparam/val_macro_f1": 0.0},
    )

    # ── Pre-training sanity checks ────────────────────────────────────────────
    if not freeze_encoder:
        logger.info("Running pre-training sanity checks...")
        run_all_checks(model, train_loader, criterion, optimizer, num_classes, device)
    else:
        logger.info("Skipping sanity checks in freeze_encoder mode.")

    # Restore clean optimizer + scheduler state after sanity overfit test
    if not freeze_encoder:
        optimizer = get_optimizer(
            model, cfg.learning_rate, cfg.weight_decay, cfg.discriminative_lr_decay
        )
    scheduler = get_scheduler(optimizer, total_steps, cfg.warmup_ratio)
    logger.info(
        f"Grad accumulation: {cfg.grad_accum_steps} steps → "
        f"effective batch={cfg.batch_size * cfg.grad_accum_steps}, "
        f"optimizer steps/epoch={len(train_loader) // cfg.grad_accum_steps}"
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    early_stopper = EarlyStopper(
        cfg.patience, cfg.output_dir, f"{model_tag}_{cfg.task}"
    )
    global_step = 0
    best_val_f1 = 0.0

    logger.info("=" * 60)
    logger.info(
        f"Starting training: {cfg.epochs} epochs, {len(train_loader)} steps/epoch"
    )
    logger.info("=" * 60)

    for epoch in range(cfg.epochs):
        # Bind epoch context to all logger calls in this iteration
        epoch_logger = logger.bind(
            epoch=epoch + 1, total_epochs=cfg.epochs, task=cfg.task
        )

        # Progressive unfreezing: unfreeze last 2 BERT encoder layers after N epochs.
        # We update the pre-registered param group's lr in-place instead of calling
        # add_param_group(), which would break PyTorch schedulers (they zip over
        # param_groups using the count fixed at scheduler creation time).
        if freeze_encoder and unfreeze_after is not None and epoch == unfreeze_after:
            for p in unfreeze_params:
                p.requires_grad = True
            fine_lr = (args.lr if args.lr is not None else 1e-3) * 0.1
            # The encoder group is the last param group (index -1), pre-registered at lr=0
            optimizer.param_groups[-1]["lr"] = fine_lr
            trainable_now = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            logger.info(
                f"[Epoch {epoch + 1}] Progressive unfreeze: last 2 BERT layers activated "
                f"(lr={fine_lr:.2e}). Trainable params: {trainable_now:,}"
            )

        # Gradual unfreezing: unfreeze one layer group every N epochs
        if cfg.gradual_unfreeze_epochs > 0:
            unfreeze_next_layer_group(model, epoch, cfg.gradual_unfreeze_epochs)

        # Train
        train_loss, train_acc, global_step = train_epoch(
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

        # Log gradient histograms
        tb_logger.log_histograms(model, epoch)

        # Validate (poem_ids/probs not needed during training loop)
        val_loss, val_f1, val_acc, val_preds, val_true, _, _ = eval_epoch(
            model, val_loader, criterion, device
        )

        # TensorBoard epoch scalars
        tb_logger.log_epoch(epoch, train_loss, train_acc, val_loss, val_f1, val_acc)

        epoch_logger.info(
            f"Epoch {epoch + 1:3d}/{cfg.epochs} | "
            f"train_loss={train_loss:.4f} | train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_F1={val_f1:.4f} | "
            f"val_acc={val_acc:.3f}"
        )

        # Log val confusion matrix to TensorBoard every 5 epochs
        if (epoch + 1) % 5 == 0:
            fig = make_confusion_figure(
                val_true,
                val_preds,
                class_names,
                f"Val Confusion Matrix — Epoch {epoch + 1}",
            )
            tb_logger.log_confusion_matrix(fig, f"confusion/{cfg.task}/val", epoch)
            plt.close(fig)

        # Early stopping
        if early_stopper.step(val_f1, model):
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

    # ── Final test evaluation on best checkpoint ──────────────────────────────
    logger.info(f"Loading best checkpoint: {early_stopper.best_ckpt}")
    model.load_state_dict(torch.load(early_stopper.best_ckpt, map_location=device))

    test_loss, test_f1, test_acc, test_preds, test_true, test_probs, test_poem_ids = (
        eval_epoch(model, test_loader, criterion, device)
    )

    logger.success(
        f"TEST RESULTS (clip-level) | loss={test_loss:.4f} | "
        f"Macro-F1={test_f1:.4f} | accuracy={test_acc:.4f}"
    )

    # Poem-level majority vote (Approach 1)
    poem_f1, poem_acc = poem_aggregate_eval(
        test_probs, test_true, test_poem_ids, class_names
    )
    logger.success(
        f"TEST RESULTS (poem-level) | "
        f"Macro-F1={poem_f1:.4f} | accuracy={poem_acc:.4f} "
        f"(averaged softmax over all clips per poem)"
    )

    report_dir = Path("outputs/reports")
    figures_dir = Path("outputs/figures")
    report_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    # Use run_id for unique filenames; fall back to default for backwards compat
    _run_id_safe = (
        getattr(args, "run_id", None)
        or f"{model_tag}_{cfg.task}_{emotion_merge_profile}"
    ).replace("/", "_")
    fig_path = figures_dir / f"emotion_confusion_{_run_id_safe}.png"
    fig = make_confusion_figure(
        test_true,
        test_preds,
        class_names,
        f"Test Confusion Matrix — {model_tag.upper()} | {cfg.task} | Macro-F1={test_f1:.3f}",
    )
    fig.savefig(fig_path, bbox_inches="tight", dpi=150)
    logger.info(f"Confusion matrix saved → {fig_path}")
    plt.close(fig)

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

    report_path = report_dir / f"{model_tag}_{cfg.task}_report.txt"
    report_path.write_text(report_str)

    tb_logger.log_hparams(
        hparam_dict={
            "model": cfg.model_name,
            "task": cfg.task,
            "lr": cfg.learning_rate,
            "batch_size": cfg.batch_size,
            "epochs": cfg.epochs,
        },
        metric_dict={"hparam/test_macro_f1": test_f1, "hparam/test_acc": test_acc},
    )
    tb_logger.close()

    # Top-2 accuracy: correct if true label is in top-2 predictions
    top2_correct = sum(
        1
        for prob, true in zip(test_probs, test_true)
        if true in sorted(range(len(prob)), key=lambda i: prob[i], reverse=True)[:2]
    )
    top2_acc = top2_correct / max(len(test_true), 1)

    # Expected Calibration Error (ECE) — 10-bin uniform binning
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    confidences = np.array([max(p) for p in test_probs])
    correctness = np.array(
        [int(p == t) for p, t in zip(test_preds, test_true)], dtype=float
    )
    ece = 0.0
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() > 0:
            bin_conf = confidences[mask].mean()
            bin_acc = correctness[mask].mean()
            ece += mask.sum() / len(test_true) * abs(bin_acc - bin_conf)

    logger.info(f"Top-2 accuracy: {top2_acc:.4f}")
    logger.info(f"ECE (10-bin):   {ece:.4f}")

    run_id = (
        getattr(args, "run_id", None)
        or f"{model_tag}_{cfg.task}_{emotion_merge_profile}"
    )
    report = {
        "run_id": run_id,
        "task": cfg.task,
        "model": cfg.model_name,
        "seed": cfg.seed,
        "context_window": getattr(args, "context_window", 1),
        "emotion_merge_profile": emotion_merge_profile,
        "loss": "focal" if cfg.use_focal_loss else "ce",
        "focal_gamma": cfg.focal_gamma if cfg.use_focal_loss else None,
        "dropout": cfg.dropout,
        "weight_decay": cfg.weight_decay,
        "unfreeze_last_n": getattr(args, "unfreeze_last_n", "full"),
        "epochs_trained": cfg.epochs,
        "val_hard_macro_f1_clip": round(best_val_f1, 4),
        "test_hard_macro_f1_clip": round(test_f1, 4),
        "test_hard_macro_f1_poem": round(poem_f1, 4),
        "test_accuracy_clip": round(test_acc, 4),
        "test_top2_acc": round(top2_acc, 4),
        "test_ece": round(float(ece), 4),
        "num_classes": num_classes,
        "checkpoint_path": str(early_stopper.best_ckpt),
        "confusion_matrix_path": str(fig_path),
        "report_txt_path": str(report_path),
    }

    # Save to specified path or default
    json_report_path = Path(
        getattr(args, "report_path", None)
        or report_dir / f"emotion_eval_{emotion_merge_profile}.json"
    )
    json_report_path.parent.mkdir(parents=True, exist_ok=True)
    json_report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.info(f"JSON report → {json_report_path}")

    logger.success(
        f"Training complete. Best val Macro-F1={best_val_f1:.4f}, "
        f"test Macro-F1={test_f1:.4f} | Top-2={top2_acc:.4f} | ECE={ece:.4f}"
    )
    logger.info(f"View curves: tensorboard --logdir {cfg.tensorboard_dir}")
    return test_f1


if __name__ == "__main__":
    logger.add("logs/train_text.log", rotation="10 MB")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=["arapoem", "mbert"],
        default="arapoem",
        help="arapoem = faisalq/bert-base-arapoembert  |  mbert = bert-base-multilingual-cased",
    )
    parser.add_argument(
        "--task",
        choices=["genre", "emotion_text"],
        default="genre",
        help="Which label column to predict.",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze BERT encoder; train only a 768→256→11 head with dropout=0.5.",
    )
    parser.add_argument(
        "--unfreeze-after",
        type=int,
        default=None,
        metavar="EPOCH",
        help="Unfreeze last 2 BERT layers at this epoch (0-based) with lr×0.1. "
        "Only used with --freeze-encoder.",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help="Override config epochs."
    )
    parser.add_argument(
        "--patience", type=int, default=None, help="Override early-stopping patience."
    )
    parser.add_argument(
        "--lr", type=float, default=None, help="Override learning rate."
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=1,
        metavar="N",
        help="Strategy 1: sliding window. N=1 disables (default), N=3 concatenates "
        "[clip_{i-1}, clip_i, clip_{i+1}] as one input (triples context, "
        "stays within AraPoemBERT's 32-token limit at median clip length).",
    )

    # ── Knob sweep arguments ─────────────────────────────────────────────────
    parser.add_argument(
        "--emotion-merge-profile",
        choices=list(EMOTION_MERGE_PROFILES.keys()),
        default="none",
        help="Label merge profile for emotion task. 'rare_merge_v1' merges Longing→Sorrow, "
        "Compassion→Delicate Love, Humor→Neutral (reduces 12→9 classes).",
    )
    parser.add_argument(
        "--loss",
        choices=["ce", "focal"],
        default=None,
        help="Loss function override. 'focal' uses FocalLoss with --focal-gamma.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focal loss gamma (only used when --loss focal). Default: 2.0",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="Override model dropout (hidden + attention). Default: use config (0.1).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=None,
        dest="weight_decay",
        help="Override AdamW weight decay. Default: use config (0.01).",
    )
    parser.add_argument(
        "--unfreeze-last-n",
        default=None,
        dest="unfreeze_last_n",
        help="Static partial fine-tuning: '0'=head-only, '2'/'3'=last N layers+head, "
        "'full'=all layers (default behavior). Overrides gradual-unfreeze.",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        dest="run_id",
        help="Experiment identifier for report filenames.",
    )
    parser.add_argument(
        "--report-path",
        default=None,
        dest="report_path",
        help="Path to write JSON evaluation report. Defaults to outputs/reports/emotion_eval_<profile>.json",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override training seed for multi-seed stability runs.",
    )
    args = parser.parse_args()
    main(args)

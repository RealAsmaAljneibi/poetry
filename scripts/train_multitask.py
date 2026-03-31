"""
scripts/train_multitask.py

Multi-task learning: joint genre + emotion classification (Knob 5 / Lecture 3).

Architecture:
  Shared AraPoemBERT encoder (CLS pooler output, 768-dim)
  ├── genre_head:   Linear(768, 8)   → 8-class genre
  └── emotion_head: Linear(768, N)   → N-class emotion (with optional merge profile)

Loss:
  total_loss = genre_weight * genre_loss + emotion_weight * emotion_loss

Rationale:
  Genre labels are cleaner and stronger (macro-F1=0.51 vs emotion 0.21).
  Sharing the encoder regularises the emotion task while genre provides extra
  gradient signal — a classic multi-objective technique (Lecture 3 multi-task section).

Course techniques:
  Week 3: multi-objective loss, focal loss, sqrt class weights
  Week 4: discriminative LR, gradual unfreezing, gradient accumulation
  Week 5: TensorBoard, Pydantic config, sanity checks

Usage:
    uv run python scripts/train_multitask.py
    uv run python scripts/train_multitask.py --emotion-merge-profile rare_merge_v1
    uv run python scripts/train_multitask.py --genre-weight 0.7 --emotion-weight 0.3

Output:
    outputs/reports/multitask_eval.json
    outputs/figures/multitask_emotion_confusion.png
    outputs/figures/multitask_genre_confusion.png
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.labels import (
    GENRE_CLASSES, encode_genre, encode_emotion,
    EMOTION_MERGE_PROFILES,
    get_merged_emotion_classes,
    encode_emotion_with_profile,
)
from src.training.trainer import (
    TensorBoardLogger, get_optimizer, get_scheduler,
    set_seed,
)

PROJECT_ROOT = Path(__file__).parent.parent
REPORT_DIR   = PROJECT_ROOT / "outputs/reports"
FIGURES_DIR  = PROJECT_ROOT / "outputs/figures"
MODEL_NAME   = "faisalq/bert-base-arapoembert"
MAX_SEQ_LEN  = 32


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


# ── Dataset ───────────────────────────────────────────────────────────────────

class MultitaskDataset(torch.utils.data.Dataset):
    """Loads clips with both genre and emotion labels."""

    def __init__(
        self,
        jsonl_path: Path,
        tokenizer,
        max_seq_len: int = MAX_SEQ_LEN,
        emotion_merge_profile: str = "none",
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: list[dict] = []

        if emotion_merge_profile != "none":
            encode_emo = lambda s: encode_emotion_with_profile(s, emotion_merge_profile)  # noqa: E731
        else:
            encode_emo = encode_emotion

        skipped = 0
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                text = rec.get("text_corrected", "").strip()
                g_id = encode_genre(rec.get("genre_en", ""))
                e_id = encode_emo(rec.get("emotion_text", ""))
                if not text or g_id == -1 or e_id == -1:
                    skipped += 1
                    continue
                poem_id = rec.get("source_poem", "") + "|" + rec.get("poet_en", "")
                self.samples.append({
                    "text": text,
                    "genre_label": g_id,
                    "emotion_label": e_id,
                    "poem_id": poem_id,
                })
        logger.info(f"Loaded {len(self.samples)} samples from {jsonl_path.name} "
                    f"(skipped {skipped})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        enc = self.tokenizer(
            s["text"], max_length=self.max_seq_len,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "genre_label":    torch.tensor(s["genre_label"], dtype=torch.long),
            "emotion_label":  torch.tensor(s["emotion_label"], dtype=torch.long),
            "poem_id":        s["poem_id"],
        }


# ── Multi-task model ──────────────────────────────────────────────────────────

class MultitaskModel(nn.Module):
    """Shared AraPoemBERT encoder + 2 classification heads."""

    def __init__(self, model_name: str, n_genre: int, n_emotion: int, dropout: float = 0.1):
        super().__init__()
        from transformers import AutoConfig as _AC
        cfg = _AC.from_pretrained(model_name, local_files_only=True)
        cfg.hidden_dropout_prob = dropout
        cfg.attention_probs_dropout_prob = dropout
        self.encoder    = AutoModel.from_pretrained(model_name, config=cfg, local_files_only=True)
        hidden          = self.encoder.config.hidden_size  # 768
        self.genre_head   = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_genre))
        self.emotion_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_emotion))

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output          # (B, 768) — CLS representation
        return self.genre_head(pooled), self.emotion_head(pooled)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_class_weights(labels: list[int], n_classes: int, device) -> torch.Tensor:
    present = np.unique(labels)
    w = compute_class_weight("balanced", classes=present, y=labels)
    ratio = w.max() / w.min()
    if ratio > 20.0:
        w = np.sqrt(w)
        logger.info(f"Class weight ratio {ratio:.1f}x > 20x → sqrt applied")
    full = np.ones(n_classes, dtype=np.float32)
    for cls_id, wt in zip(present, w):
        full[cls_id] = wt
    return torch.tensor(full, dtype=torch.float32).to(device)


def make_confusion_figure(y_true, y_pred, class_names, title) -> plt.Figure:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    n  = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n), max(6, n - 2)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=[c.split("(")[0].strip() for c in class_names],
                yticklabels=[c.split("(")[0].strip() for c in class_names], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    fig.tight_layout()
    return fig


def poem_f1(all_probs, all_true, all_poem_ids, n_classes) -> float:
    poem_probs:  dict[str, list] = defaultdict(list)
    poem_labels: dict[str, int]  = {}
    for prob, label, pid in zip(all_probs, all_true, all_poem_ids):
        poem_probs[pid].append(prob)
        poem_labels[pid] = label
    p_preds, p_true = [], []
    for pid, probs_list in poem_probs.items():
        avg = [sum(p[i] for p in probs_list) / len(probs_list) for i in range(n_classes)]
        p_preds.append(max(range(n_classes), key=lambda i: avg[i]))
        p_true.append(poem_labels[pid])
    return f1_score(p_true, p_preds, average="macro", zero_division=0)


# ── Training ──────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, scheduler, g_crit, e_crit,
                g_weight, e_weight, device, accum_steps):
    model.train()
    total_loss = 0.0
    n_batches = 0
    optimizer.zero_grad()
    for i, batch in enumerate(loader):
        ids  = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        g_labels = batch["genre_label"].to(device)
        e_labels = batch["emotion_label"].to(device)

        g_logits, e_logits = model(ids, mask)
        loss = (g_weight * g_crit(g_logits, g_labels) +
                e_weight * e_crit(e_logits, e_labels)) / accum_steps
        loss.backward()
        total_loss += loss.item() * accum_steps
        n_batches  += 1

        if (i + 1) % accum_steps == 0 or (i + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return total_loss / max(n_batches, 1)


def eval_epoch(model, loader, g_crit, e_crit, g_weight, e_weight, device, n_g, n_e):
    model.eval()
    g_preds, g_true, g_probs = [], [], []
    e_preds, e_true, e_probs = [], [], []
    poem_ids = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in loader:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            g_labels = batch["genre_label"].to(device)
            e_labels = batch["emotion_label"].to(device)

            g_logits, e_logits = model(ids, mask)
            loss = (g_weight * g_crit(g_logits, g_labels) +
                    e_weight * e_crit(e_logits, e_labels))
            total_loss += loss.item()
            n_batches += 1

            g_probs.extend(torch.softmax(g_logits, -1).cpu().tolist())
            e_probs.extend(torch.softmax(e_logits, -1).cpu().tolist())
            g_preds.extend(g_logits.argmax(-1).cpu().tolist())
            e_preds.extend(e_logits.argmax(-1).cpu().tolist())
            g_true.extend(g_labels.cpu().tolist())
            e_true.extend(e_labels.cpu().tolist())
            poem_ids.extend(batch["poem_id"])

    g_f1 = f1_score(g_true, g_preds, average="macro", zero_division=0)
    e_f1 = f1_score(e_true, e_preds, average="macro", zero_division=0)
    avg_loss = total_loss / max(n_batches, 1)
    return avg_loss, g_f1, e_f1, g_preds, e_preds, g_true, e_true, g_probs, e_probs, poem_ids


# ── Main ─────────────────────────────────────────────────────────────────────

def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Device: {device}")

    emotion_merge_profile = args.emotion_merge_profile
    emotion_classes = get_merged_emotion_classes(emotion_merge_profile)
    n_genre   = len(GENRE_CLASSES)
    n_emotion = len(emotion_classes)
    logger.info(f"Multi-task: genre={n_genre} classes | emotion={n_emotion} classes "
                f"(profile={emotion_merge_profile!r})")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = MultitaskModel(MODEL_NAME, n_genre, n_emotion, dropout=args.dropout).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {total_p:,} params")
    assert total_p <= 200_000_000, f"Exceeds 200M param limit: {total_p:,}"

    train_ds = MultitaskDataset(PROJECT_ROOT / "data/processed/train.jsonl",
                                tokenizer, MAX_SEQ_LEN, emotion_merge_profile)
    val_ds   = MultitaskDataset(PROJECT_ROOT / "data/processed/val.jsonl",
                                tokenizer, MAX_SEQ_LEN, emotion_merge_profile)
    test_ds  = MultitaskDataset(PROJECT_ROOT / "data/processed/test.jsonl",
                                tokenizer, MAX_SEQ_LEN, emotion_merge_profile)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Class weights
    g_labels = [s["genre_label"] for s in train_ds.samples]
    e_labels = [s["emotion_label"] for s in train_ds.samples]
    g_weights = make_class_weights(g_labels, n_genre,   device)
    e_weights = make_class_weights(e_labels, n_emotion, device)

    g_crit = nn.CrossEntropyLoss(weight=g_weights, label_smoothing=0.1)
    if args.focal:
        e_crit = FocalLoss(gamma=args.focal_gamma, weight=e_weights)
        logger.info(f"Emotion loss: FocalLoss(γ={args.focal_gamma})")
    else:
        e_crit = nn.CrossEntropyLoss(weight=e_weights, label_smoothing=0.1)
        logger.info("Emotion loss: CrossEntropyLoss")

    # Discriminative LR optimizer (lower LR for lower layers)
    optimizer  = get_optimizer(model.encoder, base_lr=args.lr,
                               weight_decay=args.weight_decay,
                               discriminative_lr_decay=0.9)
    # Add heads at full LR
    optimizer.add_param_group({"params": list(model.genre_head.parameters()),
                               "lr": args.lr, "weight_decay": args.weight_decay})
    optimizer.add_param_group({"params": list(model.emotion_head.parameters()),
                               "lr": args.lr, "weight_decay": args.weight_decay})

    total_steps = (len(train_loader) // args.grad_accum) * args.epochs
    scheduler   = get_scheduler(optimizer, total_steps, warmup_ratio=0.1)

    # TensorBoard
    tb = TensorBoardLogger(PROJECT_ROOT / "outputs/runs", "multitask", "genre_emotion")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_dir = PROJECT_ROOT / "outputs/models/multitask"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = ckpt_dir / "multitask_best.pt"

    best_val_e_f1 = 0.0
    bad_epochs    = 0
    patience      = args.patience

    logger.info(f"Training: {args.epochs} epochs | genre_w={args.genre_weight} | "
                f"emotion_w={args.emotion_weight} | accum={args.grad_accum}")

    for epoch in range(args.epochs):
        tr_loss = train_epoch(model, train_loader, optimizer, scheduler,
                              g_crit, e_crit, args.genre_weight, args.emotion_weight,
                              device, args.grad_accum)
        val_loss, val_g_f1, val_e_f1, *_ = eval_epoch(
            model, val_loader, g_crit, e_crit, args.genre_weight, args.emotion_weight,
            device, n_genre, n_emotion)
        tb.log_epoch(epoch, tr_loss, 0.0, val_loss, val_e_f1, 0.0)
        logger.info(f"Epoch {epoch+1:3d}/{args.epochs} | loss={tr_loss:.4f} | "
                    f"val_genre_F1={val_g_f1:.4f} | val_emotion_F1={val_e_f1:.4f}")

        if val_e_f1 > best_val_e_f1:
            best_val_e_f1 = val_e_f1
            bad_epochs    = 0
            torch.save(model.state_dict(), best_ckpt)
            logger.success(f"  New best emotion val F1={val_e_f1:.4f} → saved")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                logger.warning(f"Early stopping at epoch {epoch+1}")
                break

    # Test evaluation
    logger.info(f"Loading best checkpoint: {best_ckpt}")
    model.load_state_dict(torch.load(best_ckpt, map_location=device, weights_only=True))

    (test_loss, test_g_f1, test_e_f1,
     g_preds, e_preds, g_true, e_true,
     g_probs, e_probs, t_poem_ids) = eval_epoch(
        model, test_loader, g_crit, e_crit, args.genre_weight, args.emotion_weight,
        device, n_genre, n_emotion)

    test_g_poem_f1 = poem_f1(g_probs, g_true, t_poem_ids, n_genre)
    test_e_poem_f1 = poem_f1(e_probs, e_true, t_poem_ids, n_emotion)

    logger.success(f"TEST genre: clip F1={test_g_f1:.4f} | poem F1={test_g_poem_f1:.4f}")
    logger.success(f"TEST emotion: clip F1={test_e_f1:.4f} | poem F1={test_e_poem_f1:.4f}")

    # Confusion matrices
    fig_e = make_confusion_figure(e_true, e_preds, emotion_classes,
                                  f"Multitask Emotion (clip Macro-F1={test_e_f1:.3f})")
    fig_g = make_confusion_figure(g_true, g_preds, GENRE_CLASSES,
                                  f"Multitask Genre (clip Macro-F1={test_g_f1:.3f})")
    em_conf_path = FIGURES_DIR / "multitask_emotion_confusion.png"
    gn_conf_path = FIGURES_DIR / "multitask_genre_confusion.png"
    fig_e.savefig(em_conf_path, bbox_inches="tight", dpi=150)
    plt.close(fig_e)
    fig_g.savefig(gn_conf_path, bbox_inches="tight", dpi=150)
    plt.close(fig_g)

    # Classification reports
    e_short = [c.split("(")[0].strip() for c in emotion_classes]
    g_short = [c.split("(")[0].strip() for c in GENRE_CLASSES]
    e_present = sorted(set(e_true))
    g_present = sorted(set(g_true))
    logger.info("\nEmotion classification report:\n" + classification_report(
        e_true, e_preds, labels=e_present,
        target_names=[e_short[i] for i in e_present], zero_division=0))
    logger.info("\nGenre classification report:\n" + classification_report(
        g_true, g_preds, labels=g_present,
        target_names=[g_short[i] for i in g_present], zero_division=0))

    # Secondary metrics (ECE + Top-2 for emotion)
    e_confs = np.array([max(p) for p in e_probs])
    e_corr  = np.array([int(p == t) for p, t in zip(e_preds, e_true)], dtype=float)
    bins = np.linspace(0, 1, 11)
    ece  = sum(
        mask.sum() / len(e_true) * abs(e_corr[mask].mean() - e_confs[mask].mean())
        for lo, hi in zip(bins[:-1], bins[1:])
        if (mask := (e_confs >= lo) & (e_confs < hi)).sum() > 0
    )
    top2_acc = sum(
        1 for prob, true in zip(e_probs, e_true)
        if true in sorted(range(n_emotion), key=lambda i: prob[i], reverse=True)[:2]
    ) / len(e_true)

    report = {
        "run_id":          "multitask_genre_emotion",
        "seed":            42,
        "emotion_merge_profile": emotion_merge_profile,
        "genre_weight":    args.genre_weight,
        "emotion_weight":  args.emotion_weight,
        "focal":           args.focal,
        "focal_gamma":     args.focal_gamma if args.focal else None,
        "dropout":         args.dropout,
        "weight_decay":    args.weight_decay,
        "n_emotion_classes": n_emotion,
        "n_genre_classes":   n_genre,
        "best_val_emotion_f1":      round(float(best_val_e_f1), 4),
        "test_genre_clip_f1":       round(float(test_g_f1), 4),
        "test_genre_poem_f1":       round(float(test_g_poem_f1), 4),
        "test_emotion_clip_f1":     round(float(test_e_f1), 4),
        "test_emotion_poem_f1":     round(float(test_e_poem_f1), 4),
        "test_emotion_top2_acc":    round(float(top2_acc), 4),
        "test_emotion_ece":         round(float(ece), 4),
        "checkpoint_path":          str(best_ckpt),
        "emotion_confusion_matrix": str(em_conf_path),
        "genre_confusion_matrix":   str(gn_conf_path),
    }

    out = REPORT_DIR / "multitask_eval.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.success(f"Report → {out}")
    tb.close()


if __name__ == "__main__":
    logger.add("logs/train_multitask.log", rotation="10 MB")

    parser = argparse.ArgumentParser(description="Multi-task genre+emotion training")
    parser.add_argument("--emotion-merge-profile",
                        choices=list(EMOTION_MERGE_PROFILES.keys()),
                        default="rare_merge_v1")
    parser.add_argument("--genre-weight",   type=float, default=0.7)
    parser.add_argument("--emotion-weight", type=float, default=0.3)
    parser.add_argument("--focal",          action="store_true",
                        help="Use FocalLoss for emotion head")
    parser.add_argument("--focal-gamma",    type=float, default=2.0)
    parser.add_argument("--dropout",        type=float, default=0.1)
    parser.add_argument("--weight-decay",   type=float, default=0.01)
    parser.add_argument("--lr",             type=float, default=3e-5)
    parser.add_argument("--epochs",         type=int,   default=20)
    parser.add_argument("--patience",       type=int,   default=8)
    parser.add_argument("--batch-size",     type=int,   default=16)
    parser.add_argument("--grad-accum",     type=int,   default=4)
    args = parser.parse_args()

    main(args)

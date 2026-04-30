"""
scripts/plot_confusion_matrices.py

Generate poem-level confusion matrices for genre and emotion (final adopted systems).

  Genre  : GENRE-R4 checkpoint (AraPoemBERT, window=3, corrected merge)
           Run inference on test poems (aggregated text), majority-vote per poem.

  Emotion: poem_emotion_predictions_test.json  (full_fusion / raw variants)
           Gold label = gold_poem_emotion field; pred = predicted_poem_emotion.

Output:
  outputs/figures/genre_poem_confusion.png
  outputs/figures/emotion_poem_confusion_fusion.png
  outputs/figures/emotion_poem_confusion_raw.png

Usage:
    uv run python scripts/plot_confusion_matrices.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from sklearn.metrics import confusion_matrix, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.labels import (
    GENRE_CLASSES,
    get_merged_emotion_classes,
)

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data/processed"
FIGURES_DIR = PROJECT_ROOT / "outputs/figures"
REPORT_DIR = PROJECT_ROOT / "outputs/reports"
GENRE_CKPT = PROJECT_ROOT / "outputs/models/arapoem_genre/arapoem_genre_best.pt"
EMOTION_PRED = REPORT_DIR / "poem_emotion_predictions_test.json"
TEST_JSONL = DATA_DIR / "test.jsonl"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

EMOTION_PROFILE = "rare_merge_v1"
GENRE_MERGE_MAP = {"Madih": "Fakhr", "I'tithar": "Ghazal", "Tareef": "Hija"}


# ── Plot helper ────────────────────────────────────────────────────────────────


def plot_cm(cm: np.ndarray, class_names: list[str], title: str, out_path: Path) -> None:
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(7, n), max(6, n - 1)))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(n),
        yticks=np.arange(n),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    thresh = cm.max() / 2.0
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=9,
                color="white" if cm[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Saved → {out_path}")


# ── Genre poem-level confusion matrix ─────────────────────────────────────────


def build_genre_confusion() -> None:
    if not GENRE_CKPT.exists():
        logger.error(f"Genre checkpoint not found: {GENRE_CKPT}")
        return

    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )

    raw_classes = GENRE_CLASSES
    merged_set: set[str] = set()
    for g in raw_classes:
        short = g.split("(")[0].strip()
        merged_set.add(GENRE_MERGE_MAP.get(short, short))
    class_names = sorted(merged_set)
    n_classes = len(class_names)
    cls2id = {c: i for i, c in enumerate(class_names)}

    logger.info(f"Genre classes ({n_classes}): {class_names}")

    model = AutoModelForSequenceClassification.from_pretrained(
        "faisalq/bert-base-arapoembert",
        num_labels=n_classes,
        local_files_only=True,
        ignore_mismatched_sizes=True,
    ).to(device)
    state = torch.load(GENRE_CKPT, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        "faisalq/bert-base-arapoembert", local_files_only=True
    )

    # Load test clips and group by poem
    test_clips = [
        json.loads(line_str)
        for line_str in TEST_JSONL.read_text(encoding="utf-8").splitlines()
        if line_str.strip()
    ]
    poem_clips: dict[str, list[dict]] = defaultdict(list)
    for rec in test_clips:
        key = (
            rec.get("source_poem")
            or rec.get("poem_id")
            or f"{rec.get('poet_en', '')}|{rec.get('genre_en', '')}"
        )
        poem_clips[key].append(rec)

    y_true, y_pred = [], []
    WINDOW = 3

    for poem_key, clips in poem_clips.items():
        clips_sorted = sorted(clips, key=lambda c: c.get("start", 0))

        # Gold genre (majority vote, then merge)
        genre_counts = Counter(c.get("genre_en", "") for c in clips_sorted)
        raw_gold = (
            genre_counts.most_common(1)[0][0].split("(")[0].strip()
            if genre_counts
            else ""
        )
        gold_merged = GENRE_MERGE_MAP.get(raw_gold, raw_gold)
        if gold_merged not in cls2id:
            continue

        # Sliding-window inference
        texts = [c.get("text_corrected", "") for c in clips_sorted]
        logits_list: list[torch.Tensor] = []
        for i in range(len(texts)):
            window = texts[max(0, i - WINDOW + 1) : i + 1]
            text = " ".join(w for w in window if w.strip())
            if not text.strip():
                continue
            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=32,
                padding=True,
            ).to(device)
            with torch.no_grad():
                logits = model(**enc).logits
            logits_list.append(logits[0].cpu())

        if not logits_list:
            continue
        poem_logits = torch.stack(logits_list).mean(0)
        pred_id = poem_logits.argmax().item()
        pred_label = class_names[pred_id]

        y_true.append(cls2id[gold_merged])
        y_pred.append(pred_id)
        logger.info(f"  {poem_key[:30]:30s}  gold={gold_merged:12s}  pred={pred_label}")

    if not y_true:
        logger.error("No genre predictions produced.")
        return

    present_ids = sorted(set(y_true) | set(y_pred))
    present_names = [class_names[i] for i in present_ids]
    cm = confusion_matrix(y_true, y_pred, labels=present_ids)

    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    plot_cm(
        cm,
        present_names,
        f"Genre — Poem-level Confusion Matrix (N={len(y_true)} poems)\nMacro-F1 = {f1:.3f}  [GENRE-R4, window=3]",
        FIGURES_DIR / "genre_poem_confusion.png",
    )


# ── Emotion poem-level confusion matrices ─────────────────────────────────────


def build_emotion_confusion() -> None:
    if not EMOTION_PRED.exists():
        logger.error(f"Poem emotion predictions not found: {EMOTION_PRED}")
        return

    preds = json.load(EMOTION_PRED.open(encoding="utf-8"))
    emotion_classes = get_merged_emotion_classes(EMOTION_PROFILE)
    short_names = [c.split("(")[0].strip() for c in emotion_classes]

    variants_to_plot = {
        "full_fusion": "Full Fusion (genre-prior + gated audio)",
        "raw": "Raw text-only (logit-mean aggregation)",
    }

    for variant_key, variant_label in variants_to_plot.items():
        if variant_key not in preds:
            logger.warning(f"Variant '{variant_key}' not in predictions — skipping.")
            continue

        variant_data = preds[variant_key]
        y_true, y_pred = [], []

        # full_fusion uses emotion_poem_final; other variants use predicted_poem_emotion
        pred_field = (
            "emotion_poem_final"
            if variant_key == "full_fusion"
            else "predicted_poem_emotion"
        )

        for poem_id, entry in variant_data.items():
            gold = entry.get("gold_poem_emotion", "")
            pred = entry.get(pred_field, "")
            if not gold or not pred:
                continue
            gold_short = gold.split("(")[0].strip()
            pred_short = pred.split("(")[0].strip()
            if gold_short not in short_names or pred_short not in short_names:
                continue
            y_true.append(short_names.index(gold_short))
            y_pred.append(short_names.index(pred_short))

        if not y_true:
            logger.warning(f"No valid predictions for variant '{variant_key}'.")
            continue

        present_ids = sorted(set(y_true) | set(y_pred))
        present_names = [short_names[i] for i in present_ids]
        cm = confusion_matrix(y_true, y_pred, labels=present_ids)

        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        safe_key = variant_key.replace("/", "_")
        plot_cm(
            cm,
            present_names,
            f"Emotion — Poem-level Confusion Matrix  [{variant_label}]\n"
            f"N={len(y_true)} poems  |  Macro-F1 = {f1:.3f}",
            FIGURES_DIR / f"emotion_poem_confusion_{safe_key}.png",
        )


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    logger.add("logs/plot_confusion_matrices.log", rotation="10 MB")

    logger.info("=== Poem-level confusion matrices ===")
    logger.info("1. Genre (GENRE-R4)")
    build_genre_confusion()

    logger.info("2. Emotion (full_fusion + raw)")
    build_emotion_confusion()

    logger.success("Done.")


if __name__ == "__main__":
    main()

"""
scripts/rerun_multitask_eval.py

Re-evaluate the existing multitask checkpoint on the CURRENT test split.

The multitask model was originally trained on an older split (343 clips).
This script loads the saved checkpoint and evaluates it on the current
strict poet-disjoint test split (333 clips, 13 poems) to measure performance
on the final, properly-split dataset.

Architecture:
  Shared AraPoemBERT encoder (CLS pooler output, 768-dim)
  ├── genre_head:   Linear(768, 8)   → 8-class genre
  └── emotion_head: Linear(768, N)   → N-class emotion (with rare_merge_v1 profile)

Output:
  outputs/reports/multitask_eval_current_split.json
    - clip-level and poem-level macro F1 for both genre and emotion
    - predictions, probabilities, and aggregated metrics

Usage:
    uv run python scripts/rerun_multitask_eval.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer
from sklearn.metrics import f1_score, classification_report
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.labels import (
    GENRE_CLASSES,
    encode_genre,
    encode_emotion_with_profile,
    get_merged_emotion_classes,
)
from src.training.trainer import set_seed

PROJECT_ROOT = Path(__file__).parent.parent
REPORT_DIR = PROJECT_ROOT / "outputs/reports"
MODEL_NAME = "faisalq/bert-base-arapoembert"
MAX_SEQ_LEN = 32
EMOTION_MERGE_PROFILE = "rare_merge_v1"
CHECKPOINT_PATH = PROJECT_ROOT / "outputs/models/multitask/multitask_best.pt"


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

        skipped = 0
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                text = rec.get("text_corrected", "").strip()
                g_id = encode_genre(rec.get("genre_en", ""))
                e_id = encode_emotion_with_profile(rec.get("emotion_text", ""), emotion_merge_profile)
                if not text or g_id == -1 or e_id == -1:
                    skipped += 1
                    continue
                poem_id = rec.get("source_poem", "") + "|" + rec.get("poet_en", "")
                self.samples.append(
                    {
                        "text": text,
                        "genre_label": g_id,
                        "emotion_label": e_id,
                        "poem_id": poem_id,
                    }
                )
        logger.info(
            f"Loaded {len(self.samples)} samples from {jsonl_path.name} "
            f"(skipped {skipped})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        enc = self.tokenizer(
            s["text"],
            max_length=self.max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "genre_label": torch.tensor(s["genre_label"], dtype=torch.long),
            "emotion_label": torch.tensor(s["emotion_label"], dtype=torch.long),
            "poem_id": s["poem_id"],
        }


# ── Multi-task model ──────────────────────────────────────────────────────────


class MultitaskModel(nn.Module):
    """Shared AraPoemBERT encoder + 2 classification heads."""

    def __init__(
        self, model_name: str, n_genre: int, n_emotion: int, dropout: float = 0.1
    ):
        super().__init__()
        cfg = AutoConfig.from_pretrained(model_name, local_files_only=True)
        cfg.hidden_dropout_prob = dropout
        cfg.attention_probs_dropout_prob = dropout
        self.encoder = AutoModel.from_pretrained(
            model_name, config=cfg, local_files_only=True
        )
        hidden = self.encoder.config.hidden_size  # 768
        self.genre_head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, n_genre))
        self.emotion_head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(hidden, n_emotion)
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output  # (B, 768) — CLS representation
        return self.genre_head(pooled), self.emotion_head(pooled)


# ── Helpers ───────────────────────────────────────────────────────────────────


def poem_f1(all_probs, all_true, all_poem_ids, n_classes) -> float:
    """Compute poem-level macro F1 using majority vote aggregation."""
    poem_probs: dict[str, list] = defaultdict(list)
    poem_labels: dict[str, int] = {}
    for prob, label, pid in zip(all_probs, all_true, all_poem_ids):
        poem_probs[pid].append(prob)
        poem_labels[pid] = label
    p_preds, p_true = [], []
    for pid, probs_list in poem_probs.items():
        avg = [
            sum(p[i] for p in probs_list) / len(probs_list) for i in range(n_classes)
        ]
        p_preds.append(max(range(n_classes), key=lambda i: avg[i]))
        p_true.append(poem_labels[pid])
    return f1_score(p_true, p_preds, average="macro", zero_division=0)


def eval_epoch(model, loader, device, n_g, n_e):
    """Evaluate model on test set, return predictions and metrics."""
    model.eval()
    g_preds, g_true, g_probs = [], [], []
    e_preds, e_true, e_probs = [], [], []
    poem_ids = []

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            g_labels = batch["genre_label"].to(device)
            e_labels = batch["emotion_label"].to(device)

            g_logits, e_logits = model(ids, mask)

            g_probs.extend(torch.softmax(g_logits, -1).cpu().tolist())
            e_probs.extend(torch.softmax(e_logits, -1).cpu().tolist())
            g_preds.extend(g_logits.argmax(-1).cpu().tolist())
            e_preds.extend(e_logits.argmax(-1).cpu().tolist())
            g_true.extend(g_labels.cpu().tolist())
            e_true.extend(e_labels.cpu().tolist())
            poem_ids.extend(batch["poem_id"])

    g_f1 = f1_score(g_true, g_preds, average="macro", zero_division=0)
    e_f1 = f1_score(e_true, e_preds, average="macro", zero_division=0)
    return (
        g_f1,
        e_f1,
        g_preds,
        e_preds,
        g_true,
        e_true,
        g_probs,
        e_probs,
        poem_ids,
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    set_seed(42)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    logger.info(f"Device: {device}")

    emotion_classes = get_merged_emotion_classes(EMOTION_MERGE_PROFILE)
    n_genre = len(GENRE_CLASSES)
    n_emotion = len(emotion_classes)
    logger.info(
        f"Multi-task evaluation: genre={n_genre} classes | emotion={n_emotion} classes "
        f"(profile={EMOTION_MERGE_PROFILE!r})"
    )

    if not CHECKPOINT_PATH.exists():
        logger.error(f"Checkpoint not found: {CHECKPOINT_PATH}")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = MultitaskModel(MODEL_NAME, n_genre, n_emotion, dropout=0.1).to(device)
    total_p = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {total_p:,} params")

    logger.info(f"Loading checkpoint: {CHECKPOINT_PATH}")
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    )

    test_ds = MultitaskDataset(
        PROJECT_ROOT / "data/processed/test.jsonl",
        tokenizer,
        MAX_SEQ_LEN,
        EMOTION_MERGE_PROFILE,
    )
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

    logger.info("Running inference on test set...")
    (
        test_g_f1,
        test_e_f1,
        g_preds,
        e_preds,
        g_true,
        e_true,
        g_probs,
        e_probs,
        t_poem_ids,
    ) = eval_epoch(model, test_loader, device, n_genre, n_emotion)

    test_g_poem_f1 = poem_f1(g_probs, g_true, t_poem_ids, n_genre)
    test_e_poem_f1 = poem_f1(e_probs, e_true, t_poem_ids, n_emotion)

    logger.success(
        f"TEST genre: clip F1={test_g_f1:.4f} | poem F1={test_g_poem_f1:.4f}"
    )
    logger.success(
        f"TEST emotion: clip F1={test_e_f1:.4f} | poem F1={test_e_poem_f1:.4f}"
    )

    e_short = [c.split("(")[0].strip() for c in emotion_classes]
    g_short = [c.split("(")[0].strip() for c in GENRE_CLASSES]
    e_present = sorted(set(e_true))
    g_present = sorted(set(g_true))
    logger.info(
        "\nEmotion classification report:\n"
        + classification_report(
            e_true,
            e_preds,
            labels=e_present,
            target_names=[e_short[i] for i in e_present],
            zero_division=0,
        )
    )
    logger.info(
        "\nGenre classification report:\n"
        + classification_report(
            g_true,
            g_preds,
            labels=g_present,
            target_names=[g_short[i] for i in g_present],
            zero_division=0,
        )
    )

    e_confs = np.array([max(p) for p in e_probs])
    e_corr = np.array([int(p == t) for p, t in zip(e_preds, e_true)], dtype=float)
    bins = np.linspace(0, 1, 11)
    ece = sum(
        mask.sum() / len(e_true) * abs(e_corr[mask].mean() - e_confs[mask].mean())
        for lo, hi in zip(bins[:-1], bins[1:])
        if (mask := (e_confs >= lo) & (e_confs < hi)).sum() > 0
    )
    top2_acc = sum(
        1
        for prob, true in zip(e_probs, e_true)
        if true in sorted(range(n_emotion), key=lambda i: prob[i], reverse=True)[:2]
    ) / len(e_true)

    report = {
        "run_id": "multitask_reeval_current_split",
        "seed": 42,
        "emotion_merge_profile": EMOTION_MERGE_PROFILE,
        "n_emotion_classes": n_emotion,
        "n_genre_classes": n_genre,
        "test_split_clips": len(test_ds),
        "unique_poems": len(set(t_poem_ids)),
        "test_genre_clip_f1": round(float(test_g_f1), 4),
        "test_genre_poem_f1": round(float(test_g_poem_f1), 4),
        "test_emotion_clip_f1": round(float(test_e_f1), 4),
        "test_emotion_poem_f1": round(float(test_e_poem_f1), 4),
        "test_emotion_top2_acc": round(float(top2_acc), 4),
        "test_emotion_ece": round(float(ece), 4),
        "checkpoint_path": str(CHECKPOINT_PATH),
        "notes": "Re-evaluation on current strict poet-disjoint test split (333 clips, 13 poems). "
        "Original training used an older split (343 clips).",
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "multitask_eval_current_split.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.success(f"Report → {out}")


if __name__ == "__main__":
    logger.add("logs/rerun_multitask_eval.log", rotation="10 MB")
    main()

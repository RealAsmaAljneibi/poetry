"""
scripts/train_emotion_bilstm.py

Strategy 3 (Hierarchical BiLSTM) for emotion_text classification.

Adapts the genre BiLSTM architecture for the 12-class emotion task:
  Phase 1: precompute CLS embeddings from the fine-tuned emotion checkpoint
  Phase 2: train BiLSTM (+ optional positional features) on frozen embeddings

Key difference from genre version:
  - Uses arapoem_emotion_text_best.pt (emotion-task fine-tuned) for embedding
  - Embeds into data/processed/emotion_embeddings/{split}.pkl
  - Predicts EMOTION_CLASSES (12) instead of GENRE_CLASSES (8)
  - Uses sqrt class weights for extreme imbalance (same fix as window=3 training)

Usage:
    uv run python scripts/train_emotion_bilstm.py --precompute
    uv run python scripts/train_emotion_bilstm.py
    uv run python scripts/train_emotion_bilstm.py --precompute --pos-features
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.labels import EMOTION_CLASSES, encode_emotion
from src.training.trainer import TensorBoardLogger, EarlyStopper, set_seed

# ── Paths ─────────────────────────────────────────────────────────────────────
EMOTION_CHECKPOINT = Path(
    "outputs/models/arapoem_emotion/arapoem_emotion_text_window3_best.pt"
)
ARAPOEM_MODEL_NAME = "faisalq/bert-base-arapoembert"
EMBED_DIR = Path("data/processed/emotion_embeddings")
DATA_SPLITS = {
    "train": Path("data/processed/train.jsonl"),
    "val": Path("data/processed/val.jsonl"),
    "test": Path("data/processed/test.jsonl"),
}
OUTPUT_DIR = Path("outputs/models/emotion_bilstm")
REPORT_DIR = Path("outputs/reports")
TB_DIR = Path("outputs/runs")
SEED = 42
NUM_CLASSES = len(EMOTION_CLASSES)  # 12
MAX_SEQ_LEN = 32


# ── Phase 1: Precompute CLS embeddings ───────────────────────────────────────


def precompute_emotion_embeddings(device: torch.device, batch_size: int = 64) -> None:
    """
    Load the emotion fine-tuned AraPoemBERT, run all splits through it,
    save [CLS] vectors per clip to EMBED_DIR/{split}.pkl.
    """
    if not EMOTION_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Emotion checkpoint not found: {EMOTION_CHECKPOINT}\n"
            "Run train_text_classifier.py --task emotion_text first."
        )

    logger.info(f"Loading emotion-tuned AraPoemBERT from {EMOTION_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(ARAPOEM_MODEL_NAME, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        ARAPOEM_MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    model.load_state_dict(torch.load(EMOTION_CHECKPOINT, map_location=device))
    model = model.to(device)
    model.eval()
    logger.info("Emotion model loaded. Pre-computing CLS embeddings ...")

    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    for split, jsonl_path in DATA_SPLITS.items():
        records = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                text = rec.get("text_corrected", "").strip()
                label_str = rec.get("emotion_text", "")
                label_id = encode_emotion(label_str)
                if not text or label_id == -1:
                    continue
                records.append(
                    {
                        "poem_id": rec.get("source_poem", "unknown"),
                        "poet": rec.get("poet_en", ""),
                        "text": text,
                        "start": rec.get("start", 0),
                        "label": label_id,
                    }
                )

        cls_vectors: list[np.ndarray] = []
        for start in range(0, len(records), batch_size):
            batch_recs = records[start : start + batch_size]
            enc = tokenizer(
                [r["text"] for r in batch_recs],
                max_length=MAX_SEQ_LEN,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            with torch.no_grad():
                hidden = model.bert(
                    input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device),
                ).last_hidden_state[:, 0, :]  # [B, 768]
            cls_vectors.append(hidden.cpu().numpy())

        cls_all = np.concatenate(cls_vectors, axis=0)
        for i, rec in enumerate(records):
            rec["cls"] = cls_all[i]

        # Group by poem, sort by start, add position
        poems: dict[str, list] = defaultdict(list)
        for rec in records:
            poems[rec["poem_id"]].append(rec)
        for pid, clips in poems.items():
            clips.sort(key=lambda r: r["start"])
            for pos, clip in enumerate(clips):
                clip["pos"] = pos
                clip["n_clips"] = len(clips)

        out_path = EMBED_DIR / f"{split}.pkl"
        with open(out_path, "wb") as f:
            pickle.dump(poems, f)
        logger.success(
            f"  {split}: {len(records)} clips, {len(poems)} poems → {out_path}"
        )

    logger.success("Emotion embedding pre-computation done.")


# ── Dataset ───────────────────────────────────────────────────────────────────


class EmotionSequenceDataset(Dataset):
    def __init__(self, split: str, use_pos_features: bool = False):
        pkl_path = EMBED_DIR / f"{split}.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"Embeddings not found: {pkl_path}\nRun --precompute first."
            )
        with open(pkl_path, "rb") as f:
            poems: dict = pickle.load(f)

        self.use_pos = use_pos_features
        self.poems = []

        for pid, clips in poems.items():
            cls_seq = np.stack([c["cls"] for c in clips])  # [N, 768]
            label_seq = np.array([c["label"] for c in clips])  # [N]
            n = len(clips)

            if use_pos_features:
                pos_feat = np.array(
                    [
                        [
                            c["pos"] / max(1, n - 1),
                            float(c["pos"] == 0),
                            float(c["pos"] == n - 1),
                        ]
                        for c in clips
                    ],
                    dtype=np.float32,
                )  # [N, 3]
                cls_seq = np.concatenate([cls_seq, pos_feat], axis=1)  # [N, 771]

            self.poems.append(
                {
                    "poem_id": pid,
                    "cls_seq": torch.tensor(cls_seq, dtype=torch.float32),
                    "label_seq": torch.tensor(label_seq, dtype=torch.long),
                    "n_clips": n,
                }
            )

    def __len__(self):
        return len(self.poems)

    def __getitem__(self, idx):
        return self.poems[idx]


def collate_poems(batch: list[dict]) -> dict:
    lengths = torch.tensor([p["n_clips"] for p in batch], dtype=torch.long)
    max_n = lengths.max().item()
    D = batch[0]["cls_seq"].shape[1]
    padded_cls = torch.zeros(len(batch), max_n, D)
    padded_labels = torch.full((len(batch), max_n), -1, dtype=torch.long)
    for i, poem in enumerate(batch):
        n = poem["n_clips"]
        padded_cls[i, :n, :] = poem["cls_seq"]
        padded_labels[i, :n] = poem["label_seq"]
    return {
        "cls": padded_cls,
        "labels": padded_labels,
        "lengths": lengths,
        "poem_ids": [p["poem_id"] for p in batch],
    }


# ── Model ─────────────────────────────────────────────────────────────────────


class EmotionBiLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int = 768,
        lstm_hidden: int = 256,
        num_layers: int = 1,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(self, cls_seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(
            cls_seq, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        return self.classifier(self.drop(out))  # [B, N, C]


# ── Training / eval loops ────────────────────────────────────────────────────


def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0.0
    n_correct = n_total = 0
    for batch in loader:
        cls = batch["cls"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"].to(device)
        logits = model(cls, lengths)
        B, N, C = logits.shape
        fl = logits.view(B * N, C)
        ll = labels.view(B * N)
        mask = ll != -1
        loss = criterion(fl[mask], ll[mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        preds = fl[mask].argmax(-1)
        n_correct += (preds == ll[mask]).sum().item()
        n_total += mask.sum().item()
    return total_loss / max(len(loader), 1), n_correct / max(n_total, 1)


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_true = [], []
    poem_data: dict[str, dict] = defaultdict(lambda: {"probs": [], "labels": []})
    with torch.no_grad():
        for batch in loader:
            cls = batch["cls"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"]
            poem_ids = batch["poem_ids"]
            logits = model(cls, lengths.to(device))
            B, N, C = logits.shape
            probs = F.softmax(logits, dim=-1)
            fl = logits.view(B * N, C)
            ll = labels.view(B * N)
            mask = ll != -1
            total_loss += criterion(fl[mask], ll[mask]).item()
            all_preds.extend(fl[mask].argmax(-1).cpu().tolist())
            all_true.extend(ll[mask].cpu().tolist())
            for b_idx, pid in enumerate(poem_ids):
                n = lengths[b_idx].item()
                for pos in range(n):
                    lbl = labels[b_idx, pos].item()
                    if lbl != -1:
                        poem_data[pid]["probs"].append(probs[b_idx, pos].cpu().numpy())
                        poem_data[pid]["labels"].append(lbl)
    clip_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
    # Poem-level: mean softmax → argmax
    poem_true_list, poem_pred_list = [], []
    for pid, data in poem_data.items():
        mean_probs = np.stack(data["probs"]).mean(0)
        poem_pred_list.append(int(mean_probs.argmax()))
        poem_true_list.append(int(np.bincount(data["labels"]).argmax()))
    poem_f1 = (
        f1_score(poem_true_list, poem_pred_list, average="macro", zero_division=0)
        if poem_true_list
        else 0.0
    )
    return (
        total_loss / max(len(loader), 1),
        clip_f1,
        all_preds,
        all_true,
        poem_f1,
        poem_true_list,
        poem_pred_list,
    )


# ── Main ─────────────────────────────────────────────────────────────────────


def main(args: argparse.Namespace) -> None:
    set_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if args.precompute:
        precompute_emotion_embeddings(device)

    train_ds = EmotionSequenceDataset("train", args.pos_features)
    val_ds = EmotionSequenceDataset("val", args.pos_features)
    test_ds = EmotionSequenceDataset("test", args.pos_features)

    train_loader = DataLoader(
        train_ds, batch_size=4, shuffle=True, collate_fn=collate_poems, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False, collate_fn=collate_poems, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=4, shuffle=False, collate_fn=collate_poems, num_workers=0
    )

    # Class weights (sqrt for extreme imbalance)
    all_train_labels = [
        lbl for ds_item in train_ds for lbl in ds_item["label_seq"].tolist()
    ]
    present = np.unique(all_train_labels)
    raw_w = compute_class_weight("balanced", classes=present, y=all_train_labels)
    ratio = raw_w.max() / raw_w.min()
    if ratio > 20.0:
        raw_w = np.sqrt(raw_w)
        logger.info(f"Class weight ratio {ratio:.1f}x > 20x → sqrt applied")
    w = np.ones(NUM_CLASSES, dtype=np.float32)
    for cls_id, ww in zip(present, raw_w):
        w[cls_id] = ww
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w).to(device))

    input_dim = 771 if args.pos_features else 768
    model = EmotionBiLSTM(input_dim=input_dim, num_classes=NUM_CLASSES).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"EmotionBiLSTM: {n_params:,} params, input_dim={input_dim}, classes={NUM_CLASSES}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    total_steps = args.epochs * len(train_loader)
    warmup = max(1, int(0.1 * total_steps))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-3,
        total_steps=total_steps,
        pct_start=warmup / total_steps,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_tag = "emotion_bilstm_pos" if args.pos_features else "emotion_bilstm"
    tb_logger = TensorBoardLogger(TB_DIR, model_tag=run_tag, task="emotion_text")
    stopper = EarlyStopper(
        patience=args.patience, output_dir=OUTPUT_DIR, model_tag=run_tag
    )

    best_val_f1 = 0.0
    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_loss, val_f1, _, _, val_poem_f1, _, _ = eval_epoch(
            model, val_loader, criterion, device
        )
        logger.info(
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"train_loss={tr_loss:.4f} acc={tr_acc:.3f} | "
            f"val_loss={val_loss:.4f} clip_F1={val_f1:.4f} poem_F1={val_poem_f1:.4f}"
        )
        tb_logger.log_epoch(
            epoch=epoch,
            train_loss=tr_loss,
            train_acc=tr_acc,
            val_loss=val_loss,
            val_f1=val_f1,
            val_acc=0.0,
        )
        if stopper.step(val_f1, model):
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1

    # ── Final test evaluation ──────────────────────────────────────────────────
    logger.info(f"Loading best checkpoint: {stopper.best_ckpt}")
    model.load_state_dict(torch.load(stopper.best_ckpt, map_location=device))
    test_loss, test_f1, test_preds, test_true, poem_f1, poem_true, poem_pred = (
        eval_epoch(model, test_loader, criterion, device)
    )

    logger.success(f"TEST clip Macro-F1 = {test_f1:.4f}  poem Macro-F1 = {poem_f1:.4f}")
    short_names = [c.split("(")[0].strip() for c in EMOTION_CLASSES]
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

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(test_true, test_preds, labels=present_ids)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=present_names,
        yticklabels=present_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(
        f"Test Confusion — EmotionBiLSTM {'+ pos' if args.pos_features else ''} | F1={test_f1:.3f}"
    )
    cm_path = (
        REPORT_DIR
        / f"emotion_bilstm{'_pos' if args.pos_features else ''}_confusion.png"
    )
    fig.savefig(cm_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    logger.info(f"Confusion matrix → {cm_path}")

    result = {
        "strategy": f"BiLSTM_pos={args.pos_features}",
        "test_clip_macro_f1": round(test_f1, 4),
        "test_poem_macro_f1": round(poem_f1, 4),
        "best_val_clip_macro_f1": round(best_val_f1, 4),
        "n_test_clips": len(test_true),
        "n_test_poems": len(poem_true),
        "confusion_matrix_path": str(cm_path),
        "classification_report": report_str,
    }
    json_path = (
        REPORT_DIR / f"emotion_bilstm{'_pos' if args.pos_features else ''}_eval.json"
    )
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    logger.success(f"Results → {json_path}")
    tb_logger.close()


if __name__ == "__main__":
    logger.add("logs/train_text.log", rotation="10 MB", mode="a")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--precompute",
        action="store_true",
        help="Pre-compute emotion CLS embeddings before training.",
    )
    parser.add_argument(
        "--pos-features",
        action="store_true",
        help="Prepend positional features to CLS vectors (Strategy 2).",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()
    main(args)

"""
scripts/train_hierarchical.py

Strategy 3: Hierarchical model — BiLSTM over per-clip AraPoemBERT [CLS] embeddings.
Strategy 2: Positional features (position_ratio, is_first, is_last) prepended to
            each clip's CLS vector before the LSTM.

Motivation
----------
Clips are sequential fragments cut by Whisper's timestamp segmentation, not
independent poems. Clip i is preceded by clip i-1 and followed by clip i+1.
Classifying each clip in isolation discards the narrative arc of the poem.

Architecture
------------
  Phase 1 (pre-compute, ~5 min, run once):
    AraPoemBERT (fine-tuned checkpoint) → [CLS] embedding per clip

  Phase 2 (fast LSTM training on top of frozen embeddings):
    Per poem: sequence of [CLS] vectors → Bidirectional LSTM →
              per-timestep hidden state (512-dim) → linear → logit per clip

    Training signal: cross-entropy on ALL 2669 clip-level labels.
    Evaluation: clip-level F1 AND poem-level majority vote.

    The BiLSTM hidden state at position i encodes both:
      - forward context: clips 0..i (what came before)
      - backward context: clips i+1..N-1 (what comes next)
    → Each clip is classified with full poem awareness.

Strategy 2 (--pos-features):
    Prepend [position_ratio, is_first, is_last] to each CLS vector before LSTM.
    Gives the model explicit awareness of narrative position.

Usage
-----
    # Step 1: pre-compute embeddings from the fine-tuned checkpoint
    uv run python scripts/train_hierarchical.py --precompute

    # Step 2: train BiLSTM (uses pre-computed embeddings)
    uv run python scripts/train_hierarchical.py

    # With positional features (Strategy 2):
    uv run python scripts/train_hierarchical.py --pos-features

    # Full pipeline (precompute + train):
    uv run python scripts/train_hierarchical.py --precompute --pos-features
"""

import argparse
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import classification_report, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.labels import GENRE_CLASSES, encode_genre
from src.training.trainer import TensorBoardLogger, EarlyStopper, set_seed

# ── Paths ─────────────────────────────────────────────────────────────────────
ARAPOEM_CHECKPOINT = Path("outputs/models/arapoem_genre/arapoem_genre_best.pt")
ARAPOEM_MODEL_NAME = "faisalq/bert-base-arapoembert"
EMBED_DIR = Path("data/processed/embeddings")
DATA_SPLITS = {
    "train": Path("data/processed/train.jsonl"),
    "val": Path("data/processed/val.jsonl"),
    "test": Path("data/processed/test.jsonl"),
}
OUTPUT_DIR = Path("outputs/models/hierarchical")
REPORT_DIR = Path("outputs/reports")
TB_DIR = Path("outputs/runs")
SEED = 42
NUM_CLASSES = len(GENRE_CLASSES)  # 8
MAX_SEQ_LEN = 32  # AraPoemBERT hard limit


# ── Phase 1: Pre-compute CLS embeddings ──────────────────────────────────────


def precompute_embeddings(device: torch.device, batch_size: int = 64):
    """
    Load fine-tuned AraPoemBERT, run all splits through it, save CLS vectors.
    Output: EMBED_DIR/{split}.pkl  → dict[poem_id → sorted list of dicts]
    Each dict: {pos, cls, label, text, start, poet}
    """
    if not ARAPOEM_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Fine-tuned checkpoint not found: {ARAPOEM_CHECKPOINT}\n"
            "Run train_text_classifier.py first."
        )

    logger.info(f"Loading fine-tuned AraPoemBERT from {ARAPOEM_CHECKPOINT}")
    tokenizer = AutoTokenizer.from_pretrained(ARAPOEM_MODEL_NAME, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        ARAPOEM_MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    model.load_state_dict(torch.load(ARAPOEM_CHECKPOINT, map_location=device))
    model = model.to(device)
    model.eval()
    logger.info("Model loaded. Pre-computing embeddings...")

    EMBED_DIR.mkdir(parents=True, exist_ok=True)

    for split, jsonl_path in DATA_SPLITS.items():
        records = []
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                text = rec.get("text_corrected", "").strip()
                label_str = rec.get("genre_en", "")
                label_id = encode_genre(label_str)
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

        cls_vectors = []
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

        cls_all = np.concatenate(cls_vectors, axis=0)  # [N, 768]
        for i, rec in enumerate(records):
            rec["cls"] = cls_all[i]

        # Group by poem, sort by start time, add position index
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
            f"  {split}: {len(records)} clips across {len(poems)} poems → {out_path}"
        )

    logger.success("Pre-computation complete.")


# ── Dataset ───────────────────────────────────────────────────────────────────


class PoemSequenceDataset(Dataset):
    """
    Each item is one poem: sequence of (CLS vectors [N, 768+pos_dim], labels [N]).
    Variable N across poems; handled by a custom collate function.
    """

    def __init__(self, split: str, use_pos_features: bool = False):
        pkl_path = EMBED_DIR / f"{split}.pkl"
        if not pkl_path.exists():
            raise FileNotFoundError(
                f"Embeddings not found: {pkl_path}\nRun with --precompute first."
            )
        with open(pkl_path, "rb") as f:
            poems: dict = pickle.load(f)

        self.use_pos = use_pos_features
        self.poems = []  # list of {poem_id, cls_seq, label_seq, n_clips}

        for pid, clips in poems.items():
            cls_seq = np.stack([c["cls"] for c in clips])  # [N, 768]
            label_seq = np.array([c["label"] for c in clips])  # [N]
            n = len(clips)

            if use_pos_features:
                # [position_ratio, is_first, is_last] for each clip
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
                    "cls_seq": torch.tensor(cls_seq, dtype=torch.float32),  # [N, D]
                    "label_seq": torch.tensor(label_seq, dtype=torch.long),  # [N]
                    "n_clips": n,
                }
            )

        logger.info(
            f"PoemSequenceDataset [{split}]: {len(self.poems)} poems, "
            f"pos_features={use_pos_features}"
        )

    def __len__(self):
        return len(self.poems)

    def __getitem__(self, idx):
        return self.poems[idx]


def collate_poems(batch: list[dict]) -> dict:
    """
    Pad variable-length poem sequences for batching.
    Returns padded tensors + lengths for pack_padded_sequence.
    """
    lengths = torch.tensor([p["n_clips"] for p in batch], dtype=torch.long)
    max_n = lengths.max().item()
    D = batch[0]["cls_seq"].shape[1]

    padded_cls = torch.zeros(len(batch), max_n, D)
    padded_labels = torch.full((len(batch), max_n), -1, dtype=torch.long)  # -1 = pad

    for i, poem in enumerate(batch):
        n = poem["n_clips"]
        padded_cls[i, :n, :] = poem["cls_seq"]
        padded_labels[i, :n] = poem["label_seq"]

    return {
        "cls": padded_cls,  # [B, max_n, D]
        "labels": padded_labels,  # [B, max_n]  -1 = padding
        "lengths": lengths,  # [B]
        "poem_ids": [p["poem_id"] for p in batch],
    }


# ── Model ─────────────────────────────────────────────────────────────────────


class HierarchicalPoetryClassifier(nn.Module):
    """
    Bidirectional LSTM over per-clip AraPoemBERT [CLS] embeddings.

    Each clip's output hidden state encodes both:
      - forward context: everything that came before
      - backward context: everything that comes after
    → per-clip classification with full poem awareness.

    input_dim = 768 (CLS) or 771 (CLS + positional features)
    """

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
        # BiLSTM → 2×lstm_hidden hidden state per clip
        self.classifier = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(
        self,
        cls_seq: torch.Tensor,  # [B, N, D]
        lengths: torch.Tensor,  # [B]
    ) -> torch.Tensor:  # [B, N, num_classes]
        # Pack padded sequence for variable-length efficiency
        packed = pack_padded_sequence(
            cls_seq, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(lstm_out, batch_first=True)  # [B, N, 2H]
        out = self.drop(out)
        logits = self.classifier(out)  # [B, N, C]
        return logits


# ── Training / evaluation ─────────────────────────────────────────────────────


def train_epoch(
    model, loader, optimizer, scheduler, criterion, device
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    n_correct = n_total = 0

    for batch in loader:
        cls = batch["cls"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"].to(device)

        logits = model(cls, lengths)  # [B, N, C]
        B, N, C = logits.shape

        # Flatten for loss, ignoring padding positions (label == -1)
        flat_logits = logits.view(B * N, C)
        flat_labels = labels.view(B * N)
        mask = flat_labels != -1

        loss = criterion(flat_logits[mask], flat_labels[mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        preds = flat_logits[mask].argmax(dim=-1)
        n_correct += (preds == flat_labels[mask]).sum().item()
        n_total += mask.sum().item()

    return total_loss / max(len(loader), 1), n_correct / max(n_total, 1)


def eval_epoch(model, loader, criterion, device) -> tuple:
    """
    Returns (loss, clip_f1, clip_acc, all_preds, all_true, poem_probs_dict).
    poem_probs_dict: {poem_id: [(label, probs_vector), ...]} for poem-level eval.
    """
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

            logits = model(cls, lengths.to(device))  # [B, N, C]
            B, N, C = logits.shape
            probs = F.softmax(logits, dim=-1)

            flat_logits = logits.view(B * N, C)
            flat_labels = labels.view(B * N)
            mask = flat_labels != -1

            total_loss += criterion(flat_logits[mask], flat_labels[mask]).item()
            preds_masked = flat_logits[mask].argmax(dim=-1).cpu().tolist()
            true_masked = flat_labels[mask].cpu().tolist()
            all_preds.extend(preds_masked)
            all_true.extend(true_masked)

            # Collect per-poem softmax probs for poem-level aggregation
            for b_idx, pid in enumerate(poem_ids):
                n = lengths[b_idx].item()
                for pos in range(n):
                    lbl = labels[b_idx, pos].item()
                    if lbl == -1:
                        continue
                    poem_data[pid]["probs"].append(probs[b_idx, pos].cpu().tolist())
                    poem_data[pid]["labels"].append(lbl)

    loss = total_loss / max(len(loader), 1)
    clip_f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
    clip_acc = sum(p == t for p, t in zip(all_preds, all_true)) / max(len(all_true), 1)
    return loss, clip_f1, clip_acc, all_preds, all_true, poem_data


def poem_level_eval(poem_data: dict, class_names: list[str]) -> tuple[float, float]:
    """Average clip softmax probs per poem, argmax → poem-level F1."""
    poem_preds, poem_true = [], []
    for pid, data in poem_data.items():
        avg = np.mean(data["probs"], axis=0)
        poem_preds.append(int(avg.argmax()))
        # All clips of same poem share the same genre label
        poem_true.append(data["labels"][0])

    f1 = f1_score(poem_true, poem_preds, average="macro", zero_division=0)
    acc = sum(p == t for p, t in zip(poem_preds, poem_true)) / max(len(poem_true), 1)

    short = [c.split("(")[0].strip() for c in class_names]
    present = sorted(set(poem_true))
    report = classification_report(
        poem_true,
        poem_preds,
        labels=present,
        target_names=[short[i] for i in present],
        zero_division=0,
    )
    logger.info(f"\nPoem-level report ({len(poem_data)} poems):\n{report}")
    return f1, acc


# ── Main ──────────────────────────────────────────────────────────────────────


def main(args):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if args.precompute:
        precompute_embeddings(device)

    # ── Load datasets ────────────────────────────────────────────────────────
    train_ds = PoemSequenceDataset("train", args.pos_features)
    val_ds = PoemSequenceDataset("val", args.pos_features)
    test_ds = PoemSequenceDataset("test", args.pos_features)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_poems,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_poems,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_poems,
        num_workers=0,
    )

    input_dim = 771 if args.pos_features else 768
    logger.info(
        f"Model: BiLSTM(input={input_dim}, hidden={args.lstm_hidden}×2)  "
        f"pos_features={args.pos_features}"
    )

    # ── Class weights ─────────────────────────────────────────────────────────
    train_labels = [lbl for p in train_ds for lbl in p["label_seq"].tolist()]
    present_cls = np.unique(train_labels)
    weights = compute_class_weight("balanced", classes=present_cls, y=train_labels)
    class_w = np.ones(NUM_CLASSES, dtype=np.float32)
    for cls, w in zip(present_cls, weights):
        class_w[cls] = w
    class_weights = torch.tensor(class_w).to(device)
    logger.info(f"Class weights: min={class_w.min():.2f}  max={class_w.max():.2f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # ── Model + optimizer ─────────────────────────────────────────────────────
    model = HierarchicalPoetryClassifier(
        input_dim=input_dim,
        lstm_hidden=args.lstm_hidden,
        num_layers=args.lstm_layers,
        num_classes=NUM_CLASSES,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"LSTM params: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = max(1, int(0.1 * total_steps))
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=lambda s: (
            min((s + 1) / warmup_steps, 1.0)
            if s < warmup_steps
            else max(
                0.0,
                0.5
                * (
                    1
                    + math.cos(
                        math.pi
                        * (s - warmup_steps)
                        / max(total_steps - warmup_steps, 1)
                    )
                ),
            )
        ),
    )
    logger.info(
        f"AdamW lr={args.lr:.1e}  epochs={args.epochs}  patience={args.patience}"
    )

    # ── TensorBoard + early stopping ──────────────────────────────────────────
    tag = "hier_pos" if args.pos_features else "hier"
    tb = TensorBoardLogger(TB_DIR, tag, "genre")
    early = EarlyStopper(args.patience, OUTPUT_DIR, f"{tag}_genre")

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(
        f"Hierarchical BiLSTM | {args.epochs} epochs | "
        f"{len(train_ds)} train poems | {len(train_labels)} train clips"
    )
    logger.info("=" * 60)

    best_val_f1 = 0.0
    for epoch in range(args.epochs):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_loss, val_f1, val_acc, _, _, _ = eval_epoch(
            model, val_loader, criterion, device
        )

        tb.log_epoch(epoch, tr_loss, tr_acc, val_loss, val_f1, val_acc)
        logger.info(
            f"Epoch {epoch + 1:3d}/{args.epochs} | "
            f"train_loss={tr_loss:.4f} | train_acc={tr_acc:.3f} | "
            f"val_loss={val_loss:.4f} | val_F1={val_f1:.4f} | val_acc={val_acc:.3f}"
        )

        if early.step(val_f1, model):
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break
        best_val_f1 = max(best_val_f1, val_f1)

    # ── Test evaluation ───────────────────────────────────────────────────────
    logger.info(f"Loading best checkpoint: {early.best_ckpt}")
    model.load_state_dict(torch.load(early.best_ckpt, map_location=device))
    test_loss, test_clip_f1, test_clip_acc, test_preds, test_true, test_poem_data = (
        eval_epoch(model, test_loader, criterion, device)
    )

    logger.success(
        f"TEST (clip-level)  | Macro-F1={test_clip_f1:.4f} | acc={test_clip_acc:.4f} "
        f"| n={len(test_true)} clips"
    )

    test_poem_f1, test_poem_acc = poem_level_eval(test_poem_data, GENRE_CLASSES)
    logger.success(
        f"TEST (poem-level)  | Macro-F1={test_poem_f1:.4f} | acc={test_poem_acc:.4f} "
        f"| n={len(test_poem_data)} poems"
    )

    short = [c.split("(")[0].strip() for c in GENRE_CLASSES]
    present = sorted(set(test_true))
    report = classification_report(
        test_true,
        test_preds,
        labels=present,
        target_names=[short[i] for i in present],
        zero_division=0,
    )
    logger.info(f"\nClip-level report:\n{report}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"{tag}_genre_report.txt").write_text(report)

    logger.success(
        f"Done. Best val F1={best_val_f1:.4f}  "
        f"test clip F1={test_clip_f1:.4f}  test poem F1={test_poem_f1:.4f}"
    )
    tb.close()


if __name__ == "__main__":
    logger.add("logs/train_hierarchical.log", rotation="10 MB")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--precompute",
        action="store_true",
        help="Pre-compute AraPoemBERT CLS embeddings for all splits (run once).",
    )
    parser.add_argument(
        "--pos-features",
        action="store_true",
        help="Strategy 2: prepend [position_ratio, is_first, is_last] to each CLS vector.",
    )
    parser.add_argument(
        "--lstm-hidden", type=int, default=256, help="BiLSTM hidden size per direction"
    )
    parser.add_argument(
        "--lstm-layers", type=int, default=1, help="Number of LSTM layers"
    )
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Poems per batch")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs")
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()
    main(args)

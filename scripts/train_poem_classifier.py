"""
scripts/train_poem_classifier.py

Poem-level genre classification with AraBERT (aubmindlab/bert-base-arabertv2).

Ablation vs AraPoemBERT clip-level:
  AraPoemBERT — verse-level (32-token context, ~8 tokens/clip, 3340 examples)
  AraBERT     — poem-level  (512-token context, median 164 tokens, ~107 examples)

Hypothesis: genre is a narrative property of the full poem, not a verse-level
property.  A single verse is ambiguous; the full poem's vocabulary, imagery,
and progression reveal the genre.

Split: poet-disjoint at poem level (all poems by a poet go to one split).
For genres with <3 poems (Ritha, Badawa), stratified random split.

Usage:
    # Full fine-tune (recommended — discriminative LR + gradual unfreezing):
    uv run python scripts/train_poem_classifier.py
    uv run python scripts/train_poem_classifier.py --epochs 30 --lr 2e-5

    # Frozen-head only (feature extraction):
    uv run python scripts/train_poem_classifier.py --freeze-encoder --lr 5e-4 --epochs 50
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from loguru import logger
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.labels import GENRE_CLASSES, merge_genre_label, encode_genre
from src.training.trainer import (
    TensorBoardLogger,
    EarlyStopper,
    set_seed,
    get_optimizer,
    get_scheduler,
    unfreeze_next_layer_group,
)

# ── Constants ─────────────────────────────────────────────────────────────────
MODEL_NAME = "aubmindlab/bert-base-arabertv2"
MAX_LENGTH = 512
DATA_FILE = Path("data/processed/master_dataset.jsonl")
OUTPUT_DIR = Path("outputs/models/arabert_poem")
REPORT_DIR = Path("outputs/reports")
TB_DIR = Path("outputs/runs")
SEED = 42
NUM_CLASSES = len(GENRE_CLASSES)  # 8


# ── Build poem-level records ──────────────────────────────────────────────────


def build_poems(jsonl_path: Path) -> list[dict]:
    """
    Group clips by source_poem, concatenate text_corrected ordered by start time.
    Returns list of poem dicts: {poem_id, poet, genre, text, n_clips}.
    """
    raw: dict[str, dict] = {}
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            pid = rec.get("source_poem", "unknown")
            if pid not in raw:
                raw[pid] = {
                    "poem_id": pid,
                    "poet": rec.get("poet_en", ""),
                    "genre": merge_genre_label(rec.get("genre_en", "").strip()),
                    "clips": [],
                }
            raw[pid]["clips"].append(
                {
                    "text": rec.get("text_corrected", "").strip(),
                    "start": rec.get("start", 0),
                }
            )

    poems = []
    for pid, info in raw.items():
        info["clips"].sort(key=lambda c: c["start"])
        text = " ".join(c["text"] for c in info["clips"] if c["text"])
        label = encode_genre(info["genre"])
        if not text or label == -1:
            continue
        poems.append(
            {
                "poem_id": pid,
                "poet": info["poet"],
                "genre": info["genre"],
                "label": label,
                "text": text,
                "n_clips": len(info["clips"]),
            }
        )

    logger.info(f"Built {len(poems)} poem-level records from {jsonl_path.name}")
    token_lengths = [len(text.split()) for p in poems for text in [p["text"]]]
    logger.info(
        f"  Token counts (word-split): median={int(np.median(token_lengths))}  "
        f"p90={int(np.percentile(token_lengths, 90))}  max={max(token_lengths)}"
    )
    return poems


# ── Poet-disjoint poem-level split ───────────────────────────────────────────


def split_poems(poems: list[dict], seed: int = SEED) -> dict[str, list[dict]]:
    """
    Poet-disjoint split at poem level.
    Genres with <3 poems → stratified random (poet-disjoint impossible).
    Genres with ≥3 poems → assign poets proportionally 80/10/10.
    Returns {"train": [...], "val": [...], "test": [...]}.
    """
    random.seed(seed)
    genre_poems: dict[str, list] = defaultdict(list)
    for p in poems:
        genre_poems[p["genre"]].append(p)

    assignment: dict[str, str] = {}  # poem_id → split

    for genre, gpoems in genre_poems.items():
        if len(gpoems) < 3:
            shuffled = gpoems[:]
            random.shuffle(shuffled)
            for i, p in enumerate(shuffled):
                assignment[p["poem_id"]] = (
                    "test" if i == 0 else "val" if i == 1 else "train"
                )
            logger.info(f"  {genre}: {len(gpoems)} poems → stratified random")
            continue

        poet_gpoems: dict[str, list] = defaultdict(list)
        for p in gpoems:
            poet_gpoems[p["poet"]].append(p)

        if len(poet_gpoems) < 3:
            shuffled = gpoems[:]
            shuffled.sort(key=lambda p: -p["n_clips"])
            n = len(shuffled)
            n_train = max(1, int(0.80 * n))
            n_val = max(1, int(0.10 * n))
            for i, p in enumerate(shuffled):
                assignment[p["poem_id"]] = (
                    "train" if i < n_train else "val" if i < n_train + n_val else "test"
                )
            continue

        # Poet-disjoint: assign poets proportionally by clip count
        poets_sorted = sorted(
            poet_gpoems.items(),
            key=lambda x: (-sum(p["n_clips"] for p in x[1]), x[0]),
        )
        total = sum(p["n_clips"] for p in gpoems)
        target_train = total * 0.80
        target_val = total * 0.10
        cum = 0
        for poet, ppoems in poets_sorted:
            c = sum(p["n_clips"] for p in ppoems)
            bucket = (
                "train"
                if cum < target_train
                else "val"
                if cum < target_train + target_val
                else "test"
            )
            for p in ppoems:
                assignment[p["poem_id"]] = bucket
            cum += c

    splits: dict[str, list] = {"train": [], "val": [], "test": []}
    for poem in poems:
        splits[assignment.get(poem["poem_id"], "train")].append(poem)

    for s, ps in splits.items():
        genre_dist = defaultdict(int)
        for p in ps:
            genre_dist[p["genre"]] += 1
        logger.info(
            f"{s.upper():5s}: {len(ps):3d} poems  "
            + "  ".join(
                f"{g.split('(')[0].strip()[:6]}={n}"
                for g, n in sorted(genre_dist.items(), key=lambda x: -x[1])
            )
        )
    return splits


# ── Dataset ───────────────────────────────────────────────────────────────────


class PoemDataset(Dataset):
    def __init__(self, poems: list[dict], tokenizer, max_length: int = MAX_LENGTH):
        self.poems = poems
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.poems)

    def __getitem__(self, idx: int) -> dict:
        poem = self.poems[idx]
        enc = self.tokenizer(
            poem["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": torch.tensor(poem["label"], dtype=torch.long),
            "poem_id": poem["poem_id"],
        }


# ── Training / evaluation ─────────────────────────────────────────────────────


def train_epoch(
    model,
    loader,
    optimizer,
    scheduler,
    criterion,
    device,
    accum_steps: int = 1,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    n_correct = n_total = 0
    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss = criterion(logits, labels) / accum_steps
        loss.backward()

        total_loss += loss.item() * accum_steps
        n_correct += (logits.detach().argmax(dim=-1) == labels).sum().item()
        n_total += labels.size(0)

        is_last = (i + 1) == len(loader)
        if (i + 1) % accum_steps == 0 or is_last:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return total_loss / max(len(loader), 1), n_correct / max(n_total, 1)


def eval_epoch(
    model,
    loader,
    criterion,
    device,
) -> tuple[float, float, float, list, list]:
    model.eval()
    total_loss = 0.0
    all_preds, all_true = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
            total_loss += criterion(logits, labels).item()
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_true.extend(labels.cpu().tolist())

    loss = total_loss / max(len(loader), 1)
    f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
    acc = sum(p == t for p, t in zip(all_preds, all_true)) / max(len(all_true), 1)
    return loss, f1, acc, all_preds, all_true


# ── Main ──────────────────────────────────────────────────────────────────────


def main(args):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # ── Build poems + split ───────────────────────────────────────────────────
    poems = build_poems(DATA_FILE)
    splits = split_poems(poems)
    train_poems, val_poems, test_poems = splits["train"], splits["val"], splits["test"]
    logger.info(
        f"Split: train={len(train_poems)}  val={len(val_poems)}  test={len(test_poems)} poems"
    )

    # ── Tokenizer + model ─────────────────────────────────────────────────────
    logger.info(f"Loading {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {total_params:,} params")
    assert total_params <= 200_000_000, f"Model exceeds 200M: {total_params:,}"

    # ── Encoder setup ─────────────────────────────────────────────────────────
    if args.freeze_encoder:
        # Feature extraction: freeze everything except head
        for name, param in model.named_parameters():
            if "classifier" not in name and "pooler" not in name:
                param.requires_grad = False
        hidden = model.config.hidden_size
        model.classifier = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, NUM_CLASSES),
        ).to(device)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Frozen encoder — trainable: {trainable:,} params")
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=0.01,
        )
    else:
        # Full fine-tune: discriminative LR + gradual unfreezing
        # All layers start unfrozen; unfreeze_next_layer_group toggles layers
        # top-down so the model adapts from task head → embeddings progressively.
        logger.info(
            f"Full fine-tune: discriminative LR (base={args.lr:.1e}, decay={args.disc_decay})  "
            f"gradual unfreeze every {args.unfreeze_every} epoch(s)"
        )
        optimizer = get_optimizer(
            model,
            base_lr=args.lr,
            weight_decay=0.01,
            discriminative_lr_decay=args.disc_decay,
        )

    # ── Datasets + loaders ────────────────────────────────────────────────────
    train_ds = PoemDataset(train_poems, tokenizer)
    val_ds = PoemDataset(val_poems, tokenizer)
    test_ds = PoemDataset(test_poems, tokenizer)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    # ── Class weights + loss ──────────────────────────────────────────────────
    train_labels = [p["label"] for p in train_poems]
    present_cls = np.unique(train_labels)
    weights = compute_class_weight("balanced", classes=present_cls, y=train_labels)
    class_w = np.ones(NUM_CLASSES, dtype=np.float32)
    for cls, w in zip(present_cls, weights):
        class_w[cls] = w
    class_weights = torch.tensor(class_w).to(device)
    logger.info(f"Class weights: min={class_w.min():.2f}  max={class_w.max():.2f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)

    # ── Scheduler ─────────────────────────────────────────────────────────────
    effective_steps_per_epoch = max(1, len(train_loader) // args.accum_steps)
    total_steps = effective_steps_per_epoch * args.epochs
    scheduler = get_scheduler(optimizer, total_steps, warmup_ratio=0.1)
    logger.info(
        f"Scheduler: warmup+cosine | steps/epoch={effective_steps_per_epoch} "
        f"total={total_steps} | batch={args.batch_size} accum={args.accum_steps} "
        f"(eff. batch={args.batch_size * args.accum_steps})"
    )

    # ── TensorBoard + early stopping ──────────────────────────────────────────
    mode_tag = "frozen" if args.freeze_encoder else "full"
    tb = TensorBoardLogger(TB_DIR, f"arabert_poem_{mode_tag}", "genre")
    early = EarlyStopper(args.patience, OUTPUT_DIR, f"arabert_poem_{mode_tag}_genre")

    # ── Training loop ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(
        f"Training AraBERT poem-level ({mode_tag}) | "
        f"{args.epochs} epochs | {len(train_poems)} train poems"
    )
    logger.info("=" * 60)

    best_val_f1 = 0.0
    for epoch in range(args.epochs):
        # Gradual unfreezing: unfreeze top→bottom, 1 more layer every N epochs
        if not args.freeze_encoder and args.unfreeze_every > 0:
            unfreeze_next_layer_group(model, epoch, args.unfreeze_every)

        tr_loss, tr_acc = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            criterion,
            device,
            args.accum_steps,
        )
        val_loss, val_f1, val_acc, _, _ = eval_epoch(
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
    test_loss, test_f1, test_acc, test_preds, test_true = eval_epoch(
        model, test_loader, criterion, device
    )

    logger.success(
        f"TEST (AraBERT poem-level [{mode_tag}]) | "
        f"Macro-F1={test_f1:.4f} | accuracy={test_acc:.4f} | n={len(test_poems)} poems"
    )

    short = [c.split("(")[0].strip() for c in GENRE_CLASSES]
    present_ids = sorted(set(test_true))
    report = classification_report(
        test_true,
        test_preds,
        labels=present_ids,
        target_names=[short[i] for i in present_ids],
        zero_division=0,
    )
    logger.info(f"\n{report}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / f"arabert_poem_{mode_tag}_genre_report.txt").write_text(report)

    cm = confusion_matrix(test_true, test_preds, labels=list(range(NUM_CLASSES)))
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=short,
        yticklabels=short,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"AraBERT Poem [{mode_tag}] — Macro-F1={test_f1:.3f}")
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.tight_layout()
    fig_path = REPORT_DIR / f"arabert_poem_{mode_tag}_genre_confusion.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Confusion matrix → {fig_path}")

    logger.success(f"Done. Best val F1={best_val_f1:.4f}  test F1={test_f1:.4f}")
    tb.close()


if __name__ == "__main__":
    logger.add("logs/train_poem.log", rotation="10 MB")
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze BERT encoder; train only classification head (feature extraction mode).",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Max training epochs")
    parser.add_argument(
        "--patience", type=int, default=10, help="Early stopping patience"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5, help="Base (top-layer) learning rate"
    )
    parser.add_argument(
        "--disc-decay",
        type=float,
        default=0.9,
        help="Discriminative LR decay per layer (0.9 → lower layers get 0.9^depth × base_lr)",
    )
    parser.add_argument(
        "--unfreeze-every",
        type=int,
        default=2,
        help="Gradual unfreezing: unfreeze 1 more encoder layer every N epochs (full fine-tune only)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size (small due to 512-token poems)",
    )
    parser.add_argument(
        "--accum-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (eff. batch = batch × accum)",
    )
    args = parser.parse_args()
    main(args)

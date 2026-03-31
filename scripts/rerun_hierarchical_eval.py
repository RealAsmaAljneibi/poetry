"""
scripts/rerun_hierarchical_eval.py

Re-evaluate the existing hierarchical BiLSTM checkpoint on the CURRENT test split (333 clips).

The original checkpoint (hier_pos_genre_best.pt) was trained on an older split (343 clips).
This script re-computes embeddings on the current test.jsonl, loads the checkpoint,
and evaluates clip-level and poem-level metrics.

Pipeline:
  Step 1: Re-compute test embeddings from current test.jsonl using AraPoemBERT genre checkpoint
  Step 2: Load the hierarchical BiLSTM checkpoint (hier_pos_genre_best.pt)
  Step 3: Run inference on all test poems
  Step 4: Compute clip-level and poem-level macro F1
  Step 5: Generate classification report
  Step 6: Save results to outputs/reports/hierarchical_eval_current_split.json
"""

import json
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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.labels import GENRE_CLASSES, encode_genre

# ── Paths ─────────────────────────────────────────────────────────────────────
ARAPOEM_CHECKPOINT = Path("outputs/models/arapoem_genre/arapoem_genre_best.pt")
ARAPOEM_MODEL_NAME = "faisalq/bert-base-arapoembert"
HIERARCHICAL_CHECKPOINT = Path("outputs/models/hierarchical/hier_pos_genre_best.pt")
TEST_SPLIT = Path("data/processed/test.jsonl")
REPORT_DIR = Path("outputs/reports")
SEED = 42
NUM_CLASSES = len(GENRE_CLASSES)
MAX_SEQ_LEN = 32  # AraPoemBERT hard limit


# ── Model ─────────────────────────────────────────────────────────────────────

class HierarchicalPoetryClassifier(nn.Module):
    """
    Bidirectional LSTM over per-clip AraPoemBERT [CLS] embeddings.

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
        self.classifier = nn.Linear(lstm_hidden * 2, num_classes)

    def forward(
        self,
        cls_seq: torch.Tensor,  # [B, N, D]
        lengths: torch.Tensor,  # [B]
    ) -> torch.Tensor:  # [B, N, num_classes]
        packed = pack_padded_sequence(
            cls_seq, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        lstm_out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(lstm_out, batch_first=True)  # [B, N, 2H]
        out = self.drop(out)
        logits = self.classifier(out)  # [B, N, C]
        return logits


# ── Dataset ───────────────────────────────────────────────────────────────────

class PoemSequenceDataset(Dataset):
    """
    Each item is one poem: sequence of (CLS vectors [N, 768+3] for pos features, labels [N]).
    Variable N across poems; handled by a custom collate function.
    """

    def __init__(self, poems: dict):
        self.poems = []  # list of {poem_id, cls_seq, label_seq, n_clips}

        for pid, clips in poems.items():
            cls_seq = np.stack([c["cls"] for c in clips])  # [N, 768]
            label_seq = np.array([c["label"] for c in clips])  # [N]
            n = len(clips)

            # Add positional features: [position_ratio, is_first, is_last]
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
                    "cls_seq": torch.tensor(cls_seq, dtype=torch.float32),  # [N, 771]
                    "label_seq": torch.tensor(label_seq, dtype=torch.long),  # [N]
                    "n_clips": n,
                }
            )

        logger.info(f"PoemSequenceDataset: {len(self.poems)} poems")

    def __len__(self):
        return len(self.poems)

    def __getitem__(self, idx):
        return self.poems[idx]


def collate_poems(batch: list[dict]) -> dict:
    """
    Pad variable-length poem sequences for batching.
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
        "cls": padded_cls,  # [B, max_n, 771]
        "labels": padded_labels,  # [B, max_n]  -1 = padding
        "lengths": lengths,  # [B]
        "poem_ids": [p["poem_id"] for p in batch],
    }


# ── Step 1: Compute test embeddings ────────────────────────────────────────────

def compute_test_embeddings(device: torch.device, batch_size: int = 64) -> dict:
    """
    Load fine-tuned AraPoemBERT, run test.jsonl through it, return dict of poems.

    Output: dict[poem_id → sorted list of dicts]
    Each dict: {pos, cls, label, text, start, poet, n_clips}
    """
    logger.info(f"Loading AraPoemBERT from {ARAPOEM_CHECKPOINT}")
    if not ARAPOEM_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"AraPoemBERT checkpoint not found: {ARAPOEM_CHECKPOINT}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        ARAPOEM_MODEL_NAME, local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        ARAPOEM_MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    model.load_state_dict(torch.load(ARAPOEM_CHECKPOINT, map_location=device))
    model = model.to(device)
    model.eval()
    logger.info("Model loaded. Computing test embeddings...")

    # Load test records
    records = []
    with open(TEST_SPLIT, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            text = rec.get("text_corrected", "").strip()
            label_str = rec.get("genre_en", "")
            label_id = encode_genre(label_str)
            if not text or label_id == -1:
                logger.warning(f"Skipping record: text={text!r}, genre={label_str!r}")
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

    logger.info(f"Loaded {len(records)} valid records from {TEST_SPLIT}")

    # Batch encode
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

    logger.success(
        f"Computed embeddings: {len(records)} clips across {len(poems)} poems"
    )
    return poems


# ── Step 3 & 4: Inference and evaluation ──────────────────────────────────────

def inference(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple:
    """
    Run inference on test loader.
    Returns: (all_preds, all_true, poem_data)
    poem_data: {poem_id: {"probs": [...], "labels": [...]}}
    """
    model.eval()
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

    return all_preds, all_true, poem_data


def poem_level_eval(poem_data: dict, class_names: list[str]) -> tuple[float, float]:
    """Average clip softmax probs per poem, argmax → poem-level F1."""
    poem_preds, poem_true = [], []
    for pid, data in poem_data.items():
        avg = np.mean(data["probs"], axis=0)
        poem_preds.append(int(avg.argmax()))
        poem_true.append(data["labels"][0])

    f1 = f1_score(poem_true, poem_preds, average="macro", zero_division=0)
    acc = (
        sum(p == t for p, t in zip(poem_preds, poem_true))
        / max(len(poem_true), 1)
    )

    short = [c.split("(")[0].strip() for c in class_names]
    present = sorted(set(poem_true))
    report = classification_report(
        poem_true,
        poem_preds,
        labels=present,
        target_names=[short[i] for i in present],
        zero_division=0,
    )
    logger.info(f"\nPoem-level evaluation ({len(poem_data)} poems):\n{report}")
    return f1, acc, report


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Step 1: Re-compute test embeddings
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: RE-COMPUTING TEST EMBEDDINGS FROM CURRENT TEST.JSONL")
    logger.info("=" * 70)
    poems = compute_test_embeddings(device)

    # Create dataset and loader
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: LOADING HIERARCHICAL BILSTM CHECKPOINT")
    logger.info("=" * 70)
    test_ds = PoemSequenceDataset(poems)
    test_loader = DataLoader(
        test_ds, batch_size=16, shuffle=False, collate_fn=collate_poems, num_workers=0
    )

    # Step 2: Load checkpoint
    if not HIERARCHICAL_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Hierarchical checkpoint not found: {HIERARCHICAL_CHECKPOINT}"
        )

    model = HierarchicalPoetryClassifier(
        input_dim=771,  # 768 + 3 positional features
        lstm_hidden=256,
        num_layers=1,
        num_classes=NUM_CLASSES,
        dropout=0.4,
    ).to(device)

    logger.info(f"Loading checkpoint: {HIERARCHICAL_CHECKPOINT}")
    model.load_state_dict(torch.load(HIERARCHICAL_CHECKPOINT, map_location=device))
    logger.success("Checkpoint loaded")

    # Step 3 & 4: Inference
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3 & 4: RUNNING INFERENCE ON TEST POEMS")
    logger.info("=" * 70)
    test_preds, test_true, test_poem_data = inference(model, test_loader, device)

    # Compute clip-level metrics
    clip_f1 = f1_score(test_true, test_preds, average="macro", zero_division=0)
    clip_acc = sum(p == t for p, t in zip(test_preds, test_true)) / max(
        len(test_true), 1
    )

    logger.success(
        f"TEST (clip-level) | Macro-F1={clip_f1:.4f} | acc={clip_acc:.4f} "
        f"| n={len(test_true)} clips"
    )

    # Compute poem-level metrics
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: POEM-LEVEL EVALUATION")
    logger.info("=" * 70)
    poem_f1, poem_acc, poem_report = poem_level_eval(test_poem_data, GENRE_CLASSES)

    logger.success(
        f"TEST (poem-level) | Macro-F1={poem_f1:.4f} | acc={poem_acc:.4f} "
        f"| n={len(test_poem_data)} poems"
    )

    # Classification report (clip-level)
    short = [c.split("(")[0].strip() for c in GENRE_CLASSES]
    present = sorted(set(test_true))
    clip_report = classification_report(
        test_true,
        test_preds,
        labels=present,
        target_names=[short[i] for i in present],
        zero_division=0,
    )
    logger.info(f"\nClip-level evaluation:\n{clip_report}")

    # Step 6: Save results
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: SAVING RESULTS")
    logger.info("=" * 70)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    results = {
        "checkpoint": str(HIERARCHICAL_CHECKPOINT),
        "test_split": str(TEST_SPLIT),
        "num_poems": len(test_poem_data),
        "num_clips": len(test_true),
        "clip_level": {
            "macro_f1": float(clip_f1),
            "accuracy": float(clip_acc),
            "num_clips": len(test_true),
            "report": clip_report,
        },
        "poem_level": {
            "macro_f1": float(poem_f1),
            "accuracy": float(poem_acc),
            "num_poems": len(test_poem_data),
            "report": poem_report,
        },
    }

    output_file = REPORT_DIR / "hierarchical_eval_current_split.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.success(f"Results saved to {output_file}")

    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Checkpoint:         {HIERARCHICAL_CHECKPOINT}")
    logger.info(f"Test split:         {TEST_SPLIT} ({len(test_true)} clips)")
    logger.info(f"Poems:              {len(test_poem_data)}")
    logger.info(f"Clip-level F1:      {clip_f1:.4f}")
    logger.info(f"Poem-level F1:      {poem_f1:.4f}")
    logger.info(f"Report saved to:    {output_file}")
    logger.info("=" * 70)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.add("logs/rerun_hierarchical_eval.log", rotation="10 MB")
    main()

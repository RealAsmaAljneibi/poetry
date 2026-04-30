"""
scripts/genre_conditioned_emotion.py

Genre-conditioned emotion inference for Nabat-AI.

MOTIVATION
──────────
The raw emotion model (K1_merge_v1, window=1, 9 classes) predicts
p(emotion | text) without knowledge of genre.  But genre is known at
inference time (predicted by the genre classifier before this step).
Genre provides a strong prior: a Wataniyya poem is unlikely to express
Sorrow; a Ritha poem is unlikely to express Humor.

TWO DECODING MODES
──────────────────
1. Genre-constrained decoding (recommended)
   - Run emotion model as normal → get softmax probabilities
   - If top-1 predicted emotion ∈ GENRE_EXPECTED_EMOTIONS[genre] → keep it
   - Otherwise → choose the highest-probability emotion inside the
     expected set (falls back to raw top-1 if no expected emotion found)
   - Always stores both: emotion_raw and emotion_genre_conditioned

2. Genre-prior reweighting (optional, controlled by --lambda-prior > 0)
   - p'(e) ∝ p(e|text) * prior(e|genre)^λ
   - prior(e|genre) estimated from the train split
   - λ=0 reduces to raw; λ=1 is full reweighting

EVALUATION
──────────
Compares raw vs genre-conditioned on the test split:
  - clip Macro-F1
  - poem Macro-F1
  - partial-credit score
  - genre-plausibility rate (fraction of predictions ∈ expected set)

Usage:
    uv run python scripts/genre_conditioned_emotion.py
    uv run python scripts/genre_conditioned_emotion.py --lambda-prior 0.5
    uv run python scripts/genre_conditioned_emotion.py --mode constrained
    uv run python scripts/genre_conditioned_emotion.py --mode reweight --lambda-prior 0.3

Output:
    outputs/reports/genre_conditioned_emotion.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.labels import (
    GENRE2ID,
    apply_emotion_merge,
    get_genre_expected_emotions,
    get_merged_emotion_classes,
)
from src.evaluation.metrics import emotion_partial_credit
from scripts.train_text_classifier import NabatiTextDataset

logger.add("logs/genre_conditioned_emotion.log", rotation="10 MB")

PROJECT_ROOT = Path(__file__).parent.parent
EMOTION_CKPT = (
    PROJECT_ROOT / "outputs/models/arapoem_emotion/arapoem_emotion_text_best.pt"
)
GENRE_CKPT = PROJECT_ROOT / "outputs/models/arapoem_genre/arapoem_genre_best.pt"
MODEL_NAME = "faisalq/bert-base-arapoembert"
MERGE_PROFILE = "rare_merge_v1"
CONTEXT_WIN_EMO = 1  # best emotion window
CONTEXT_WIN_GEN = 3  # best genre window
MAX_SEQ_LEN = 32
BATCH_SIZE = 32


# ── Argument parsing ────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Genre-conditioned emotion evaluation")
    p.add_argument(
        "--mode",
        choices=["constrained", "reweight", "both"],
        default="both",
        help="Decoding mode (default: both)",
    )
    p.add_argument(
        "--lambda-prior",
        type=float,
        default=0.5,
        help="λ for genre-prior reweighting (default: 0.5). 0=raw, 1=full.",
    )
    p.add_argument(
        "--merge-profile", default=MERGE_PROFILE, choices=["none", "rare_merge_v1"]
    )
    return p.parse_args()


# ── Data helpers ────────────────────────────────────────────────────────────────


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


# ── Genre-prior estimation from train split ─────────────────────────────────────


def estimate_genre_emotion_prior(
    train_path: Path, merge_profile: str
) -> dict[str, dict[str, float]]:
    """
    Estimate P(emotion | genre) from the train split.
    Returns {genre_label: {emotion_label: probability}}.
    """
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for rec in load_jsonl(train_path):
        genre = rec.get("genre_en", "")
        emotion = rec.get("emotion_text", "") or rec.get("emotion_cat_en", "")
        if not genre or not emotion:
            continue
        emotion_mapped = apply_emotion_merge(emotion, merge_profile)
        counts[genre][emotion_mapped] += 1

    prior: dict[str, dict[str, float]] = {}
    for genre, emo_counts in counts.items():
        total = sum(emo_counts.values())
        prior[genre] = {e: c / total for e, c in emo_counts.items()} if total else {}
    return prior


# ── Constrained decoding ────────────────────────────────────────────────────────


def constrained_decode(
    probs: np.ndarray,
    id2label: dict[int, str],
    genre: str,
    merge_profile: str,
) -> tuple[str, float]:
    """
    Given softmax probabilities and predicted genre, return the
    highest-probability emotion that belongs to GENRE_EXPECTED_EMOTIONS[genre].
    Falls back to raw argmax if genre has no expected emotions or none match.
    Returns (emotion_label, confidence).
    """
    expected = get_genre_expected_emotions(genre, merge_profile)
    ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    top1_label = id2label.get(ranked[0], "")
    if top1_label in expected:
        return top1_label, float(probs[ranked[0]])
    for idx in ranked:
        label = id2label.get(idx, "")
        if label in expected:
            return label, float(probs[idx])
    # expected set empty or no overlap — fall back to raw top-1
    return top1_label, float(probs[ranked[0]])


# ── Genre-prior reweighting ─────────────────────────────────────────────────────


def reweight_decode(
    probs: np.ndarray,
    id2label: dict[int, str],
    genre: str,
    prior: dict[str, dict[str, float]],
    lam: float,
) -> tuple[str, float]:
    """
    p'(e) ∝ p(e|text) * prior(e|genre)^λ
    Returns (emotion_label, normalised_confidence).
    """
    genre_prior = prior.get(genre, {})
    n = len(probs)
    weighted = np.zeros(n)
    for i in range(n):
        label = id2label.get(i, "")
        p_genre = genre_prior.get(label, 1e-6)  # smoothed zero
        weighted[i] = probs[i] * (p_genre**lam)

    total = weighted.sum()
    if total > 0:
        weighted /= total

    best_idx = int(np.argmax(weighted))
    return id2label.get(best_idx, ""), float(weighted[best_idx])


# ── Evaluation helpers ──────────────────────────────────────────────────────────


def macro_f1_from_lists(preds: list[str], trues: list[str]) -> float:
    labels = sorted(set(trues))
    return float(
        f1_score(trues, preds, labels=labels, average="macro", zero_division=0)
    )


def poem_level_f1(preds: list[str], trues: list[str], poem_ids: list[str]) -> float:
    """Majority-vote within poem, then compute Macro-F1 over poems."""
    poem_pred: dict[str, list[str]] = defaultdict(list)
    poem_true: dict[str, str] = {}
    for pred, true, pid in zip(preds, trues, poem_ids):
        poem_pred[pid].append(pred)
        poem_true[pid] = true

    p_preds, p_trues = [], []
    for pid in poem_pred:
        counts: dict[str, int] = defaultdict(int)
        for lbl in poem_pred[pid]:
            counts[lbl] += 1
        p_preds.append(max(counts, key=lambda k: counts[k]))
        p_trues.append(poem_true[pid])

    labels = sorted(set(p_trues))
    return float(
        f1_score(p_trues, p_preds, labels=labels, average="macro", zero_division=0)
    )


def genre_plausibility_rate(
    preds: list[str], genres: list[str], merge_profile: str
) -> float:
    """Fraction of predictions that fall within GENRE_EXPECTED_EMOTIONS for their genre."""
    plausible = 0
    total = 0
    for pred, genre in zip(preds, genres):
        if not genre:
            continue
        expected = get_genre_expected_emotions(genre, merge_profile)
        if expected:
            plausible += int(pred in expected)
            total += 1
    return plausible / total if total else 0.0


def mean_partial_credit_score(
    preds: list[str], trues: list[str], audio_emotions: list[str], genres: list[str]
) -> float:
    scores = [
        emotion_partial_credit(p, a, t, g)
        for p, a, t, g in zip(preds, audio_emotions, trues, genres)
    ]
    return float(np.mean(scores))


# ── Main ────────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    merge_profile = args.merge_profile
    lam = args.lambda_prior

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)

    merged_classes = get_merged_emotion_classes(merge_profile)
    num_emo = len(merged_classes)
    id2emo = {i: lbl for i, lbl in enumerate(merged_classes)}

    genre_classes = list(GENRE2ID.keys())
    num_gen = len(genre_classes)
    id2gen = {i: lbl for i, lbl in enumerate(genre_classes)}

    # ── Load emotion model ──────────────────────────────────────────────────────
    logger.info(
        "Loading emotion model: {} ({} classes, window={})",
        EMOTION_CKPT.name,
        num_emo,
        CONTEXT_WIN_EMO,
    )
    emo_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_emo,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    ).to(device)
    state = torch.load(EMOTION_CKPT, map_location=device, weights_only=False)
    emo_model.load_state_dict(state, strict=False)
    emo_model.eval()

    # ── Load genre model ────────────────────────────────────────────────────────
    logger.info(
        "Loading genre model: {} ({} classes, window={})",
        GENRE_CKPT.name,
        num_gen,
        CONTEXT_WIN_GEN,
    )
    gen_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_gen,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    ).to(device)
    gen_state = torch.load(GENRE_CKPT, map_location=device, weights_only=False)
    gen_model.load_state_dict(gen_state, strict=False)
    gen_model.eval()

    # ── Build datasets ──────────────────────────────────────────────────────────
    test_path = PROJECT_ROOT / "data/processed/test.jsonl"
    emo_ds = NabatiTextDataset(
        test_path,
        tokenizer,
        task="emotion_text",
        max_seq_len=MAX_SEQ_LEN,
        context_window=CONTEXT_WIN_EMO,
        emotion_merge_profile=merge_profile,
    )
    gen_ds = NabatiTextDataset(
        test_path,
        tokenizer,
        task="genre",
        max_seq_len=MAX_SEQ_LEN,
        context_window=CONTEXT_WIN_GEN,
    )

    emo_loader = DataLoader(emo_ds, batch_size=BATCH_SIZE, shuffle=False)
    gen_loader = DataLoader(gen_ds, batch_size=BATCH_SIZE, shuffle=False)

    # ── Run emotion inference ───────────────────────────────────────────────────
    all_emo_probs: list[np.ndarray] = []
    all_true_emo: list[str] = []
    all_poem_ids: list[str] = []

    with torch.no_grad():
        for batch in emo_loader:
            logits = emo_model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for p in probs:
                all_emo_probs.append(p)
            for tid in batch["label"].tolist():
                all_true_emo.append(id2emo.get(tid, ""))
            all_poem_ids.extend(batch["poem_id"])

    # ── Run genre inference ─────────────────────────────────────────────────────
    all_pred_genres: list[str] = []
    with torch.no_grad():
        for batch in gen_loader:
            logits = gen_model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits
            pred_ids = logits.argmax(dim=-1).cpu().tolist()
            for gid in pred_ids:
                all_pred_genres.append(id2gen.get(gid, ""))

    assert len(all_emo_probs) == len(all_pred_genres) == len(all_true_emo), (
        f"Length mismatch: {len(all_emo_probs)} vs {len(all_pred_genres)} vs {len(all_true_emo)}"
    )

    # ── Load raw test records for audio emotion reference ───────────────────────
    raw_rows: dict[str, dict] = {}
    for rec in load_jsonl(test_path):
        key = rec["source_poem"] + "|" + rec.get("poet_en", "")
        raw_rows[key] = rec
    audio_emotions: list[str] = [
        (raw_rows.get(pid, {}).get("emotion_audio") or "") for pid in all_poem_ids
    ]
    true_genres_ref: list[str] = [
        (raw_rows.get(pid, {}).get("genre_en") or "") for pid in all_poem_ids
    ]

    # ── Estimate genre-emotion prior from train split ───────────────────────────
    genre_prior = estimate_genre_emotion_prior(
        PROJECT_ROOT / "data/processed/train.jsonl", merge_profile
    )

    # ── Raw baseline ─────────────────────────────────────────────────────────────
    raw_preds = [id2emo.get(int(np.argmax(p)), "") for p in all_emo_probs]

    # ── Constrained decoding ─────────────────────────────────────────────────────
    constrained_preds: list[str] = []
    constrained_confs: list[float] = []
    for probs, genre in zip(all_emo_probs, all_pred_genres):
        label, conf = constrained_decode(probs, id2emo, genre, merge_profile)
        constrained_preds.append(label)
        constrained_confs.append(conf)

    # ── Reweight decoding ────────────────────────────────────────────────────────
    reweight_preds: list[str] = []
    reweight_confs: list[float] = []
    for probs, genre in zip(all_emo_probs, all_pred_genres):
        label, conf = reweight_decode(probs, id2emo, genre, genre_prior, lam)
        reweight_preds.append(label)
        reweight_confs.append(conf)

    # ── Compute metrics ───────────────────────────────────────────────────────────
    def eval_system(preds: list[str], name: str) -> dict:
        clip_f1 = macro_f1_from_lists(preds, all_true_emo)
        poem_f1 = poem_level_f1(preds, all_true_emo, all_poem_ids)
        pc = mean_partial_credit_score(
            preds, all_true_emo, audio_emotions, true_genres_ref
        )
        plaus = genre_plausibility_rate(preds, all_pred_genres, merge_profile)
        logger.info(
            "{}: clip_f1={:.4f}  poem_f1={:.4f}  partial_credit={:.4f}  plausibility={:.1%}",
            name,
            clip_f1,
            poem_f1,
            pc,
            plaus,
        )
        return {
            "clip_macro_f1": round(clip_f1, 4),
            "poem_macro_f1": round(poem_f1, 4),
            "mean_partial_credit": round(pc, 4),
            "genre_plausibility": round(plaus, 4),
        }

    logger.info("=" * 60)
    logger.info("Genre-Conditioned Emotion Evaluation")
    logger.info("=" * 60)
    raw_metrics = eval_system(raw_preds, "raw")
    constrained_metrics = eval_system(constrained_preds, "constrained")
    reweight_metrics = eval_system(reweight_preds, f"reweight(λ={lam})")

    # ── Determine best system ─────────────────────────────────────────────────────
    systems = {
        "raw": raw_metrics,
        "constrained": constrained_metrics,
        f"reweight_lambda{lam}": reweight_metrics,
    }
    best_system = max(systems, key=lambda k: systems[k]["clip_macro_f1"])
    logger.info("Best system by clip Macro-F1: {}", best_system)

    # ── Compute delta raw vs constrained ─────────────────────────────────────────
    delta_constrained = {
        k: round(constrained_metrics[k] - raw_metrics[k], 4) for k in raw_metrics
    }

    # ── Save results ──────────────────────────────────────────────────────────────
    out = {
        "merge_profile": merge_profile,
        "context_window": CONTEXT_WIN_EMO,
        "num_classes": num_emo,
        "lambda_prior": lam,
        "n_test_clips": len(all_emo_probs),
        "systems": systems,
        "best_system": best_system,
        "delta_constrained_vs_raw": delta_constrained,
        "adopted_system": (
            "constrained"
            if constrained_metrics["clip_macro_f1"] >= raw_metrics["clip_macro_f1"]
            else "raw"
        ),
        "note": (
            "Genre-constrained decoding selects the highest-probability emotion "
            "inside GENRE_EXPECTED_EMOTIONS[predicted_genre].  If top-1 is already "
            "genre-plausible it is kept unchanged.  Genre-prior reweighting uses "
            f"p'(e) ∝ p(e|text)*prior(e|genre)^λ with λ={lam}."
        ),
    }
    out_path = PROJECT_ROOT / "outputs/reports/genre_conditioned_emotion.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.info("Results saved → {}", out_path)

    # ── Human-readable summary ────────────────────────────────────────────────────
    logger.info("\n" + "=" * 65)
    logger.info("  Genre-Conditioned Emotion Results")
    logger.info("=" * 65)
    for name, m in systems.items():
        tag = " ← BEST" if name == best_system else ""
        logger.info(
            f"  {name:35s}  clip_F1={m['clip_macro_f1']:.4f}"
            f"  partial_credit={m['mean_partial_credit']:.4f}"
            f"  plaus={m['genre_plausibility']:.1%}{tag}"
        )
    logger.info("\n  Δ constrained vs raw:")
    for k, v in delta_constrained.items():
        arrow = "+" if v >= 0 else ""
        logger.info(f"    {k:25s}: {arrow}{v:+.4f}")
    logger.info(f"\n  Adopted final system: {out['adopted_system']}")
    logger.info("=" * 65)


if __name__ == "__main__":
    main()

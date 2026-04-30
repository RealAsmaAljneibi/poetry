"""
scripts/evaluate_asr.py

4-Tier ASR evaluation for Nabat-AI Whisper fine-tuning.

Tiers
─────
  1. Standard WER  — word error rate (jiwer-compatible definition)
  2. Standard CER  — character error rate (no phonemic weighting)
  3. Soft-CER      — research-informed weighted Levenshtein diagnostic for Gulf Arabic
  4. Category breakdown — per-split (corrected-only vs all),
                          per-genre, and corrected vs non-corrected subsets

Input expectations (JSONL, one record per line):
  text_whisper    : Whisper hypothesis (zero-shot or fine-tuned)
  text_corrected  : human-corrected reference (from Label Studio Pass A)
  genre_en        : genre label (for per-genre breakdown)
  corrected       : bool — True if clip was flagged for ASR correction pass

Usage:
    uv run python scripts/evaluate_asr.py --split test
    uv run python scripts/evaluate_asr.py --split test --hypothesis text_finetuned
    uv run python scripts/evaluate_asr.py --split val --hypothesis text_whisper
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import (
    soft_cer,
    standard_cer,
    standard_wer,
    match_error_rate,
    word_information_lost,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR = Path("data/processed")
REPORT_DIR = Path("outputs/reports")

SPLIT_FILES = {
    "train": DATA_DIR / "train.jsonl",
    "val": DATA_DIR / "val.jsonl",
    "test": DATA_DIR / "test.jsonl",
}


# ── Data loading ──────────────────────────────────────────────────────────────


def load_split(split: str) -> pd.DataFrame:
    path = SPLIT_FILES[split]
    if not path.exists():
        logger.error(f"Split not found: {path}")
        sys.exit(1)
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return pd.DataFrame(records)


# ── Per-record metric computation ─────────────────────────────────────────────


def compute_metrics_row(
    hyp: str,
    ref: str,
) -> dict[str, float]:
    """Compute WER, CER, Soft-CER (dialect-aware), and complement metrics MER/WIL."""
    hyp = str(hyp) if hyp else ""
    ref = str(ref) if ref else ""
    return {
        "wer": standard_wer(hyp, ref),
        "cer": standard_cer(hyp, ref),
        "soft_cer": soft_cer(hyp, ref),
        "mer": match_error_rate(hyp, ref),
        "wil": word_information_lost(hyp, ref),
    }


# ── Aggregation helpers ───────────────────────────────────────────────────────


def summarise(rows: list[dict[str, float]], label: str) -> dict:
    if not rows:
        return {"label": label, "n": 0}
    arr = np.array(
        [[r["wer"], r["cer"], r["soft_cer"], r["mer"], r["wil"]] for r in rows]
    )
    return {
        "label": label,
        "n": len(rows),
        "WER": float(arr[:, 0].mean()),
        "CER": float(arr[:, 1].mean()),
        "Soft-CER (dialect-aware)": float(arr[:, 2].mean()),
        "Soft-CER / CER ratio": (
            float(arr[:, 2].mean() / arr[:, 1].mean())
            if arr[:, 1].mean() > 0
            else float("nan")
        ),
        "MER (complement)": float(arr[:, 3].mean()),
        "WIL (complement)": float(arr[:, 4].mean()),
    }


def print_summary(s: dict) -> None:
    logger.info("─" * 55)
    logger.info(f"  {s['label']}  (n={s['n']})")
    logger.info("─" * 55)
    if s["n"] == 0:
        logger.warning("  No valid records.")
        return
    logger.info(f"  WER       : {s['WER']:.4f}")
    logger.info(f"  CER       : {s['CER']:.4f}")
    logger.info(
        f"  Soft-CER  : {s['Soft-CER (dialect-aware)']:.4f}  [dialect-aware, key metric]"
    )
    logger.info(
        f"  Soft/CER  : {s.get('Soft-CER / CER ratio', float('nan')):.3f}"
        "  (<1 means normalization-aware weighting lowers apparent error)"
    )
    logger.info(f"  MER       : {s.get('MER (complement)', 0):.4f}  [complement]")
    logger.info(f"  WIL       : {s.get('WIL (complement)', 0):.4f}  [complement]")


# ── Main ──────────────────────────────────────────────────────────────────────


def main(args: argparse.Namespace) -> None:
    logger.add("logs/evaluate_asr.log", rotation="10 MB")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info(f"ASR Evaluation  |  split={args.split}  |  hyp={args.hypothesis}")
    logger.info("=" * 60)

    df = load_split(args.split)
    logger.info(f"Loaded {len(df)} records")

    ref_col = "text_corrected"
    hyp_col = args.hypothesis

    if hyp_col not in df.columns:
        logger.error(
            f"Hypothesis column '{hyp_col}' not found. Available: {list(df.columns)}"
        )
        sys.exit(1)

    if ref_col not in df.columns:
        logger.error(f"Reference column '{ref_col}' not found in JSONL.")
        sys.exit(1)

    # Keep only rows with non-empty reference AND hypothesis
    mask = df[ref_col].notna() & df[hyp_col].notna()
    df = df[mask].copy()
    logger.info(f"Records with both hyp + ref: {len(df)}")

    if df.empty:
        logger.error("No valid records to evaluate.")
        sys.exit(1)

    # ── Tier 1–3: compute per-record metrics ─────────────────────────────────
    logger.info("Computing WER / CER / Soft-CER per record...")
    metric_rows: list[dict[str, float]] = [
        compute_metrics_row(str(row[hyp_col]), str(row[ref_col]))
        for _, row in df.iterrows()
    ]

    # ── Tier 1: Overall summary ───────────────────────────────────────────────
    overall = summarise(metric_rows, f"Overall  [{args.split}]")
    print_summary(overall)

    # ── Tier 4a: Corrected vs non-corrected clips ─────────────────────────────
    breakdowns: list[dict] = [overall]

    if "corrected" in df.columns:
        df_corr = df[df["corrected"]]
        df_nocorr = df[~df["corrected"]]
        rows_corr = [
            compute_metrics_row(str(r[hyp_col]), str(r[ref_col]))
            for _, r in df_corr.iterrows()
        ]
        rows_nocorr = [
            compute_metrics_row(str(r[hyp_col]), str(r[ref_col]))
            for _, r in df_nocorr.iterrows()
        ]
        s_corr = summarise(rows_corr, "Corrected clips (dialect-gap)")
        s_nocorr = summarise(rows_nocorr, "Non-corrected clips")
        print_summary(s_corr)
        print_summary(s_nocorr)
        breakdowns.extend([s_corr, s_nocorr])
    else:
        logger.warning("'corrected' column not found — skipping Tier 4a breakdown")

    # ── Tier 4b: Per-genre breakdown ──────────────────────────────────────────
    genre_col = "genre_en"
    genre_summaries: list[dict] = []

    if genre_col in df.columns:
        logger.info("─" * 55)
        logger.info("  Soft-CER per genre")
        logger.info("─" * 55)

        genre_metrics: defaultdict[str, list[dict]] = defaultdict(list)
        for i, row in df.iterrows():
            genre = str(row.get(genre_col, "Unknown")).strip()
            genre_metrics[genre].append(metric_rows[df.index.get_loc(i)])

        for genre in sorted(genre_metrics):
            s = summarise(genre_metrics[genre], f"Genre: {genre}")
            genre_summaries.append(s)
            logger.info(
                f"  {genre[:35]:35s}  n={s['n']:4d} "
                f"WER={s.get('WER', 0):.3f}  "
                f"CER={s.get('CER', 0):.3f}  "
                f"Soft-CER={s.get('Soft-CER (dialect-aware)', 0):.3f}  "
                f"MER={s.get('MER (complement)', 0):.3f}"
            )
    else:
        logger.warning("'genre_en' column not found — skipping per-genre breakdown")

    # ── Per-record delta: CER − Soft-CER ─────────────────────────────────────
    df["wer"] = [r["wer"] for r in metric_rows]
    df["cer"] = [r["cer"] for r in metric_rows]
    df["soft_cer"] = [r["soft_cer"] for r in metric_rows]
    df["mer"] = [r["mer"] for r in metric_rows]
    df["wil"] = [r["wil"] for r in metric_rows]
    df["cer_delta"] = df["cer"] - df["soft_cer"]  # positive = soft helps

    delta_mean = df["cer_delta"].mean()
    delta_std = df["cer_delta"].std()
    logger.info("─" * 55)
    logger.info(
        "  CER − Soft-CER delta  (positive = normalization-aware weighting helps)"
    )
    logger.info(f"  Mean : {delta_mean:+.4f}  Std: {delta_std:.4f}")

    top_helped = df.nlargest(5, "cer_delta")[[ref_col, hyp_col, "cer", "soft_cer"]]
    logger.info("  Top-5 clips where soft weighting helped most:")
    for _, row in top_helped.iterrows():
        logger.info(
            f"    ref='{str(row[ref_col])[:40]}' "
            f"CER={row['cer']:.3f} → Soft={row['soft_cer']:.3f}"
        )

    # ── Save report ───────────────────────────────────────────────────────────
    report = {
        "split": args.split,
        "hypothesis": hyp_col,
        "overall": overall,
        "breakdowns": breakdowns,
        "per_genre": genre_summaries,
        "cer_delta": {
            "mean": float(delta_mean),
            "std": float(delta_std),
            "interpretation": "positive = normalization-aware weighting reduces apparent error",
        },
        "soft_cer_note": (
            "Soft-CER is a dialect-aware key metric inspired by Arabic ASR normalization "
            "practice (e.g. MGB-3 / MR-WER framing), CODA/CAPHI-style dialect equivalence thinking, and "
            "phonologically weighted edit distance. It captures Gulf dialect variation awareness."
        ),
        "metric_tiers": {
            "key": ["WER", "CER", "Soft-CER (dialect-aware)"],
            "complement": ["MER", "WIL"],
        },
    }

    out_json = REPORT_DIR / f"asr_eval_{args.split}_{hyp_col}.json"
    out_csv = REPORT_DIR / f"asr_eval_{args.split}_{hyp_col}.csv"

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Per-record CSV
    df[["wer", "cer", "soft_cer", "mer", "wil", "cer_delta", ref_col, hyp_col]].to_csv(
        out_csv, index=False
    )

    logger.success(f"Report → {out_json}")
    logger.success(f"Per-record CSV → {out_csv}")
    logger.info("=" * 60)
    logger.success("ASR evaluation complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="4-Tier ASR evaluation for Nabat-AI")
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="test",
        help="Dataset split to evaluate (default: test)",
    )
    parser.add_argument(
        "--hypothesis",
        default="text_whisper",
        help="JSONL column containing model hypotheses (default: text_whisper)",
    )
    main(parser.parse_args())

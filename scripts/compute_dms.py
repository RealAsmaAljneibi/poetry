"""
scripts/compute_dms.py

Delivery Mismatch Score (DMS) — cultural analysis of Nabati oral poetry.

Definition:
    For each clip, compare:
      - text_arousal:  Arousal implied by the TEXT emotion label
      - audio_arousal: Arousal measured from the AUDIO

    DMS = 1 if text_arousal ≠ audio_arousal, else 0

Why this is a feature, not a failure:
    Nabati oral poetry is performed art. A poet can deliver a grief-filled
    verse (Sorrow → Low arousal expected) with a forceful, energetic voice
    (High arousal observed) — expressing resilience or defiance through tone.
    This mismatch is culturally meaningful, not a model error.

Output:
    outputs/reports/dms_analysis.json  — overall + per-genre mismatch rates
    outputs/figures/dms_per_genre.png  — bar chart

Usage:
    uv run python scripts/compute_dms.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.arousal_labels import emotion_to_arousal

PROJECT_ROOT = Path(__file__).parent.parent


def compute_dms(master_jsonl: Path) -> dict:
    """Compute DMS statistics over the full dataset."""
    rows = [json.loads(line) for line in open(master_jsonl)]

    total = 0
    mismatches = 0
    skipped = 0

    genre_total: defaultdict[str, int] = defaultdict(int)
    genre_mismatch: defaultdict[str, int] = defaultdict(int)
    examples_mismatch: list[dict] = []
    examples_match: list[dict] = []

    for row in rows:
        emotion_text = row.get("emotion_text")
        emotion_audio = row.get("emotion_audio")
        genre = row.get("genre_en", "Unknown")

        text_ar = emotion_to_arousal(emotion_text)
        audio_ar = emotion_to_arousal(emotion_audio)

        if text_ar is None or audio_ar is None:
            skipped += 1
            continue

        total += 1
        genre_total[genre] += 1
        is_mismatch = text_ar != audio_ar

        if is_mismatch:
            mismatches += 1
            genre_mismatch[genre] += 1
            if len(examples_mismatch) < 5:
                examples_mismatch.append(
                    {
                        "source_poem": row.get("source_poem"),
                        "poet_en": row.get("poet_en"),
                        "genre": genre,
                        "text_corrected": row.get("text_corrected", "")[:80],
                        "emotion_text": emotion_text,
                        "text_arousal": text_ar,
                        "emotion_audio": emotion_audio,
                        "audio_arousal": audio_ar,
                        "interpretation": f"Text says '{emotion_text}' ({text_ar}) but delivery is {audio_ar}",
                    }
                )
        elif len(examples_match) < 3:
            examples_match.append(
                {
                    "source_poem": row.get("source_poem"),
                    "genre": genre,
                    "emotion_text": emotion_text,
                    "text_arousal": text_ar,
                    "emotion_audio": emotion_audio,
                    "audio_arousal": audio_ar,
                }
            )

    overall_dms = mismatches / max(total, 1)
    logger.info(f"Overall DMS: {mismatches}/{total} = {overall_dms:.1%}")

    per_genre = {}
    for genre in sorted(genre_total):
        rate = genre_mismatch[genre] / max(genre_total[genre], 1)
        per_genre[genre] = {
            "total": genre_total[genre],
            "mismatches": genre_mismatch[genre],
            "mismatch_rate": round(rate, 4),
        }
        logger.info(
            f"  {genre:50s} {rate:.1%}  ({genre_mismatch[genre]}/{genre_total[genre]})"
        )

    return {
        "overall": {
            "total_clips": total,
            "skipped": skipped,
            "mismatches": mismatches,
            "mismatch_rate": round(overall_dms, 4),
            "interpretation": (
                "When text_arousal ≠ audio_arousal the poet delivers the verse "
                "with an energy level that contradicts the semantic content — "
                "a culturally significant performance device in Nabati poetry."
            ),
        },
        "per_genre": per_genre,
        "examples_mismatch": examples_mismatch,
        "examples_match": examples_match,
    }


def plot_dms_per_genre(per_genre: dict, out_path: Path) -> None:
    genres = list(per_genre.keys())
    rates = [per_genre[g]["mismatch_rate"] for g in genres]

    # Short genre names for readability
    short = [g.split("(")[0].strip() for g in genres]
    x = np.arange(len(genres))

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(x, [r * 100 for r in rates], color="steelblue", edgecolor="white")
    ax.axhline(
        y=np.mean(rates) * 100,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean: {np.mean(rates):.1%}",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(short, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mismatch Rate (%)")
    ax.set_title(
        "Delivery Mismatch Rate per Genre\n(text-implied arousal ≠ audio arousal)"
    )
    ax.legend()

    for bar, rate in zip(bars, rates):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{rate:.0%}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"DMS per-genre plot saved: {out_path}")


def main() -> None:
    log_path = PROJECT_ROOT / "logs/compute_dms.log"
    logger.add(str(log_path), rotation="10 MB", level="DEBUG")

    master_jsonl = PROJECT_ROOT / "data/processed/master_dataset.jsonl"
    if not master_jsonl.exists():
        logger.error(
            f"master_dataset.jsonl not found at {master_jsonl}. Run just generate-data first."
        )
        sys.exit(1)

    logger.info("Computing Delivery Mismatch Score ...")
    results = compute_dms(master_jsonl)

    out_json = PROJECT_ROOT / "outputs/reports/dms_analysis.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info(f"DMS analysis saved: {out_json}")

    plot_dms_per_genre(
        results["per_genre"],
        PROJECT_ROOT / "outputs/figures/dms_per_genre.png",
    )

    overall = results["overall"]
    logger.info("=" * 50)
    logger.info(
        f"Overall DMS: {overall['mismatch_rate']:.1%}  "
        f"({overall['mismatches']}/{overall['total_clips']} clips)"
    )
    logger.info("=" * 50)


if __name__ == "__main__":
    main()

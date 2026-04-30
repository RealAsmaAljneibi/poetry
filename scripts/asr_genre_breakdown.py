from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.labels import merge_genre_label
from src.evaluation.metrics import soft_cer, standard_cer, standard_wer

PROJECT_ROOT = Path(__file__).parent.parent
REPORT_JSON = PROJECT_ROOT / "outputs/reports/asr_genre_breakdown.json"
REPORT_MD = PROJECT_ROOT / "outputs/reports/asr_genre_breakdown.md"


def load_rows(split: str = "test") -> list[dict]:
    path = PROJECT_ROOT / f"data/processed/{split}.jsonl"
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        hyp = (rec.get("text_whisper") or "").strip()
        ref = (rec.get("text_corrected") or "").strip()
        if not hyp or not ref:
            continue
        rows.append(
            {
                "genre": merge_genre_label(rec.get("genre_en", "")),
                "hyp": hyp,
                "ref": ref,
            }
        )
    return rows


def genre_note(genre: str) -> str:
    short = genre.split("(")[0].strip()
    if short in {"Ghazal", "Badawa"}:
        return (
            "Dialect-heavy lexicon and colloquial morphology raise character variation."
        )
    if short in {"Hikma", "Madih", "Fakhr"}:
        return "More formulaic or formal vocabulary makes transcription easier."
    if short == "Hija":
        return "Satirical short clips often rely on compressed phrasing and local references."
    return "Mixed lexical register."


def summarise(rows: list[dict]) -> dict:
    by_genre: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_genre[row["genre"]].append(row)

    summaries = []
    for genre in sorted(by_genre):
        items = by_genre[genre]
        wers = [standard_wer(item["hyp"], item["ref"]) for item in items]
        cers = [standard_cer(item["hyp"], item["ref"]) for item in items]
        softs = [soft_cer(item["hyp"], item["ref"]) for item in items]
        summaries.append(
            {
                "genre": genre,
                "n_clips": len(items),
                "WER": round(float(np.mean(wers)), 4),
                "CER": round(float(np.mean(cers)), 4),
                "Soft-CER": round(float(np.mean(softs)), 4),
                "notes": genre_note(genre),
            }
        )
    return {"per_genre": summaries}


def render_markdown(summary: dict) -> str:
    lines = [
        "# ASR Per-Genre Breakdown",
        "",
        "Soft-CER is reported here as a research-informed diagnostic only. It follows Arabic",
        "normalization-aware evaluation practice rather than claiming a new benchmark metric.",
        "",
        "| Genre | n_clips | WER | CER | Soft-CER | Notes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in summary["per_genre"]:
        lines.append(
            f"| {row['genre']} | {row['n_clips']} | {row['WER']:.4f} | "
            f"{row['CER']:.4f} | {row['Soft-CER']:.4f} | {row['notes']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    logger.add(PROJECT_ROOT / "logs/asr_genre_breakdown.log", rotation="10 MB")
    rows = load_rows("test")
    summary = summarise(rows)

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    REPORT_MD.write_text(render_markdown(summary), encoding="utf-8")

    logger.success(f"ASR genre breakdown → {REPORT_MD}")
    logger.success(f"ASR genre breakdown JSON → {REPORT_JSON}")


if __name__ == "__main__":
    main()

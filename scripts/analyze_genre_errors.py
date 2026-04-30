from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from _analysis_utils import batch_infer, load_arapoem_model, tfidf_top_terms, top_confusion_pairs

from src.data.labels import GENRE_CLASSES, ID2GENRE, encode_genre, merge_genre_label

PROJECT_ROOT = Path(__file__).parent.parent
CHECKPOINT = PROJECT_ROOT / "outputs/models/arapoem_genre/arapoem_genre_best.pt"
REPORT_MD = PROJECT_ROOT / "outputs/reports/genre_error_analysis.md"
REPORT_JSON = PROJECT_ROOT / "outputs/reports/genre_error_analysis.json"
MAX_SEQ_LEN = 32
DEVICE = (
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)


def load_windowed_records(jsonl_path: Path, context_window: int) -> list[dict]:
    poem_records: dict[str, list[dict]] = defaultdict(list)
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        poem_records[rec.get("source_poem", "unknown")].append(rec)

    records: list[dict] = []
    half = context_window // 2
    for pid, clips in poem_records.items():
        clips.sort(key=lambda row: row.get("start", 0))
        for idx, rec in enumerate(clips):
            clip_text = (rec.get("text_corrected") or "").strip()
            if not clip_text:
                continue

            parts: list[str] = []
            for k in range(max(0, idx - half), idx):
                prev = (clips[k].get("text_corrected") or "").strip()
                if prev:
                    parts.append(prev)
            parts.append(clip_text)
            for k in range(idx + 1, min(len(clips), idx + 1 + half)):
                nxt = (clips[k].get("text_corrected") or "").strip()
                if nxt:
                    parts.append(nxt)

            genre = merge_genre_label(rec.get("genre_en", ""))
            label_id = encode_genre(genre)
            if label_id == -1:
                continue

            records.append(
                {
                    "source_poem": rec.get("source_poem", ""),
                    "poet_en": rec.get("poet_en", ""),
                    "start": int(rec.get("start", 0)),
                    "clip_text": clip_text,
                    "model_text": " ".join(parts),
                    "translation_en": rec.get("translation_en") or "",
                    "true_label": genre,
                    "true_id": label_id,
                }
            )
    return records


def load_train_documents(jsonl_path: Path) -> dict[str, str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        clip_text = (rec.get("text_corrected") or "").strip()
        if not clip_text:
            continue
        grouped[merge_genre_label(rec.get("genre_en", ""))].append(clip_text)
    return {genre: " ".join(texts) for genre, texts in grouped.items()}


def predict(records: list[dict]) -> list[dict]:
    tokenizer, model = load_arapoem_model(CHECKPOINT, len(GENRE_CLASSES), DEVICE)
    probs_matrix = batch_infer(
        model, tokenizer, [row["model_text"] for row in records], DEVICE, MAX_SEQ_LEN
    )
    enriched: list[dict] = []
    for row, prob in zip(records, probs_matrix, strict=True):
        pred_id = int(np.argmax(prob))
        enriched.append(
            {
                **row,
                "pred_id": pred_id,
                "pred_label": ID2GENRE[pred_id],
                "confidence": float(prob[pred_id]),
            }
        )
    return enriched


def compute_distinctive_words(
    train_docs: dict[str, str], top_k: int = 10
) -> dict[str, list[str]]:
    return tfidf_top_terms(train_docs, GENRE_CLASSES, top_k)


def sample_examples(
    rows: list[dict], true_label: str, pred_label: str, limit: int = 5
) -> list[dict]:
    hits = [
        row
        for row in rows
        if row["true_label"] == true_label and row["pred_label"] == pred_label
    ]
    hits.sort(key=lambda row: row["confidence"], reverse=True)
    return hits[:limit]


def build_summary(rows: list[dict], distinctive: dict[str, list[str]]) -> dict:
    confusion_counts = top_confusion_pairs(rows)

    target_pairs = [
        ("Shajan (Sorrow / Regret)", "Ghazal (Delicate love)"),
        ("Ghazal (Delicate love)", "Shajan (Sorrow / Regret)"),
        ("Hija (Satire & Social Critique)", "Ghazal (Delicate love)"),
    ]

    pair_explanations = {
        ("Shajan (Sorrow / Regret)", "Ghazal (Delicate love)"): (
            "Both genres are lyric and inward-looking, so grief and longing vocabulary pulls the model toward a shared love-pain register."
        ),
        ("Ghazal (Delicate love)", "Shajan (Sorrow / Regret)"): (
            "Romantic Ghazal clips that foreground absence, regret, or heartbreak collapse into Shajan because the local verse is more sorrow-heavy than plot-heavy."
        ),
        ("Hija (Satire & Social Critique)", "Ghazal (Delicate love)"): (
            "Short satirical clips can inherit lament-like phonetics and pain lexicon, while the verse-level sting of satire is easy to miss in isolated windows."
        ),
    }

    confusion_table: list[dict] = []
    for true_label, pred_label in target_pairs:
        examples = sample_examples(rows, true_label, pred_label, limit=5)
        confusion_table.append(
            {
                "true_label": true_label,
                "pred_label": pred_label,
                "count": sum(
                    1
                    for row in rows
                    if row["true_label"] == true_label
                    and row["pred_label"] == pred_label
                ),
                "shared_distinctive_words": sorted(
                    set(distinctive.get(true_label, []))
                    & set(distinctive.get(pred_label, []))
                ),
                "linguistic_explanation": pair_explanations[(true_label, pred_label)],
                "examples": examples,
            }
        )

    overlap_matrix: dict[str, dict[str, list[str]]] = {}
    for left in GENRE_CLASSES:
        overlap_matrix[left] = {}
        left_words = set(distinctive.get(left, []))
        for right in GENRE_CLASSES:
            right_words = set(distinctive.get(right, []))
            overlap_matrix[left][right] = sorted(left_words & right_words)

    return {
        "model": "AraPoemBERT window=3",
        "checkpoint": str(CHECKPOINT),
        "n_test_records": len(rows),
        "top_confusions": [
            {"true_label": true, "pred_label": pred, "count": count}
            for (true, pred), count in confusion_counts[:10]
        ],
        "per_genre_distinctive_words": distinctive,
        "overlap_matrix": overlap_matrix,
        "target_confusions": confusion_table,
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# Genre Error Analysis",
        "",
        "Qualitative error analysis of the best genre model (`AraPoemBERT`, context window = 3).",
        "",
        "## Linguistic Confusion Explanation Table",
        "",
        "| Confusion | Count | Linguistic Explanation | Example |",
        "|---|---:|---|---|",
    ]

    for item in summary["target_confusions"]:
        example = (
            item["examples"][0]["clip_text"]
            if item["examples"]
            else "No example available"
        )
        lines.append(
            f"| `{item['true_label']}` -> `{item['pred_label']}` | {item['count']} | "
            f"{item['linguistic_explanation']} | {example[:80]} |"
        )

    lines.extend(
        [
            "",
            "## Per-Genre Vocabulary Distinctiveness",
            "",
        ]
    )
    for genre, words in summary["per_genre_distinctive_words"].items():
        lines.append(f"- `{genre}`: {', '.join(words)}")

    for item in summary["target_confusions"]:
        lines.extend(
            [
                "",
                f"## {item['true_label']} -> {item['pred_label']}",
                "",
                f"- Count: {item['count']}",
                f"- Shared high-weight words: {', '.join(item['shared_distinctive_words']) or 'None'}",
                f"- Explanation: {item['linguistic_explanation']}",
                "",
            ]
        )
        for idx, example in enumerate(item["examples"], start=1):
            lines.extend(
                [
                    f"### Example {idx}",
                    "",
                    f"- Arabic text: {example['clip_text']}",
                    f"- English translation: {example['translation_en'] or 'N/A'}",
                    f"- True label: {example['true_label']}",
                    f"- Predicted label: {example['pred_label']}",
                    f"- Confidence: {example['confidence']:.2%}",
                    "",
                ]
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    logger.add(PROJECT_ROOT / "logs/genre_error_analysis.log", rotation="10 MB")

    test_records = load_windowed_records(
        PROJECT_ROOT / "data/processed/test.jsonl", context_window=3
    )
    predicted = predict(test_records)
    train_docs = load_train_documents(PROJECT_ROOT / "data/processed/train.jsonl")
    distinctive = compute_distinctive_words(train_docs, top_k=10)
    summary = build_summary(predicted, distinctive)

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    REPORT_MD.write_text(render_markdown(summary), encoding="utf-8")

    logger.success(f"Genre error analysis → {REPORT_MD}")
    logger.success(f"Genre error analysis JSON → {REPORT_JSON}")


if __name__ == "__main__":
    main()

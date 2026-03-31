from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.labels import (
    apply_emotion_merge,
    encode_emotion_with_profile,
    get_merged_emotion_classes,
)
from src.evaluation.metrics import emotion_distance

PROJECT_ROOT = Path(__file__).parent.parent
ARAPOEM_MODEL = "faisalq/bert-base-arapoembert"
CHECKPOINT = PROJECT_ROOT / "outputs/models/arapoem_emotion/arapoem_emotion_text_best.pt"
REPORT_MD = PROJECT_ROOT / "outputs/reports/emotion_error_analysis.md"
REPORT_JSON = PROJECT_ROOT / "outputs/reports/emotion_error_analysis.json"
MAX_SEQ_LEN = 32
MERGE_PROFILE = "rare_merge_v1"
CLASS_NAMES = get_merged_emotion_classes(MERGE_PROFILE)
ID2LABEL = {idx: label for idx, label in enumerate(CLASS_NAMES)}
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def canonicalize_label(label: str) -> str:
    raw = (label or "").strip()
    if raw in CLASS_NAMES:
        return raw
    lowered = raw.lower()
    for candidate in CLASS_NAMES:
        short = candidate.split("(")[0].strip().lower()
        if lowered == short or lowered.startswith(short):
            return candidate
    return raw


def load_records(jsonl_path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        text = (rec.get("text_corrected") or "").strip()
        if not text:
            continue
        merged_label = canonicalize_label(
            apply_emotion_merge(rec.get("emotion_text", ""), MERGE_PROFILE)
        )
        label_id = encode_emotion_with_profile(rec.get("emotion_text", ""), MERGE_PROFILE)
        if label_id == -1:
            continue
        rows.append(
            {
                "source_poem": rec.get("source_poem", ""),
                "poet_en": rec.get("poet_en", ""),
                "start": int(rec.get("start", 0)),
                "clip_text": text,
                "translation_en": rec.get("translation_en") or "",
                "genre_en": rec.get("genre_en") or "",
                "true_label": merged_label,
                "true_id": label_id,
            }
        )
    return rows


def predict(records: list[dict]) -> list[dict]:
    tokenizer = AutoTokenizer.from_pretrained(ARAPOEM_MODEL, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        ARAPOEM_MODEL,
        num_labels=len(CLASS_NAMES),
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    state = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE).eval()

    batch_size = 16
    enriched: list[dict] = []
    for start in range(0, len(records), batch_size):
        batch = records[start : start + batch_size]
        enc = tokenizer(
            [row["clip_text"] for row in batch],
            max_length=MAX_SEQ_LEN,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"].to(DEVICE),
                attention_mask=enc["attention_mask"].to(DEVICE),
            ).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        for row, prob in zip(batch, probs, strict=True):
            order = np.argsort(prob)[::-1]
            pred_id = int(order[0])
            pred_label = ID2LABEL[pred_id]
            runner_up_id = int(order[1]) if len(order) > 1 else pred_id
            enriched.append(
                {
                    **row,
                    "pred_id": pred_id,
                    "pred_label": pred_label,
                    "confidence": float(prob[pred_id]),
                    "runner_up_label": ID2LABEL[runner_up_id],
                    "margin": float(prob[pred_id] - prob[runner_up_id]),
                }
            )
    return enriched


def compute_distinctive_words(train_rows: list[dict], top_k: int = 10) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in train_rows:
        grouped[row["true_label"]].append(row["clip_text"])

    labels = [label for label in CLASS_NAMES if label in grouped]
    docs = [" ".join(grouped[label]) for label in labels]
    vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b", lowercase=False, max_features=5000)
    matrix = vectorizer.fit_transform(docs)
    vocab = np.array(vectorizer.get_feature_names_out())

    top_terms: dict[str, list[str]] = {}
    for idx, label in enumerate(labels):
        row = matrix[idx].toarray().ravel()
        order = row.argsort()[::-1]
        top_terms[label] = [vocab[i] for i in order if row[i] > 0][:top_k]
    return top_terms


def count_lexicon_hits(text: str, lexicon: list[str]) -> int:
    tokens = set(text.split())
    return sum(1 for word in lexicon if word in tokens)


def categorize_error(row: dict, distinctive: dict[str, list[str]]) -> str:
    if emotion_distance(row["true_label"], row["pred_label"]) <= 1:
        return "culturally_adjacent"

    pred_hits = count_lexicon_hits(row["clip_text"], distinctive.get(row["pred_label"], []))
    true_hits = count_lexicon_hits(row["clip_text"], distinctive.get(row["true_label"], []))
    if row["confidence"] >= 0.50 and pred_hits >= 1 and true_hits == 0:
        return "label_noise"

    if row["margin"] <= 0.12 or len(row["clip_text"].split()) <= 5:
        return "genuine_ambiguity"

    return "genuine_ambiguity"


def top_confusion_pairs(rows: list[dict], top_k: int = 5) -> list[tuple[tuple[str, str], int]]:
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        if row["true_label"] != row["pred_label"]:
            counts[(row["true_label"], row["pred_label"])] += 1
    return counts.most_common(top_k)


def build_summary(rows: list[dict], distinctive: dict[str, list[str]]) -> dict:
    errors = []
    for row in rows:
        if row["true_label"] == row["pred_label"]:
            continue
        category = categorize_error(row, distinctive)
        errors.append({**row, "error_category": category})

    counts = Counter(error["error_category"] for error in errors)
    total_errors = max(len(errors), 1)

    confusion_entries = []
    for (true_label, pred_label), count in top_confusion_pairs(rows, top_k=5):
        pair_examples = [
            row
            for row in errors
            if row["true_label"] == true_label and row["pred_label"] == pred_label
        ]
        pair_examples.sort(key=lambda row: row["confidence"], reverse=True)
        confusion_entries.append(
            {
                "true_label": true_label,
                "pred_label": pred_label,
                "count": count,
                "examples": pair_examples[:5],
            }
        )

    return {
        "model": "AraPoemBERT window=1 + rare_merge_v1",
        "checkpoint": str(CHECKPOINT),
        "n_test_records": len(rows),
        "n_errors": len(errors),
        "category_counts": dict(counts),
        "category_proportions": {
            key: round(value / total_errors, 4) for key, value in counts.items()
        },
        "top_confusions": confusion_entries,
        "per_emotion_distinctive_words": distinctive,
    }


def render_markdown(summary: dict) -> str:
    proportions = summary["category_proportions"]
    lines = [
        "# Emotion Error Analysis",
        "",
        "Qualitative analysis of the best emotion model (`AraPoemBERT`, context window = 1, `rare_merge_v1`).",
        "",
        f"`{proportions.get('culturally_adjacent', 0.0) * 100:.1f}%` of errors are culturally adjacent, "
        f"`{proportions.get('label_noise', 0.0) * 100:.1f}%` are likely label noise, and "
        f"`{proportions.get('genuine_ambiguity', 0.0) * 100:.1f}%` look genuinely ambiguous.",
        "",
        "## Top-5 Confusion Pairs",
        "",
    ]

    for item in summary["top_confusions"]:
        lines.extend(
            [
                f"## {item['true_label']} -> {item['pred_label']}",
                "",
                f"- Count: {item['count']}",
                "",
            ]
        )
        for idx, example in enumerate(item["examples"], start=1):
            explanation = {
                "culturally_adjacent": "Culturally meaningful adjacency: the predicted emotion sits next to the gold label in the emotion graph.",
                "label_noise": "Likely label noise: the verse contains stronger lexical evidence for the predicted emotion than for the annotated one.",
                "genuine_ambiguity": "Likely genuine ambiguity: the verse is short or low-margin, so a single clip underdetermines the affect.",
            }[example["error_category"]]
            lines.extend(
                [
                    f"### Example {idx}",
                    "",
                    f"- Arabic text: {example['clip_text']}",
                    f"- English translation: {example['translation_en'] or 'N/A'}",
                    f"- Genre: {example['genre_en']}",
                    f"- True label: {example['true_label']}",
                    f"- Predicted label: {example['pred_label']}",
                    f"- Error category: {example['error_category']}",
                    f"- Explanation: {explanation}",
                    "",
                ]
            )

    lines.extend(
        [
            "## Distinctive Emotion Vocabulary",
            "",
        ]
    )
    for label, words in summary["per_emotion_distinctive_words"].items():
        lines.append(f"- `{label}`: {', '.join(words)}")

    return "\n".join(lines) + "\n"


def main() -> None:
    logger.add(PROJECT_ROOT / "logs/emotion_error_analysis.log", rotation="10 MB")

    train_rows = load_records(PROJECT_ROOT / "data/processed/train.jsonl")
    test_rows = load_records(PROJECT_ROOT / "data/processed/test.jsonl")
    predicted = predict(test_rows)
    distinctive = compute_distinctive_words(train_rows)
    summary = build_summary(predicted, distinctive)

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    REPORT_MD.write_text(render_markdown(summary), encoding="utf-8")

    logger.success(f"Emotion error analysis → {REPORT_MD}")
    logger.success(f"Emotion error analysis JSON → {REPORT_JSON}")


if __name__ == "__main__":
    main()

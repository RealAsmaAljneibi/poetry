"""
Partial-credit evaluation for the current canonical poem-level emotion outputs.

Reads:
  - outputs/reports/emotion_fusion_eval.json
  - outputs/reports/poem_emotion_predictions_test.json

Writes:
  - outputs/reports/emotion_partial_credit_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from loguru import logger
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation.metrics import (
    emotion_partial_credit,
    emotion_ndcg_at_3,
    inter_annotator_kappa,
    krippendorff_alpha_nominal,
    normalize_emotion,
)

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_PREDICTIONS = (
    PROJECT_ROOT / "outputs/reports/poem_emotion_predictions_test.json"
)
DEFAULT_FUSION_REPORT = PROJECT_ROOT / "outputs/reports/emotion_fusion_eval.json"
OUT_PATH = PROJECT_ROOT / "outputs/reports/emotion_partial_credit_eval.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Poem-level emotion partial-credit evaluation"
    )
    p.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    p.add_argument("--fusion-report", type=Path, default=DEFAULT_FUSION_REPORT)
    return p.parse_args()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def predicted_label_for_variant(variant_name: str, payload: dict) -> str:
    if variant_name == "full_fusion":
        return payload.get("emotion_poem_final", "")
    return payload.get("predicted_poem_emotion", "")


def tier_name(score: float) -> str:
    if score == 1.0:
        return "T1_exact"
    if score == 0.65:
        return "T2_adjacent"
    if score == 0.45:
        return "T3_genre_plausible"
    if score == 0.30:
        return "T4_one_cluster_away"
    if score == 0.20:
        return "T5_valid_label"
    return "T0_invalid"


def compute_inter_annotator_agreement(test_jsonl_path: Path) -> dict:
    audio_refs: list[str] = []
    text_refs: list[str] = []
    for line in test_jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        a = normalize_emotion(row.get("emotion_audio", ""))
        t = normalize_emotion(row.get("emotion_text", ""))
        if a and t:
            audio_refs.append(a)
            text_refs.append(t)
    if len(audio_refs) < 2:
        return {"n_pairs": 0, "cohen_kappa": None, "krippendorff_alpha": None}
    return {
        "n_pairs": len(audio_refs),
        "cohen_kappa": round(inter_annotator_kappa(audio_refs, text_refs), 4),
        "krippendorff_alpha": round(
            krippendorff_alpha_nominal(audio_refs, text_refs), 4
        ),
    }


def main() -> None:
    args = parse_args()
    logger.add(
        PROJECT_ROOT / "logs/evaluate_emotion_partial_credit.log", rotation="10 MB"
    )
    predictions = load_json(args.predictions)
    fusion_report = load_json(args.fusion_report)

    test_jsonl = PROJECT_ROOT / "data/processed/test.jsonl"
    agreement = (
        compute_inter_annotator_agreement(test_jsonl) if test_jsonl.exists() else {}
    )

    merge_profile = fusion_report.get("merge_profile", "rare_merge_v1")
    try:
        from src.data.labels import get_merged_emotion_classes

        labels = get_merged_emotion_classes(merge_profile)
    except Exception:
        labels = []

    systems_report = fusion_report.get("test", {}).get("systems", {})
    results: dict[str, dict] = {}
    ndcg_by_variant: dict[str, float] = {}
    for variant_name, poems in predictions.items():
        scores: list[float] = []
        tier_counts: dict[str, int] = defaultdict(int)
        by_true_class: dict[str, list[float]] = defaultdict(list)
        ndcg_scores: list[float] = []
        for payload in poems.values():
            gold = payload.get("gold_poem_emotion", "")
            pred = predicted_label_for_variant(variant_name, payload)
            genre = payload.get("manual_genre", "")
            audio_aux = payload.get("audio_emotion_poem_aux", "")
            score = emotion_partial_credit(pred, audio_aux, gold, genre)
            scores.append(score)
            tier_counts[tier_name(score)] += 1
            by_true_class[gold].append(score)

            # nDCG@3 per poem if we have probability info and labels
            probs = payload.get("poem_probabilities")
            if probs and labels:
                prob_vec = [float(probs.get(label, 0.0)) for label in labels]
                ndcg_scores.append(
                    emotion_ndcg_at_3(prob_vec, gold, audio_aux, genre, labels)
                )

        n = max(len(scores), 1)
        mean_ndcg = round(float(np.mean(ndcg_scores)), 4) if ndcg_scores else None
        ndcg_by_variant[variant_name] = mean_ndcg if mean_ndcg is not None else 0.0
        results[variant_name] = {
            "n_poems": len(scores),
            "mean_partial_credit": round(sum(scores) / n, 4),
            "ndcg_at_3": mean_ndcg,
            "tier_breakdown": {
                tier: {"count": count, "pct": round(count / n, 4)}
                for tier, count in sorted(tier_counts.items())
            },
            "per_class": {
                label: {"mean_pc": round(sum(vals) / len(vals), 4), "n": len(vals)}
                for label, vals in sorted(by_true_class.items())
            },
            "hard_metrics": systems_report.get(variant_name, {}).get(
                "poem_metrics", {}
            ),
        }

    report = {
        "source_predictions": str(args.predictions.relative_to(PROJECT_ROOT)),
        "source_fusion_report": str(args.fusion_report.relative_to(PROJECT_ROOT)),
        "adopted_poem_aggregation": fusion_report.get(
            "adopted_poem_aggregation", "conf_weighted"
        ),
        "adopted_variant_test": fusion_report.get("test", {}).get("adopted_variant"),
        "inter_annotator_agreement": agreement,
        "ndcg_at_3_by_variant": ndcg_by_variant,
        "systems": results,
        "scoring_tiers": {
            "1.00": "Exact match",
            "0.65": "Adjacent emotion cluster",
            "0.45": "Genre-plausible",
            "0.30": "One cluster away",
            "0.20": "Any valid emotion label",
            "0.00": "Invalid / empty",
        },
    }
    OUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.success("Saved partial-credit report → {}", OUT_PATH)


if __name__ == "__main__":
    main()

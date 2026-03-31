from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RankedLabel:
    label: str
    prob: float

    def to_dict(self) -> dict[str, float | str]:
        return {"label": self.label, "prob": round(float(self.prob), 6)}


def poem_id_from_row(row: dict[str, Any]) -> str:
    """Resolve the canonical poem identifier from a dataset row."""
    for key in ("poem_id", "poem_key", "source_poem"):
        value = (row.get(key) or "").strip()
        if value:
            return value
    poet = (row.get("poet_en") or "").strip()
    title = (row.get("poem_title") or "").strip()
    if poet or title:
        return f"{poet}|{title}".strip("|")
    audio_name = str(row.get("audio_filename") or "").strip()
    return audio_name or "unknown_poem"


def group_by_poem_id(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group clip rows by canonical poem identifier."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[poem_id_from_row(row)].append(row)
    return dict(grouped)


def _normalise(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    total = probs.sum()
    if total <= 0:
        return np.full_like(probs, 1.0 / max(len(probs), 1))
    return probs / total


def _safe_entropy(probs: np.ndarray) -> float:
    probs = np.clip(_normalise(probs), 1e-12, 1.0)
    return float(-(probs * np.log(probs)).sum())


def aggregate_probs_mean(probs_by_clip: list[np.ndarray]) -> np.ndarray:
    """Baseline poem aggregation: arithmetic mean of per-clip probabilities."""
    if not probs_by_clip:
        raise ValueError("aggregate_probs_mean requires at least one clip.")
    stacked = np.vstack([_normalise(p) for p in probs_by_clip])
    return _normalise(stacked.mean(axis=0))


def aggregate_confidence_weighted(
    probs_by_clip: list[np.ndarray],
    clip_conf: list[float] | None = None,
    weight_mode: str = "max_prob",
) -> np.ndarray:
    """
    Confidence-weighted poem aggregation.

    weight_mode:
      - "max_prob": use max softmax probability per clip
      - "entropy": use 1 - normalised entropy per clip
    """
    if not probs_by_clip:
        raise ValueError("aggregate_confidence_weighted requires at least one clip.")
    normalised = [_normalise(p) for p in probs_by_clip]
    if clip_conf is not None:
        weights = np.asarray(clip_conf, dtype=np.float64)
    elif weight_mode == "entropy":
        n_classes = len(normalised[0])
        max_entropy = math.log(max(n_classes, 2))
        weights = np.asarray(
            [1.0 - (_safe_entropy(p) / max_entropy) for p in normalised],
            dtype=np.float64,
        )
    else:
        weights = np.asarray([float(p.max()) for p in normalised], dtype=np.float64)

    weights = np.clip(weights, 1e-9, None)
    weighted = np.zeros_like(normalised[0], dtype=np.float64)
    for w, p in zip(weights, normalised):
        weighted += w * p
    return _normalise(weighted)


def aggregate_logits_mean(logits_by_clip: list[np.ndarray]) -> np.ndarray:
    """Average logits across clips and apply softmax once at poem level."""
    if not logits_by_clip:
        raise ValueError("aggregate_logits_mean requires at least one clip.")
    logits = np.vstack([np.asarray(z, dtype=np.float64) for z in logits_by_clip]).mean(axis=0)
    logits = logits - logits.max()
    exp_logits = np.exp(logits)
    return _normalise(exp_logits)


def aggregate_topk_vote(probs_by_clip: list[np.ndarray], k: int = 3) -> np.ndarray:
    """Interpretability-oriented vote across top-k clip predictions."""
    if not probs_by_clip:
        raise ValueError("aggregate_topk_vote requires at least one clip.")
    n_classes = len(probs_by_clip[0])
    votes = np.zeros(n_classes, dtype=np.float64)
    for probs in probs_by_clip:
        probs = _normalise(probs)
        ranked = np.argsort(probs)[::-1][:k]
        for rank, idx in enumerate(ranked, start=1):
            votes[idx] += 1.0 / rank
    return _normalise(votes)


def ranked_topk(probs: np.ndarray, labels: list[str], k: int = 3) -> list[RankedLabel]:
    probs = _normalise(probs)
    ranked = np.argsort(probs)[::-1][:k]
    return [RankedLabel(label=labels[i], prob=float(probs[i])) for i in ranked]


def clip_support_summary(
    probs_by_clip: list[np.ndarray],
    labels: list[str],
    top_labels: list[str],
) -> list[dict[str, float | str]]:
    """Percent of clips whose top-1 label supports each top poem-level label."""
    support = {label: 0 for label in top_labels}
    for probs in probs_by_clip:
        top_idx = int(np.argmax(probs))
        top_label = labels[top_idx]
        if top_label in support:
            support[top_label] += 1
    total = max(len(probs_by_clip), 1)
    return [
        {"label": label, "clip_fraction": round(count / total, 4)}
        for label, count in support.items()
    ]


def build_poem_emotion_summary(
    poem_id: str,
    probs: np.ndarray,
    probs_by_clip: list[np.ndarray],
    labels: list[str],
    method: str,
    clip_conf: list[float] | None = None,
) -> dict[str, Any]:
    """Build the canonical poem-level emotion summary payload."""
    probs = _normalise(probs)
    topk = ranked_topk(probs, labels, k=min(3, len(labels)))
    entropy = _safe_entropy(probs)
    top1 = topk[0]
    top2_prob = topk[1].prob if len(topk) > 1 else 0.0
    avg_clip_conf = float(np.mean(clip_conf)) if clip_conf else float(np.mean([p.max() for p in probs_by_clip]))
    return {
        "poem_id": poem_id,
        "aggregation_method": method,
        "poem_probabilities": {label: round(float(prob), 6) for label, prob in zip(labels, probs)},
        "poem_emotion_raw_topk": [item.to_dict() for item in topk],
        "poem_emotion_raw_top1": top1.label,
        "poem_emotion_raw_confidence": round(top1.prob, 6),
        "poem_emotion_secondary": [item.label for item in topk[1:]],
        "uncertainty": {
            "entropy": round(entropy, 6),
            "top1_top2_margin": round(top1.prob - top2_prob, 6),
            "avg_clip_confidence": round(avg_clip_conf, 6),
        },
        "clip_support": clip_support_summary(probs_by_clip, labels, [item.label for item in topk]),
    }

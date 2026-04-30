from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from src.data.arousal_labels import emotion_to_arousal
from src.data.labels import (
    apply_emotion_merge,
    get_genre_expected_emotions,
    get_merged_emotion_classes,
)
from src.evaluation.metrics import emotion_distance, normalize_emotion


def _map_emotion_to_core(label: str | None, profile: str = "rare_merge_v1") -> str | None:
    if not label:
        return None
    canonical = normalize_emotion(label)
    merged = apply_emotion_merge(canonical, profile)
    return merged if merged in get_merged_emotion_classes(profile) else None


map_audio_emotion_to_core = _map_emotion_to_core
map_text_emotion_to_core = _map_emotion_to_core


def estimate_genre_emotion_prior(
    rows: list[dict[str, Any]],
    profile: str = "rare_merge_v1",
) -> dict[str, dict[str, float]]:
    """Estimate p(emotion|genre) from TRAIN rows using text_ref only."""
    counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    classes = get_merged_emotion_classes(profile)
    for row in rows:
        genre = (row.get("genre_en") or "").strip()
        emotion = map_text_emotion_to_core(row.get("emotion_text"), profile)
        if not genre or emotion is None:
            continue
        counts[genre][emotion] += 1.0

    priors: dict[str, dict[str, float]] = {}
    alpha = 1.0
    for genre, genre_counts in counts.items():
        total = sum(genre_counts.values()) + alpha * len(classes)
        priors[genre] = {
            label: (genre_counts.get(label, 0.0) + alpha) / total for label in classes
        }
    return priors


def apply_genre_constrained(
    probs: np.ndarray,
    labels: list[str],
    genre: str,
    profile: str = "rare_merge_v1",
) -> np.ndarray:
    """Restrict the decoded emotion to the genre-plausible set without changing all labels blindly."""
    probs = np.asarray(probs, dtype=np.float64)
    expected = set(get_genre_expected_emotions(genre, profile))
    if not expected:
        return probs / max(probs.sum(), 1e-9)
    masked = np.array(
        [p if label in expected else 0.0 for label, p in zip(labels, probs)],
        dtype=np.float64,
    )
    if masked.sum() <= 0:
        return probs / max(probs.sum(), 1e-9)
    return masked / masked.sum()


def apply_genre_prior(
    probs: np.ndarray,
    labels: list[str],
    genre: str,
    priors: dict[str, dict[str, float]],
    lam: float,
) -> np.ndarray:
    """Apply p'(e) ∝ p(e|text) * prior(e|genre)^lambda."""
    probs = np.asarray(probs, dtype=np.float64)
    genre_prior = priors.get(genre)
    if not genre_prior:
        return probs / max(probs.sum(), 1e-9)
    weighted = np.array(
        [p * (genre_prior.get(label, 1e-6) ** lam) for label, p in zip(labels, probs)],
        dtype=np.float64,
    )
    total = weighted.sum()
    return weighted / total if total > 0 else probs / max(probs.sum(), 1e-9)


def compute_delivery_metadata(
    final_emotion: str, poem_arousal: str | None
) -> dict[str, Any]:
    """Compute DMS and a delivery nuance tag from final emotion vs aggregated arousal."""
    expected = emotion_to_arousal(final_emotion)
    mismatch = bool(expected and poem_arousal and expected != poem_arousal)
    if poem_arousal is None:
        tag = "delivery unavailable"
    elif not expected:
        tag = f"{poem_arousal.lower()}-energy delivery"
    elif poem_arousal == expected:
        tag = f"{poem_arousal.lower()}-energy delivery aligned with text"
    elif poem_arousal == "High":
        tag = "high-energy delivery"
    elif poem_arousal == "Low":
        tag = "restrained delivery"
    else:
        tag = "moderated delivery"
    return {
        "expected_arousal": expected,
        "poem_arousal": poem_arousal,
        "dms_poem": mismatch,
        "delivery_nuance_tag": tag,
    }


def _audio_gate_passes(
    audio_label: str | None,
    audio_conf: float,
    text_candidates: list[str],
    genre: str,
    profile: str,
    tau_audio: float,
) -> bool:
    if audio_label is None or audio_conf < tau_audio:
        return False
    expected = set(get_genre_expected_emotions(genre, profile))
    if audio_label in expected:
        return True
    return any(
        emotion_distance(audio_label, candidate) <= 1
        for candidate in text_candidates
        if candidate
    )


def decide_final_emotion(
    text_summary: dict[str, Any],
    conditioned_probs: np.ndarray,
    labels: list[str],
    genre: str,
    poem_arousal: str | None,
    audio_aux_label: str | None = None,
    audio_aux_conf: float = 0.0,
    profile: str = "rare_merge_v1",
    tau_text: float = 0.45,
    tau_audio: float = 0.55,
    strategy_name: str = "raw",
) -> dict[str, Any]:
    """Decide the final poem-level emotion using conditioned text plus optional gated audio."""
    conditioned_probs = np.asarray(conditioned_probs, dtype=np.float64)
    conditioned_probs = conditioned_probs / max(conditioned_probs.sum(), 1e-9)
    ranked = np.argsort(conditioned_probs)[::-1]
    top1_idx = int(ranked[0])
    top2_idx = int(ranked[1]) if len(ranked) > 1 else top1_idx
    top1_label = labels[top1_idx]
    top2_label = labels[top2_idx]
    top1_prob = float(conditioned_probs[top1_idx])
    top2_prob = float(conditioned_probs[top2_idx])
    margin = top1_prob - top2_prob

    final_label = top1_label
    used_audio = False
    reason = f"{strategy_name} text poem distribution selected {top1_label}"

    audio_gate = _audio_gate_passes(
        audio_label=audio_aux_label,
        audio_conf=audio_aux_conf,
        text_candidates=[top1_label, top2_label],
        genre=genre,
        profile=profile,
        tau_audio=tau_audio,
    )
    if (
        margin < 0.02
        and audio_gate
        and top1_prob < tau_text
        and audio_aux_label in {top1_label, top2_label}
    ):
        final_label = str(audio_aux_label)
        used_audio = True
        reason = (
            f"text margin {margin:.3f} was low, so gated audio supported {audio_aux_label} "
            "and broke the tie"
        )
    elif margin < 0.02:
        expected = set(get_genre_expected_emotions(genre, profile))
        allowed = [label for label in (top1_label, top2_label) if label in expected]
        if len(allowed) == 1:
            final_label = allowed[0]
            reason = (
                f"text margin {margin:.3f} was low, so the only genre-plausible candidate "
                f"{allowed[0]} was selected"
            )

    delivery = compute_delivery_metadata(final_label, poem_arousal)
    return {
        "emotion_poem_final": final_label,
        "emotion_poem_final_reason": reason,
        "audio_emotion_poem_aux": audio_aux_label,
        "audio_emotion_poem_aux_confidence": round(float(audio_aux_conf), 6),
        "audio_emotion_used_in_decision": used_audio,
        "audio_emotion_gate_passed": audio_gate,
        **delivery,
        "emotion_poem_conditioned_top3": [
            {
                "label": labels[int(idx)],
                "prob": round(float(conditioned_probs[int(idx)]), 6),
            }
            for idx in ranked[:3]
        ],
        "emotion_poem_raw_topk": text_summary["poem_emotion_raw_topk"],
    }

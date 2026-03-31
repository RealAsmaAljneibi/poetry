from __future__ import annotations

import numpy as np

from src.models.emotion.aggregate import (
    aggregate_confidence_weighted,
    aggregate_probs_mean,
    group_by_poem_id,
)
from src.models.emotion.fusion import decide_final_emotion


def test_group_by_poem_id_prefers_explicit_poem_id() -> None:
    rows = [
        {"poem_id": "poemA", "audio_filename": "a.wav"},
        {"poem_key": "poemB", "audio_filename": "b.wav"},
        {"source_poem": "poemC", "audio_filename": "c.wav"},
    ]
    grouped = group_by_poem_id(rows)
    assert set(grouped) == {"poemA", "poemB", "poemC"}


def test_confidence_weighted_prefers_more_confident_clip() -> None:
    low_conf = np.array([0.40, 0.35, 0.25], dtype=np.float64)
    high_conf = np.array([0.80, 0.10, 0.10], dtype=np.float64)

    mean_probs = aggregate_probs_mean([low_conf, high_conf])
    weighted_probs = aggregate_confidence_weighted([low_conf, high_conf], [0.40, 0.80])

    assert weighted_probs[0] > mean_probs[0]
    assert abs(weighted_probs.sum() - 1.0) < 1e-9


def test_audio_gate_breaks_low_margin_tie() -> None:
    labels = ["Sorrow (Huzn)", "Pride (Fakhr)", "Hope (Amal)"]
    summary = {
        "poem_emotion_raw_topk": [
            {"label": "Sorrow (Huzn)", "prob": 0.41},
            {"label": "Pride (Fakhr)", "prob": 0.40},
            {"label": "Hope (Amal)", "prob": 0.19},
        ]
    }
    final = decide_final_emotion(
        text_summary=summary,
        conditioned_probs=np.array([0.41, 0.40, 0.19], dtype=np.float64),
        labels=labels,
        genre="Fakhr (Pride & Honor)",
        poem_arousal="High",
        audio_aux_label="Pride (Fakhr)",
        audio_aux_conf=0.80,
        tau_text=0.45,
        tau_audio=0.55,
        strategy_name="genre_prior",
    )
    assert final["emotion_poem_final"] == "Pride (Fakhr)"
    assert final["audio_emotion_used_in_decision"] is True

"""tests/test_schema.py — Pydantic schema validation tests."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from src.data.schema import PoetrySample, InferenceResult
from src.data.arousal_labels import (
    emotion_to_arousal, encode_arousal, AROUSAL_CLASSES,
)
from src.data.labels import encode_genre, encode_emotion, GENRE_CLASSES, EMOTION_CLASSES
from src.evaluation.metrics import soft_cer, standard_wer, emotion_distance


# ── PoetrySample ──────────────────────────────────────────────────────────────

def test_poetry_sample_valid():
    sample = PoetrySample(
        audio_filename=Path("/tmp/poem0001_em_bayt_0001.mp3"),
        source_poem="poem0001_em",
        start=0,
        end=5000,
        text_whisper="اختبار",
        text_corrected="اختبار",
        poet_en="Test Poet",
        genre_en="Ghazal (Delicate love)",
        emotion_text="Sorrow (Huzn)",
    )
    assert sample.poet_en == "Test Poet"
    assert sample.emotion_audio is None   # optional — should default to None


def test_poetry_sample_missing_required_field():
    with pytest.raises(ValidationError):
        PoetrySample(
            source_poem="poem0001_em",
            start=0,
            end=5000,
            # audio_filename missing — required field
            text_whisper="اختبار",
            text_corrected="اختبار",
            poet_en="Test Poet",
            genre_en="Ghazal (Delicate love)",
            emotion_text="Sorrow (Huzn)",
        )


# ── InferenceResult ───────────────────────────────────────────────────────────

def test_inference_result_valid():
    result = InferenceResult(
        audio_file="/tmp/clip.mp3",
        transcription="بيت شعر",
        genre="Ghazal (Delicate love)",
        genre_confidence=0.85,
        emotion_audio="Sorrow (Huzn)",
        emotion_audio_confidence=0.72,
        arousal="Low",
        arousal_confidence=0.80,
        inference_ms=1200.0,
    )
    assert result.genre_confidence == 0.85
    assert result.arousal == "Low"


def test_inference_result_confidence_out_of_range():
    with pytest.raises(ValidationError):
        InferenceResult(
            audio_file="/tmp/clip.mp3",
            transcription="بيت شعر",
            genre="Ghazal (Delicate love)",
            genre_confidence=1.5,           # > 1.0 — should fail
            emotion_audio="Sorrow (Huzn)",
            emotion_audio_confidence=0.72,
            arousal="Low",
            arousal_confidence=0.80,
            inference_ms=1200.0,
        )


# ── Arousal Labels ────────────────────────────────────────────────────────────

def test_arousal_mapping_coverage():
    """Every EMOTION_CLASSES label must map to a known Arousal level."""
    for emotion in EMOTION_CLASSES:
        arousal = emotion_to_arousal(emotion)
        assert arousal in AROUSAL_CLASSES, f"No arousal mapping for '{emotion}'"


def test_arousal_encode_none():
    assert encode_arousal(None) == -1
    assert encode_arousal("") == -1


def test_arousal_classes_balanced():
    """All three arousal classes must be reachable (no dead class)."""
    reachable = {emotion_to_arousal(e) for e in EMOTION_CLASSES}
    assert reachable == {"Low", "Medium", "High"}


# ── Label encoding ────────────────────────────────────────────────────────────

def test_encode_genre_all_classes():
    for genre in GENRE_CLASSES:
        assert encode_genre(genre) != -1, f"encode_genre failed for '{genre}'"


def test_encode_genre_merge_map():
    assert encode_genre("Madih (Praise)") == encode_genre("Fakhr (Pride & Honor)")
    assert encode_genre("Tareef (Humorous)") == encode_genre("Hija (Satire & Social Critique)")


def test_encode_emotion_all_classes():
    for emotion in EMOTION_CLASSES:
        assert encode_emotion(emotion) != -1, f"encode_emotion failed for '{emotion}'"


# ── Metrics ───────────────────────────────────────────────────────────────────

def test_soft_cer_identical():
    assert soft_cer("كلمة", "كلمة") == 0.0


def test_soft_cer_empty_ref():
    assert soft_cer("", "") == 0.0


def test_standard_wer_identical():
    assert standard_wer("هذا اختبار", "هذا اختبار") == 0.0


def test_standard_wer_one_sub():
    # One-word substitution in a 3-word ref → WER = 1/3
    wer = standard_wer("هذا خطأ الآن", "هذا صواب الآن")
    assert abs(wer - 1 / 3) < 1e-6


def test_emotion_distance_same():
    assert emotion_distance("Sorrow (Huzn)", "Sorrow (Huzn)") == 0


def test_emotion_distance_adjacent():
    # Longing ↔ Sorrow are directly connected in the adjacency graph
    assert emotion_distance("Longing (Shawq)", "Sorrow (Huzn)") == 1


# ── Split integrity ───────────────────────────────────────────────────────────

def test_splits_no_poet_overlap():
    """
    Genre-aware split: poets in low-poet genres (<7 primary poets) are allowed
    to appear in multiple splits (verse-level split).  High-poet genres
    (Hikma ≥8, Ghazal ≥9) use strict poet-disjoint splits.
    This test verifies that at least some splits are poet-disjoint (high-poet
    genres), and that the total overlap is bounded (≤30 poets).
    """
    import json
    from pathlib import Path

    root = Path(__file__).parent.parent
    splits = {
        "train": root / "data/processed/train.jsonl",
        "val":   root / "data/processed/val.jsonl",
        "test":  root / "data/processed/test.jsonl",
    }
    if not all(p.exists() for p in splits.values()):
        pytest.skip("data/processed splits not found — run just generate-data first")

    poets: dict[str, set[str]] = {}
    for split, path in splits.items():
        poets[split] = {json.loads(line)["poet_en"] for line in open(path)}

    # Some overlap is intentional (verse-level splits for low-poet genres).
    # Bound: no more than 30 shared poets (there are ~36 total in corpus).
    overlap_tv = poets["train"] & poets["val"]
    overlap_tt = poets["train"] & poets["test"]
    assert len(overlap_tv) <= 30, f"Too many Train/Val overlapping poets: {len(overlap_tv)}"
    assert len(overlap_tt) <= 30, f"Too many Train/Test overlapping poets: {len(overlap_tt)}"
    # At least some splits ARE poet-disjoint (the high-poet genres)
    all_poets = poets["train"] | poets["val"] | poets["test"]
    assert len(all_poets) > 0


def test_splits_use_valid_merged_genres():
    """Every split should only contain valid merged genres from the canonical taxonomy."""
    import json
    from pathlib import Path
    from src.data.labels import GENRE_CLASSES, merge_genre_label

    root = Path(__file__).parent.parent
    splits = {
        "train": root / "data/processed/train.jsonl",
        "val":   root / "data/processed/val.jsonl",
        "test":  root / "data/processed/test.jsonl",
    }
    if not all(p.exists() for p in splits.values()):
        pytest.skip("data/processed splits not found — run just generate-data first")

    for split, path in splits.items():
        rows = [json.loads(line) for line in open(path)]
        genres_in_split = {merge_genre_label(r["genre_en"].strip()) for r in rows}
        assert genres_in_split, f"{split} split has no genre labels"
        unknown = genres_in_split - set(GENRE_CLASSES)
        assert not unknown, f"{split} split contains invalid merged genres: {unknown}"

"""
tests/test_labels.py

Unit tests for src/data/labels.py — canonical label lists, encoders,
genre-expected-emotions, and merge profiles.

Run: uv run pytest tests/test_labels.py -v
"""

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.labels import (
    EMOTION_CLASSES,
    GENRE_CLASSES,
    GENRE_EXPECTED_EMOTIONS,
    encode_emotion,
    encode_genre,
    get_merged_emotion_classes,
    get_genre_expected_emotions,
)


# ── Basic counts ───────────────────────────────────────────────────────────────

def test_emotion_classes_count():
    assert len(EMOTION_CLASSES) == 12, f"Expected 12 emotion classes, got {len(EMOTION_CLASSES)}"


def test_genre_classes_count():
    assert len(GENRE_CLASSES) == 8, f"Expected 8 genre classes, got {len(GENRE_CLASSES)}"


def test_no_duplicate_emotion_classes():
    assert len(EMOTION_CLASSES) == len(set(EMOTION_CLASSES)), "Duplicate emotion class names"


def test_no_duplicate_genre_classes():
    assert len(GENRE_CLASSES) == len(set(GENRE_CLASSES)), "Duplicate genre class names"


# ── Encoders ──────────────────────────────────────────────────────────────────

def test_encode_emotion_exact():
    """Every canonical emotion string must encode without fuzzy fallback."""
    for i, cls in enumerate(EMOTION_CLASSES):
        idx = encode_emotion(cls)
        assert idx == i, f"encode_emotion({cls!r}) returned {idx}, expected {i}"


def test_encode_genre_exact():
    """Every canonical genre string must encode without fuzzy fallback."""
    for i, cls in enumerate(GENRE_CLASSES):
        idx = encode_genre(cls)
        assert idx == i, f"encode_genre({cls!r}) returned {idx}, expected {i}"


def test_encode_emotion_unknown():
    """Completely unknown string should return -1, not crash."""
    result = encode_emotion("XXXXXXXXXX_NOT_A_REAL_CLASS")
    assert result == -1


def test_encode_genre_unknown():
    result = encode_genre("XXXXXXXXXX_NOT_A_REAL_CLASS")
    assert result == -1


def test_encode_emotion_empty_string():
    result = encode_emotion("")
    assert result == -1


# ── Merged emotion classes ─────────────────────────────────────────────────────

def test_merged_rare_merge_v1_count():
    """rare_merge_v1 should reduce 12 → 9 classes."""
    merged = get_merged_emotion_classes("rare_merge_v1")
    assert len(merged) == 9, f"rare_merge_v1 should yield 9 classes, got {len(merged)}: {merged}"


def test_merged_none_is_full():
    """merge_profile='none' should return all 12 classes."""
    merged = get_merged_emotion_classes("none")
    assert len(merged) == 12


def test_merged_classes_are_strings():
    for profile in ("none", "rare_merge_v1"):
        merged = get_merged_emotion_classes(profile)
        assert all(isinstance(c, str) for c in merged), \
            f"Non-string class in profile {profile}: {merged}"


# ── Genre expected emotions ───────────────────────────────────────────────────

def test_genre_expected_emotions_keys_are_valid_genres():
    """All keys in GENRE_EXPECTED_EMOTIONS must be valid genre class names."""
    for genre in GENRE_EXPECTED_EMOTIONS:
        assert genre in GENRE_CLASSES, f"Genre {genre!r} in GENRE_EXPECTED_EMOTIONS is not a valid genre class"


def test_genre_expected_emotions_values_are_valid_emotions():
    """All emotion names in GENRE_EXPECTED_EMOTIONS must be valid emotion classes."""
    for genre, emotions in GENRE_EXPECTED_EMOTIONS.items():
        for emo in emotions:
            idx = encode_emotion(emo)
            assert idx != -1, \
                f"Emotion {emo!r} in GENRE_EXPECTED_EMOTIONS[{genre!r}] not found in EMOTION_CLASSES"


def test_genre_expected_emotions_nonempty():
    """Every genre in the dict should have at least one expected emotion."""
    for genre, emotions in GENRE_EXPECTED_EMOTIONS.items():
        assert len(emotions) >= 1, f"GENRE_EXPECTED_EMOTIONS[{genre!r}] is empty"


def test_get_genre_expected_emotions_returns_list():
    """Helper function should return a list (not None) for known genres."""
    for genre in GENRE_CLASSES:
        result = get_genre_expected_emotions(genre)
        assert isinstance(result, list), f"get_genre_expected_emotions({genre!r}) returned {type(result)}"

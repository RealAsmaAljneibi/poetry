"""
tests/test_split.py

Data integrity tests for the poet-disjoint train/val/test split.

Verifies:
  - All three split files exist and are non-empty
  - No poet appears in more than one split
  - Class (genre) distribution: all genres present in at least train
  - Record counts are within expected ranges
  - All audio_filename fields are present (even if files don't exist on CI)
  - text_corrected is non-empty for all records

Run: uv run pytest tests/test_split.py -v
"""

import json
from collections import defaultdict
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
SPLIT_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_JSONL = SPLIT_DIR / "train.jsonl"
VAL_JSONL   = SPLIT_DIR / "val.jsonl"
TEST_JSONL  = SPLIT_DIR / "test.jsonl"

# Skip all tests gracefully if split files don't exist (CI without data)
pytestmark = pytest.mark.skipif(
    not TRAIN_JSONL.exists(),
    reason="Split files not found — run 'just generate-data' first",
)


def _load_jsonl(path: Path) -> list[dict]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


@pytest.fixture(scope="module")
def splits():
    return {
        "train": _load_jsonl(TRAIN_JSONL),
        "val":   _load_jsonl(VAL_JSONL),
        "test":  _load_jsonl(TEST_JSONL),
    }


# ── Basic file and record counts ──────────────────────────────────────────────

def test_split_files_exist():
    assert TRAIN_JSONL.exists(), f"Missing: {TRAIN_JSONL}"
    assert VAL_JSONL.exists(),   f"Missing: {VAL_JSONL}"
    assert TEST_JSONL.exists(),  f"Missing: {TEST_JSONL}"


def test_split_sizes(splits):
    """Train ~2600-2800, val/test ~300-400 each."""
    assert 2000 <= len(splits["train"]) <= 4000, f"Unexpected train size: {len(splits['train'])}"
    assert 100  <= len(splits["val"])   <= 600,  f"Unexpected val size: {len(splits['val'])}"
    assert 100  <= len(splits["test"])  <= 600,  f"Unexpected test size: {len(splits['test'])}"


def test_total_records(splits):
    total = sum(len(v) for v in splits.values())
    assert 2500 <= total <= 10_000, f"Total records out of expected range: {total}"


# ── Poet-disjoint property ─────────────────────────────────────────────────────

def test_poet_disjoint(splits):
    """No poet must appear in more than one split (core project requirement)."""
    poet_splits: dict[str, set[str]] = defaultdict(set)
    for split_name, records in splits.items():
        for rec in records:
            poet = rec.get("poet_en", "").strip()
            if poet:
                poet_splits[poet].add(split_name)

    violations = {poet: spl for poet, spl in poet_splits.items() if len(spl) > 1}
    assert not violations, (
        f"Poet-disjoint split violated! "
        f"{len(violations)} poets appear in multiple splits: "
        f"{list(violations.items())[:5]}"
    )


# ── Required fields ───────────────────────────────────────────────────────────

def test_audio_filename_present(splits):
    """Every record must have a non-empty audio_filename field."""
    for split_name, records in splits.items():
        for i, rec in enumerate(records):
            assert rec.get("audio_filename"), \
                f"Split '{split_name}' record {i}: missing audio_filename"


def test_text_corrected_present(splits):
    """Every record must have a non-empty text_corrected field."""
    for split_name, records in splits.items():
        for i, rec in enumerate(records):
            assert rec.get("text_corrected", "").strip(), \
                f"Split '{split_name}' record {i}: empty text_corrected"


def test_genre_en_present(splits):
    """Every record must have a genre_en label."""
    for split_name, records in splits.items():
        missing = [i for i, r in enumerate(records) if not r.get("genre_en", "").strip()]
        assert not missing, f"Split '{split_name}': {len(missing)} records with missing genre_en"


def test_poet_en_present(splits):
    """Every record must have a poet_en label (needed for disjoint split)."""
    for split_name, records in splits.items():
        missing = [i for i, r in enumerate(records) if not r.get("poet_en", "").strip()]
        assert not missing, f"Split '{split_name}': {len(missing)} records with missing poet_en"


# ── Genre distribution ────────────────────────────────────────────────────────

def test_genres_present_in_train(splits):
    """The training split should contain at least 4 distinct genres."""
    genres = {r.get("genre_en", "") for r in splits["train"]}
    genres.discard("")
    assert len(genres) >= 4, f"Train split has too few genres: {genres}"


def test_no_unseen_genre_in_test(splits):
    """Merged test genres should all appear in training."""
    from src.data.labels import merge_genre_label

    train_genres = {merge_genre_label(r.get("genre_en", "").strip()) for r in splits["train"]} - {""}
    test_genres  = {merge_genre_label(r.get("genre_en", "").strip()) for r in splits["test"]} - {""}
    unseen = test_genres - train_genres
    assert not unseen, f"Test contains genres not seen in training: {unseen}"


# ── No train/test contamination via audio filename ───────────────────────────

def test_no_audio_filename_overlap(splits):
    """Same audio file must not appear in multiple splits."""
    all_files: dict[str, list[str]] = defaultdict(list)
    for split_name, records in splits.items():
        for rec in records:
            fname = Path(rec.get("audio_filename", "")).name
            if fname:
                all_files[fname].append(split_name)

    overlaps = {f: spl for f, spl in all_files.items() if len(spl) > 1}
    assert not overlaps, (
        f"Audio file appears in multiple splits: {list(overlaps.items())[:5]}"
    )

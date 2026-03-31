from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import app  # noqa: E402


def _first_test_clip() -> str:
    test_path = PROJECT_ROOT / "data/processed/test.jsonl"
    if not test_path.exists():
        pytest.skip("data/processed/test.jsonl not found")
    row = json.loads(test_path.read_text(encoding="utf-8").splitlines()[0])
    return row["audio_filename"]


def test_validate_poem_text_rejects_clearly_non_arabic() -> None:
    result = app.validate_poem_text("this is definitely not arabic poetry")
    assert result["hard_reject"] is True


def test_validate_poem_text_warns_on_uncertain_short_input() -> None:
    result = app.validate_poem_text("يا قلب")
    assert result["hard_reject"] is False


def test_validate_poem_text_accepts_arabic_poetic_text() -> None:
    result = app.validate_poem_text("يا ليل خبرني عن أمر المعاناة\nوهل الهوى يرضى بدمع المقلتين")
    assert result["hard_reject"] is False


def test_get_clip_row_matches_original_filename_only() -> None:
    clip = _first_test_clip()
    matched = app.get_clip_row(clip)
    assert matched is not None
    assert Path(matched["audio_filename"]).name == Path(clip).name

    renamed = str(Path(clip).with_name(f"external_sim_{Path(clip).name}"))
    assert app.get_clip_row(renamed) is None


def test_transcribe_audio_query_prefers_corrected_text_for_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    clip = _first_test_clip()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("Whisper should not be loaded for corpus clips")

    monkeypatch.setattr(app, "load_whisper", fail_if_called)
    query_text, source = app.transcribe_audio_query(Path(clip), device="cpu", use_lora=False)

    assert source == "Corrected transcript (corpus)"
    assert query_text


def test_transcribe_audio_query_falls_back_to_asr_for_external(monkeypatch: pytest.MonkeyPatch) -> None:
    clip = _first_test_clip()
    external_clip = Path(clip).with_name(f"external_sim_{Path(clip).name}")

    monkeypatch.setattr(app, "load_whisper", lambda device, use_lora=False: ("processor", "model"))
    monkeypatch.setattr(app, "transcribe", lambda audio_path, processor, model, device: "قصيدة خارجية")

    query_text, source = app.transcribe_audio_query(external_clip, device="cpu", use_lora=False)

    assert source == "Whisper transcript"
    assert query_text == "قصيدة خارجية"

"""Loader tests for `src.data.dataset.NabatiDataset`."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

from src.data.dataset import NabatiDataset
from src.data.labels import encode_emotion, encode_genre


def _write_test_wav(path: Path, sr: int = 16_000, seconds: float = 0.25) -> None:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    samples = 0.2 * np.sin(2 * np.pi * 440 * t)
    pcm = np.int16(samples * 32767)

    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(pcm.tobytes())


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False) for record in records),
        encoding="utf-8",
    )


def _sample_record(audio_path: Path, **overrides: object) -> dict[str, object]:
    record: dict[str, object] = {
        "audio_filename": str(audio_path),
        "source_poem": "poem0001",
        "poem_title": "اختبار",
        "start": 0,
        "end": 4000,
        "text_whisper": "يا سلام",
        "text_corrected": "يا سلام",
        "poet_en": "Test Poet",
        "poet_ar": "شاعر",
        "genre_en": "Ghazal (Delicate love)",
        "genre_ar": "غزل",
        "emotion_text": "Sorrow (Huzn)",
        "emotion_text_ar": "حزن",
        "emotion_audio": "Pride (Fakhr)",
    }
    record.update(overrides)
    return record


def test_nabati_dataset_loads_item_and_encodes_labels(tmp_path: Path) -> None:
    audio_path = tmp_path / "clip.wav"
    jsonl_path = tmp_path / "samples.jsonl"
    _write_test_wav(audio_path)
    _write_jsonl(jsonl_path, [_sample_record(audio_path)])

    dataset = NabatiDataset(jsonl_path, max_audio_sec=1, is_train=False)

    assert len(dataset) == 1
    item = dataset[0]
    assert item["audio_tensor"].shape[0] == 128
    assert item["transcription"] == "يا سلام"
    assert int(item["genre_id"]) == encode_genre("Ghazal (Delicate love)")
    assert int(item["emotion_id"]) == encode_emotion("Pride (Fakhr)")


def test_nabati_dataset_falls_back_to_text_emotion_when_audio_missing(tmp_path: Path) -> None:
    audio_path = tmp_path / "clip.wav"
    jsonl_path = tmp_path / "samples.jsonl"
    _write_test_wav(audio_path)
    _write_jsonl(
        jsonl_path,
        [_sample_record(audio_path, emotion_audio=None, emotion_text="Hope (Amal)")],
    )

    dataset = NabatiDataset(jsonl_path, max_audio_sec=1, is_train=False)
    item = dataset[0]

    assert int(item["emotion_id"]) == encode_emotion("Hope (Amal)")


def test_nabati_dataset_skips_invalid_rows(tmp_path: Path) -> None:
    audio_path = tmp_path / "clip.wav"
    jsonl_path = tmp_path / "samples.jsonl"
    _write_test_wav(audio_path)
    valid = _sample_record(audio_path)
    invalid = {"source_poem": "missing-required-fields"}
    _write_jsonl(jsonl_path, [invalid, valid])

    dataset = NabatiDataset(jsonl_path, max_audio_sec=1, is_train=False)

    assert len(dataset) == 1
    assert dataset[0]["transcription"] == "يا سلام"


def test_nabati_dataset_missing_audio_returns_silence(tmp_path: Path) -> None:
    jsonl_path = tmp_path / "samples.jsonl"
    missing_audio = tmp_path / "missing.wav"
    _write_jsonl(jsonl_path, [_sample_record(missing_audio)])

    dataset = NabatiDataset(jsonl_path, max_audio_sec=1, is_train=False)
    wav = dataset._load_audio(missing_audio)

    assert wav.shape == (16_000,)
    assert np.allclose(wav, 0.0)

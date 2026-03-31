"""
MotherDuck-style Gradio app for Nabat-AI.

Modes:
    - Audio Upload
    - Record Audio
    - Text Input
    - Search Corpus
    - Poetry Map
"""

from __future__ import annotations

import argparse
import difflib
import html
import json
import os
import socket
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

BOOT_PROJECT_ROOT = Path(__file__).resolve().parent.parent
BOOT_CACHE_DIR = BOOT_PROJECT_ROOT / "outputs" / ".cache"
BOOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(BOOT_CACHE_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(BOOT_CACHE_DIR / "matplotlib"))

import gradio as gr  # noqa: E402
import librosa  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import torch  # noqa: E402
from loguru import logger  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

matplotlib.use("Agg")

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from demo import (  # noqa: E402
    AROUSAL_CLASSES,
    EMOTION_TEXT_CLASSES,
    MAX_SEQ_LEN,
    SAMPLE_RATE,
    _extract_arousal_features,
    get_missing_runtime_assets,
    load_arousal_model,
    load_cnn,
    load_emotion_model,
    load_genre_model,
    load_poem_prediction_lookup,
    load_retriever,
    load_whisper,
    predict_genre,
    run_demo,
    transcribe,
)
from data.arousal_labels import emotion_to_arousal  # noqa: E402
from data.labels import EMOTION_CLASSES  # noqa: E402
from emotion.aggregate import aggregate_logits_mean, aggregate_probs_mean, build_poem_emotion_summary  # noqa: E402
from emotion.fusion import apply_genre_prior, decide_final_emotion, estimate_genre_emotion_prior, map_audio_emotion_to_core  # noqa: E402

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
MAP_CACHE_PATH = REPORT_DIR / "poetry_map_points.json"
PLOT_BG = "#f4efea"
CARD_BG = "#fbf8f5"
BORDER = "#d7d0ca"
TEXT = "#383838"
MUTED = "#7f746a"
ACCENT = "#706458"

CORPUS_PATHS = [
    PROJECT_ROOT / "data/processed/master_dataset.jsonl",
    PROJECT_ROOT / "data/processed/train.jsonl",
    PROJECT_ROOT / "data/processed/val.jsonl",
    PROJECT_ROOT / "data/processed/test.jsonl",
]

_GENRE_CACHE: dict[str, tuple[Any, Any]] = {}
_EMOTION_CACHE: dict[str, tuple[Any, Any]] = {}
_AROUSAL_CACHE: dict[str, tuple[Any, Any]] = {}
_CNN_CACHE: dict[str, Any] = {}
_RETRIEVER_CACHE: dict[str, Any] = {}
# Populated once in build_ui() so helper functions can embed icons without needing icon_map passed as args
_UI_ICON_MAP: dict[str, str] = {}


def normalise_text(text: str) -> str:
    return " ".join((text or "").strip().split())


def html_text(text: str | None) -> str:
    return html.escape(text or "")


def clip_id_from_audio(audio_filename: str | Path) -> str:
    return Path(audio_filename).stem


def confidence_bucket(prob: float | None) -> tuple[str, str]:
    value = float(prob or 0.0)
    if value >= 0.70:
        return "High", "#2f6f54"
    if value >= 0.45:
        return "Medium", "#8a6d2f"
    return "Low", "#8a4731"


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max()
    probs = np.exp(logits)
    return probs / max(probs.sum(), 1e-9)


def ranked_pairs(labels: list[str], probs: np.ndarray, k: int = 3) -> list[dict[str, float | str]]:
    probs = np.asarray(probs, dtype=np.float64)
    ranked = np.argsort(probs)[::-1][:k]
    return [{"label": labels[int(idx)], "prob": float(probs[int(idx)])} for idx in ranked]


def resolve_device(device: str) -> tuple[str, str]:
    if device == "cuda" and torch.cuda.is_available():
        return "cuda", ""
    if device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", ""
    if device in {"cuda", "mps"}:
        return "cpu", f"{device.upper()} is not available on this machine; using CPU."
    return "cpu", ""


def arabic_ratio(text: str) -> float:
    chars = [ch for ch in (text or "") if not ch.isspace()]
    if not chars:
        return 0.0
    arabic = sum(1 for ch in chars if "\u0600" <= ch <= "\u06ff" or "\u0750" <= ch <= "\u077f")
    return arabic / max(len(chars), 1)


def poetry_likeness(text: str) -> tuple[float, list[str]]:
    cleaned = normalise_text(text)
    tokens = cleaned.split()
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    reasons: list[str] = []
    score = 0.0

    if len(tokens) >= 4:
        score += 0.25
    else:
        reasons.append("too few words")
    if len(tokens) >= 8:
        score += 0.15
    if len(lines) >= 2:
        score += 0.20
    if any(token.endswith(("ه", "ك", "ي", "ا", "ان", "ون")) for token in tokens[-4:]):
        score += 0.10
    if sum(ch.isdigit() for ch in cleaned) == 0:
        score += 0.10
    else:
        reasons.append("contains many digits")
    if "http" not in cleaned and "@" not in cleaned:
        score += 0.10
    else:
        reasons.append("looks like a link or handle")
    if arabic_ratio(cleaned) >= 0.65:
        score += 0.10
    if len(tokens) > 40:
        score += 0.05

    return min(score, 1.0), reasons


def validate_poem_text(text: str) -> dict[str, Any]:
    ratio = arabic_ratio(text)
    score, reasons = poetry_likeness(text)
    hard_reject = False
    warn = False
    message = ""

    if ratio < 0.45:
        hard_reject = True
        message = "This input is mostly non-Arabic, so the app cannot analyze it as Nabati poetry."
    elif score < 0.22 and len(normalise_text(text).split()) <= 3:
        hard_reject = True
        joined = ", ".join(reasons) if reasons else "it does not look poetic"
        message = f"This input does not look like an Arabic poem ({joined})."
    elif score < 0.45:
        warn = True
        message = "This text may be prose or a partial phrase, so the poetry analysis may be less reliable."

    return {
        "arabic_ratio": round(ratio, 3),
        "poetry_likeness": round(score, 3),
        "hard_reject": hard_reject,
        "warn": warn,
        "message": message,
    }


def genre_plain_name(genre: str) -> str:
    return genre.split("(")[0].strip() if "(" in genre else genre.strip()


def emotion_plain_name(emotion: str) -> str:
    return emotion.split("(")[0].strip() if "(" in emotion else emotion.strip()


def build_poem_description(
    poem_id: str | None,
    arabic_text: str,
    genre: str,
    top_emotions: list[dict[str, Any]],
    imagery_tags: list[str] | str | None,
    translations: list[str] | None = None,
) -> str:
    translation_lines = [normalise_text(line) for line in (translations or []) if normalise_text(line)]
    if translation_lines:
        snippet = " ".join(translation_lines[:2]).strip()
        if snippet:
            return snippet[:220] + ("..." if len(snippet) > 220 else "")

    if isinstance(imagery_tags, str):
        tags = [tag.strip() for tag in imagery_tags.split(",") if tag.strip()]
    else:
        tags = [tag.strip() for tag in (imagery_tags or []) if tag.strip()]

    emotion_text = ", ".join(emotion_plain_name(str(item["label"])) for item in top_emotions[:2]) or "mixed feeling"
    genre_text = genre_plain_name(genre) or "poetic"
    if tags:
        tag_text = ", ".join(tags[:3])
        return f"Short description: A {genre_text.lower()} poem shaped by {emotion_text.lower()}, with imagery around {tag_text.lower()}."
    if poem_id:
        return f"Short description: A {genre_text.lower()} poem carrying {emotion_text.lower()} in the current model reading."
    return f"Short description: An Arabic poem that the system reads as {genre_text.lower()} with {emotion_text.lower()}."


def _path_to_data_uri(path: str, max_px: int = 160, bg: str = "#f4efea") -> str:
    """Read an image, composite onto the page background, resize, return base64 data URI.

    Compositing onto bg removes the transparent/checkerboard look — the icon
    blends cleanly into the page's own cream background colour.
    """
    import base64, io
    try:
        from PIL import Image
        img = Image.open(path).convert("RGBA")
        img.thumbnail((max_px, max_px), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"
    except Exception:
        import mimetypes
        mime = mimetypes.guess_type(path)[0] or "image/png"
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"data:{mime};base64,{b64}"


def discover_icons() -> dict[str, str]:
    """Map every Arabic cultural icon in src/icons/ to a named UI role.
    Returns a dict of role → base64 data URI (so no file-serving required).

    Role assignments (by filename stem):
        dallah   → "analyze"   (dallah coffee-pot = deep analysis / hospitality of knowledge)
        finjan   → "search"    (finjan coffee-cup  = exploration / seeking)
        dates    → "imagery"   (dates = nature, the Nabati landscape)
        dukhon   → "semiotics" (dukhon incense = hidden meaning / layers of interpretation)
        dihn_oud → "map"       (oud oil = essence, the concentrated poetic corpus)
    """
    icon_dir = PROJECT_ROOT / "src" / "icons"
    if not icon_dir.exists():
        return {}
    files = {
        p.stem.lower().replace(" ", "_"): p
        for p in icon_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".png", ".svg", ".jpg", ".jpeg", ".webp"}
    }
    named: list[tuple[str, str]] = [
        ("dallah",   "analyze"),
        ("finjan",   "search"),
        ("dates",    "imagery"),
        ("dukhon",   "semiotics"),
        ("dihn_oud", "map"),
    ]
    icon_map: dict[str, str] = {}
    for stem, role in named:
        if stem in files:
            try:
                icon_map[role] = _path_to_data_uri(str(files[stem]))
            except Exception:
                pass
    # Fallback: assign any leftover icons to unmapped roles
    roles_in_order = [r for _, r in named]
    assigned_paths = set(files[s] for s, _ in named if s in files and _ in icon_map)
    remaining = [(s, p) for s, p in sorted(files.items()) if p not in assigned_paths]
    for role in roles_in_order:
        if role not in icon_map and remaining:
            _, p = remaining.pop(0)
            try:
                icon_map[role] = _path_to_data_uri(str(p))
            except Exception:
                pass
    return icon_map


def render_inline_icon(data_uri: str | None, alt_text: str, size: int = 22) -> str:
    """Return a small inline <img> using a base64 data URI."""
    if not data_uri:
        return ""
    return (
        f"<img src='{data_uri}' alt='{html_text(alt_text)}' "
        f"style='width:{size}px;height:{size}px;object-fit:contain;"
        f"vertical-align:middle;margin-right:6px;filter:sepia(20%) saturate(80%);' />"
    )


def render_start_icon(data_uri: str | None, fallback_class: str, fallback_symbol: str, alt_text: str) -> str:
    if data_uri:
        return (
            f"<div class='icon-container'>"
            f"<img src='{data_uri}' alt='{html_text(alt_text)}' class='icon-no-bg start-icon-img' />"
            "</div>"
        )
    return f"<div class='icon-fallback {fallback_class}'>{html_text(fallback_symbol)}</div>"


@lru_cache(maxsize=1)
def load_corpus_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in CORPUS_PATHS:
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            audio_name = Path(rec.get("audio_filename", "")).name
            if not audio_name or audio_name in seen:
                continue
            seen.add(audio_name)
            rows.append(rec)
    rows.sort(key=lambda row: (row.get("source_poem", ""), int(row.get("start", 0))))
    return rows


@lru_cache(maxsize=1)
def corpus_indexes() -> dict[str, Any]:
    rows = load_corpus_rows()
    by_audio_name: dict[str, dict[str, Any]] = {}
    by_clip_id: dict[str, dict[str, Any]] = {}
    by_poem_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        audio_name = Path(row.get("audio_filename", "")).name
        by_audio_name[audio_name] = row
        by_clip_id[clip_id_from_audio(audio_name)] = row
        by_poem_id[row.get("source_poem", "")].append(row)

    for poem_rows in by_poem_id.values():
        poem_rows.sort(key=lambda row: int(row.get("start", 0)))

    poem_texts = {
        poem_id: "\n".join(normalise_text(row.get("text_corrected") or row.get("text_whisper") or "") for row in poem_rows).strip()
        for poem_id, poem_rows in by_poem_id.items()
    }
    return {
        "rows": rows,
        "by_audio_name": by_audio_name,
        "by_clip_id": by_clip_id,
        "by_poem_id": dict(by_poem_id),
        "poem_texts": poem_texts,
    }


def get_clip_row(audio_path: str | Path) -> dict[str, Any] | None:
    return corpus_indexes()["by_audio_name"].get(Path(audio_path).name)


def get_poem_rows(poem_id: str | None) -> list[dict[str, Any]]:
    if not poem_id:
        return []
    return corpus_indexes()["by_poem_id"].get(poem_id, [])


def get_full_corrected_text(poem_rows: list[dict[str, Any]]) -> str:
    return "\n".join(normalise_text(row.get("text_corrected") or row.get("text_whisper") or "") for row in poem_rows).strip()


def collect_poem_tags(poem_rows: list[dict[str, Any]]) -> list[str]:
    tags: list[str] = []
    for row in poem_rows:
        for tag in str(row.get("imagery_tags_en") or "").split(","):
            clean = normalise_text(tag)
            if clean and clean not in tags:
                tags.append(clean)
    return tags


def collect_translation_lines(poem_rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    for row in poem_rows:
        clean = normalise_text(row.get("translation_en") or "")
        if clean and clean not in lines:
            lines.append(clean)
    return lines


def match_text_to_corpus(text: str) -> tuple[str | None, dict[str, Any] | None]:
    query = normalise_text(text)
    if not query:
        return None, None

    indexes = corpus_indexes()
    if query in indexes["by_poem_id"]:
        poem_rows = indexes["by_poem_id"][query]
        return query, poem_rows[0] if poem_rows else None

    if query in indexes["by_clip_id"]:
        row = indexes["by_clip_id"][query]
        return row.get("source_poem"), row

    for poem_id, poem_text in indexes["poem_texts"].items():
        if normalise_text(poem_text) == query:
            poem_rows = indexes["by_poem_id"][poem_id]
            return poem_id, poem_rows[0] if poem_rows else None

    for row in indexes["rows"]:
        if normalise_text(row.get("text_corrected", "")) == query or normalise_text(row.get("text_whisper", "")) == query:
            return row.get("source_poem"), row
    return None, None


@lru_cache(maxsize=1)
def load_prediction_lookup_cached() -> dict[str, dict[str, dict[str, Any]]]:
    return load_poem_prediction_lookup()


@lru_cache(maxsize=1)
def get_genre_priors() -> dict[str, dict[str, float]]:
    train_path = PROJECT_ROOT / "data/processed/train.jsonl"
    rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return estimate_genre_emotion_prior(rows, profile="rare_merge_v1")


def get_genre_assets(device: str):
    if device not in _GENRE_CACHE:
        _GENRE_CACHE[device] = load_genre_model(device)
    return _GENRE_CACHE[device]


def get_emotion_assets(device: str):
    if device not in _EMOTION_CACHE:
        _EMOTION_CACHE[device] = load_emotion_model(device)
    return _EMOTION_CACHE[device]


def get_arousal_assets(device: str):
    if device not in _AROUSAL_CACHE:
        _AROUSAL_CACHE[device] = load_arousal_model(device)
    return _AROUSAL_CACHE[device]


def get_cnn_assets(device: str):
    if device not in _CNN_CACHE:
        _CNN_CACHE[device] = load_cnn(device)
    return _CNN_CACHE[device]


def get_retriever_cached(device: str):
    if device not in _RETRIEVER_CACHE:
        _RETRIEVER_CACHE[device] = load_retriever(device)
    return _RETRIEVER_CACHE[device]


def transcribe_audio_query(audio_path: Path, device: str, use_lora: bool) -> tuple[str, str]:
    clip_row = get_clip_row(audio_path)
    if clip_row:
        corrected = normalise_text(clip_row.get("text_corrected") or clip_row.get("text_whisper") or "")
        return corrected, "Corrected transcript (corpus)"

    processor, whisper = load_whisper(device, use_lora=use_lora)
    text = transcribe(audio_path, processor, whisper, device)
    return text, "Whisper transcript"


def run_text_logits(text: str, tokenizer, model, device: str) -> tuple[np.ndarray, np.ndarray]:
    enc = tokenizer(
        text,
        max_length=MAX_SEQ_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        ).logits[0].detach().cpu().numpy()
    probs = softmax_np(logits)
    return logits, probs


def predict_arousal_probs(audio_path: Path, scaler, model, device: str) -> np.ndarray | None:
    feats = _extract_arousal_features(audio_path)
    if feats is None:
        return None
    feats_scaled = scaler.transform(feats.reshape(1, -1))
    x = torch.tensor(feats_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(x)[0].detach().cpu().numpy()
    return softmax_np(logits)


def predict_audio_probs(audio_path: Path, cnn_model, device: str) -> np.ndarray | None:
    if cnn_model is None:
        return None
    wav, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    wav = wav[: 8 * SAMPLE_RATE]
    mel = librosa.feature.melspectrogram(y=wav, sr=SAMPLE_RATE, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    target_len = 8 * SAMPLE_RATE // 512
    if mel_db.shape[1] < target_len:
        mel_db = np.pad(mel_db, ((0, 0), (0, target_len - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :target_len]
    x = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = cnn_model(x)[0].detach().cpu().numpy()
    return softmax_np(logits)


def project_audio_probs_to_core(audio_probs: np.ndarray) -> np.ndarray:
    core = np.zeros(len(EMOTION_TEXT_CLASSES), dtype=np.float64)
    for idx, label in enumerate(EMOTION_CLASSES):
        mapped = map_audio_emotion_to_core(label, profile="rare_merge_v1")
        if mapped and mapped in EMOTION_TEXT_CLASSES:
            core[EMOTION_TEXT_CLASSES.index(mapped)] += float(audio_probs[idx])
    if core.sum() <= 0:
        return np.full(len(EMOTION_TEXT_CLASSES), 1.0 / max(len(EMOTION_TEXT_CLASSES), 1))
    return core / core.sum()


def build_runtime_poem_analysis(
    poem_rows: list[dict[str, Any]],
    device: str,
    selected_clip_audio: str | None = None,
) -> dict[str, Any]:
    poem_id = poem_rows[0].get("source_poem", "") if poem_rows else ""
    if not poem_rows:
        return {}

    cached_raw = load_prediction_lookup_cached().get("raw", {}).get(poem_id, {})
    cached_full = load_prediction_lookup_cached().get("full_fusion", {}).get(poem_id, {})

    emo_tok, emo_model = get_emotion_assets(device)
    if emo_model is None or emo_tok is None:
        return {}

    logits_by_clip: list[np.ndarray] = []
    probs_by_clip: list[np.ndarray] = []
    clip_conf: list[float] = []
    clip_emotions: dict[str, dict[str, Any]] = {}
    for row in poem_rows:
        clip_text = normalise_text(row.get("text_corrected") or row.get("text_whisper") or "")
        logits, probs = run_text_logits(clip_text, emo_tok, emo_model, device)
        logits_by_clip.append(logits)
        probs_by_clip.append(probs)
        clip_conf.append(float(probs.max()))
        clip_emotions[row.get("audio_filename", "")] = {
            "topk": ranked_pairs(EMOTION_TEXT_CLASSES, probs, 3),
            "top1": EMOTION_TEXT_CLASSES[int(np.argmax(probs))],
            "confidence": float(probs.max()),
        }

    selected_audio = selected_clip_audio or poem_rows[0].get("audio_filename", "")
    selected_clip_emotion = clip_emotions.get(selected_audio, next(iter(clip_emotions.values())))

    if cached_raw:
        raw_topk = cached_raw.get("poem_emotion_raw_topk", [])
        raw_probs = np.array(
            [float(cached_raw.get("poem_probabilities", {}).get(label, 0.0)) for label in EMOTION_TEXT_CLASSES],
            dtype=np.float64,
        )
        if raw_probs.sum() <= 0:
            raw_probs = aggregate_logits_mean(logits_by_clip)
        text_summary = {
            "poem_emotion_raw_topk": raw_topk,
            "poem_emotion_raw_top1": cached_raw.get("poem_emotion_raw_top1"),
            "poem_emotion_raw_confidence": cached_raw.get("poem_emotion_raw_confidence"),
            "poem_emotion_secondary": cached_raw.get("poem_emotion_secondary", []),
            "uncertainty": cached_raw.get("uncertainty", {}),
            "clip_support": cached_raw.get("clip_support", []),
        }
    else:
        raw_probs = aggregate_logits_mean(logits_by_clip)
        text_summary = build_poem_emotion_summary(
            poem_id=poem_id,
            probs=raw_probs,
            probs_by_clip=probs_by_clip,
            labels=EMOTION_TEXT_CLASSES,
            method="logit_mean",
            clip_conf=clip_conf,
        )

    poem_arousal_label = None
    poem_arousal_conf = None
    dms_poem = None
    delivery_tag = None
    arousal_curve: list[dict[str, Any]] = []
    scaler, arousal_model = get_arousal_assets(device)
    if scaler is not None and arousal_model is not None:
        arousal_probs_by_clip: list[np.ndarray] = []
        for idx, row in enumerate(poem_rows, start=1):
            audio_path = Path(row.get("audio_filename", ""))
            probs = predict_arousal_probs(audio_path, scaler, arousal_model, device) if audio_path.exists() else None
            if probs is None:
                continue
            arousal_probs_by_clip.append(probs)
            arousal_curve.append(
                {
                    "clip_index": idx,
                    "label": AROUSAL_CLASSES[int(np.argmax(probs))],
                    "confidence": float(probs.max()),
                    "score": float(np.dot(probs, np.array([0.0, 1.0, 2.0]))),
                    "audio_filename": str(audio_path),
                }
            )
        if arousal_probs_by_clip:
            poem_arousal_probs = aggregate_probs_mean(arousal_probs_by_clip)
            arousal_idx = int(np.argmax(poem_arousal_probs))
            poem_arousal_label = AROUSAL_CLASSES[arousal_idx]
            poem_arousal_conf = float(poem_arousal_probs[arousal_idx])

    audio_aux_label = None
    audio_aux_conf = 0.0
    audio_aux_topk: list[dict[str, float | str]] = []
    audio_core_probs = None
    cnn_model = get_cnn_assets(device)
    if cnn_model is not None:
        core_probs_by_clip: list[np.ndarray] = []
        for row in poem_rows:
            audio_path = Path(row.get("audio_filename", ""))
            probs = predict_audio_probs(audio_path, cnn_model, device) if audio_path.exists() else None
            if probs is None:
                continue
            core_probs_by_clip.append(project_audio_probs_to_core(probs))
        if core_probs_by_clip:
            audio_core_probs = aggregate_probs_mean(core_probs_by_clip)
            audio_aux_topk = ranked_pairs(EMOTION_TEXT_CLASSES, audio_core_probs, 3)
            audio_aux_label = str(audio_aux_topk[0]["label"])
            audio_aux_conf = float(audio_aux_topk[0]["prob"])

    manual_genre = poem_rows[0].get("genre_en", "")
    conditioned = apply_genre_prior(raw_probs, EMOTION_TEXT_CLASSES, manual_genre, get_genre_priors(), 1.0)
    final = decide_final_emotion(
        text_summary=text_summary,
        conditioned_probs=conditioned,
        labels=EMOTION_TEXT_CLASSES,
        genre=manual_genre,
        poem_arousal=poem_arousal_label,
        audio_aux_label=audio_aux_label,
        audio_aux_conf=audio_aux_conf,
        profile="rare_merge_v1",
        tau_text=0.45,
        tau_audio=0.55,
        strategy_name="genre_prior",
    )

    if cached_full:
        final["emotion_poem_final"] = cached_full.get("emotion_poem_final", final["emotion_poem_final"])
        final["emotion_poem_final_reason"] = cached_full.get(
            "emotion_poem_final_reason",
            final["emotion_poem_final_reason"],
        )
        final["audio_emotion_poem_aux"] = cached_full.get("audio_emotion_poem_aux", final["audio_emotion_poem_aux"])
        final["audio_emotion_poem_aux_confidence"] = cached_full.get(
            "audio_emotion_poem_aux_confidence",
            final["audio_emotion_poem_aux_confidence"],
        )
        final["audio_emotion_used_in_decision"] = cached_full.get(
            "audio_emotion_used_in_decision",
            final["audio_emotion_used_in_decision"],
        )
        final["dms_poem"] = cached_full.get("dms_poem", final["dms_poem"])
        final["delivery_nuance_tag"] = cached_full.get("delivery_nuance_tag", final["delivery_nuance_tag"])
        poem_arousal_label = cached_full.get("poem_arousal", poem_arousal_label)
        poem_arousal_conf = cached_full.get("poem_arousal_confidence", poem_arousal_conf)

    return {
        "poem_id": poem_id,
        "manual_genre": manual_genre,
        "selected_clip_emotion": selected_clip_emotion,
        "raw_summary": text_summary,
        "raw_probs": raw_probs,
        "conditioned_probs": conditioned,
        "final": final,
        "arousal_poem": poem_arousal_label,
        "arousal_poem_confidence": poem_arousal_conf,
        "dms_poem": final.get("dms_poem", dms_poem),
        "delivery_nuance_tag": final.get("delivery_nuance_tag", delivery_tag),
        "audio_aux_topk": audio_aux_topk,
        "arousal_curve": arousal_curve,
    }


def confidence_badge(prob: float | None) -> str:
    level, color = confidence_bucket(prob)
    pct = f"{100 * float(prob or 0.0):.1f}%"
    return (
        f"<span class='confidence' style='border-color:{color}; color:{color};'>"
        f"{pct} · {level}</span>"
    )


def similarity_badge(score: float) -> str:
    level, color = confidence_bucket(score)
    return (
        f"<span class='confidence' style='border-color:{color}; color:{color};'>"
        f"{score:.3f} · {level}</span>"
    )


def pill(text: str, tone: str = "default") -> str:
    tone_map = {
        "default": ("#f7f1eb", BORDER),
        "soft": ("#f0ece7", BORDER),
        "good": ("#eef7f0", "#bdd5c4"),
        "warn": ("#faf4e8", "#d9c7a3"),
        "muted": ("#f6f4f1", "#ddd5cf"),
    }
    bg, border = tone_map.get(tone, tone_map["default"])
    return f"<span class='pill' style='background:{bg}; border-color:{border};'>{text}</span>"


def diff_html(corrected: str, whisper: str) -> str:
    corrected_words = normalise_text(corrected).split()
    whisper_words = normalise_text(whisper).split()
    pieces = []
    for token in difflib.ndiff(whisper_words, corrected_words):
        marker, word = token[:2], token[2:]
        if marker == "+ ":
            pieces.append(f"<span class='diff-add'>{word}</span>")
        elif marker == "- ":
            pieces.append(f"<span class='diff-del'>{word}</span>")
        elif marker == "  ":
            pieces.append(f"<span>{word}</span>")
    return " ".join(pieces)


def save_emotion_mix_figure(poem_id: str, topk: list[dict[str, Any]]) -> None:
    labels = [str(item["label"]).split("(")[0].strip() for item in topk]
    probs = [float(item["prob"]) for item in topk]
    path = FIGURES_DIR / f"poem_emotion_mix_{poem_id}.png"
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.barh(labels[::-1], probs[::-1], color="#6f665f")
    ax.set_xlim(0, max(max(probs) * 1.15, 0.4))
    ax.set_xlabel("Probability")
    ax.set_title("Poem Emotion Mix")
    ax.grid(axis="x", linestyle=":", alpha=0.25)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_tension_curve_figure(poem_id: str, arousal_curve: list[dict[str, Any]]) -> None:
    if not arousal_curve:
        return
    path = FIGURES_DIR / f"arousal_tension_curve_{poem_id}.png"
    xs = [item["clip_index"] for item in arousal_curve]
    ys = [item["score"] for item in arousal_curve]
    fig, ax = plt.subplots(figsize=(6, 3.0))
    ax.plot(xs, ys, color="#6f665f", linewidth=2.0, marker="o", markersize=4)
    ax.set_ylim(-0.1, 2.1)
    ax.set_yticks([0, 1, 2], ["Low", "Med", "High"])
    ax.set_xlabel("Clip / verse index")
    ax.set_title("Delivery Curve")
    ax.grid(axis="y", linestyle=":", alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_emotion_mix_plot(poem_id: str, topk: list[dict[str, Any]]) -> go.Figure:
    if not topk:
        fig = go.Figure()
        fig.update_layout(template=None, paper_bgcolor=PLOT_BG, plot_bgcolor=PLOT_BG, title="Poem Emotion Mix")
        return fig
    save_emotion_mix_figure(poem_id, topk)
    labels = [str(item["label"]).split("(")[0].strip() for item in topk]
    probs = [float(item["prob"]) for item in topk]
    fig = go.Figure(
        go.Bar(
            x=probs[::-1],
            y=labels[::-1],
            orientation="h",
            marker_color="#6f665f",
            text=[f"{100*p:.1f}%" for p in probs[::-1]],
            textposition="outside",
            hovertemplate="%{y}: %{x:.2%}<extra></extra>",
        )
    )
    fig.update_layout(
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font={"family": "Aeonik Mono, Menlo, monospace", "color": TEXT},
        margin={"l": 24, "r": 12, "t": 36, "b": 24},
        title="Poem Emotion Mix",
        xaxis_title="Probability",
        yaxis_title="",
        height=280,
    )
    return fig


def build_tension_curve_plot(poem_id: str, arousal_curve: list[dict[str, Any]]) -> go.Figure:
    save_tension_curve_figure(poem_id, arousal_curve)
    if not arousal_curve:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor=PLOT_BG,
            plot_bgcolor=PLOT_BG,
            font={"family": "Aeonik Mono, Menlo, monospace", "color": TEXT},
            title="Delivery Curve",
            annotations=[{
                "text": "No poem-level audio curve for this input.",
                "showarrow": False,
                "xref": "paper",
                "yref": "paper",
                "x": 0.5,
                "y": 0.5,
            }],
        )
        return fig
    df = pd.DataFrame(arousal_curve)
    fig = px.line(
        df,
        x="clip_index",
        y="score",
        markers=True,
        hover_data={"label": True, "confidence": ":.2f", "score": False},
    )
    fig.update_traces(line_color="#6f665f", marker_color="#6f665f")
    fig.update_layout(
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font={"family": "Aeonik Mono, Menlo, monospace", "color": TEXT},
        margin={"l": 24, "r": 12, "t": 36, "b": 24},
        title="Tension / Delivery Curve",
        yaxis={"tickvals": [0, 1, 2], "ticktext": ["Low", "Med", "High"], "range": [-0.1, 2.1]},
        xaxis_title="Clip / verse index",
        yaxis_title="Arousal",
        height=280,
        showlegend=False,
    )
    return fig


def token_overlap(query: str, tags: str | None) -> list[str]:
    if not tags:
        return []
    q_words = {word.lower() for word in query.replace("\n", " ").split() if word.strip()}
    t_words = {word.lower().strip(",") for word in tags.split() if word.strip()}
    return sorted(q_words & t_words)


def retrieval_rows(
    hits: list[dict[str, Any]],
    query_text: str,
    current_genre: str | None,
    current_emotion: str | None,
    exclude_poem_id: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for hit in hits:
        if exclude_poem_id and hit.get("source_poem") == exclude_poem_id:
            continue
        matched = token_overlap(query_text, hit.get("imagery_tags_en"))
        rows.append(
            {
                "poem_id": hit.get("source_poem", ""),
                "poet_en": hit.get("poet_en", ""),
                "score": float(hit.get("score", 0.0)),
                "snippet": (hit.get("text_corrected", "") or "")[:90],
                "matched_tags": ", ".join(matched) if matched else "none",
                "same_genre": "yes" if current_genre and hit.get("genre_en") == current_genre else "no",
                "same_emotion": "yes" if current_emotion and hit.get("emotion_text") == current_emotion else "no",
                "genre_en": hit.get("genre_en", ""),
            }
        )
        if len(rows) >= 5:
            break
    return rows


def retrieval_html(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<div class='card'><div class='small'>No retrieval results available yet.</div></div>"
    table_rows = []
    for row in rows:
        snippet = row["snippet"] + ("..." if len(row["snippet"]) >= 90 else "")
        table_rows.append(
            "<tr>"
            f"<td>{row['score']:.3f}</td>"
            f"<td>{row['poet_en']}</td>"
            f"<td>{row['poem_id']}</td>"
            f"<td>{row['genre_en']}</td>"
            f"<td>{row['same_genre']}</td>"
            f"<td>{row['same_emotion']}</td>"
            f"<td>{row['matched_tags']}</td>"
            f"<td>{snippet}</td>"
            "</tr>"
        )
    return (
        "<div class='card'><div class='section-title'>Why these retrieval results?</div>"
        "<table class='compact-table'>"
        "<thead><tr><th>Score</th><th>Poet</th><th>Poem</th><th>Genre</th><th>Same genre</th>"
        "<th>Same emotion</th><th>Matched tags</th><th>Snippet</th></tr></thead>"
        f"<tbody>{''.join(table_rows)}</tbody></table></div>"
    )


def topk_table_html(title: str, items: list[dict[str, Any]]) -> str:
    if not items:
        return ""
    rows = "".join(
        "<tr>"
        f"<td>{item['label']}</td>"
        f"<td>{confidence_badge(float(item['prob']))}</td>"
        "</tr>"
        for item in items
    )
    return (
        f"<div class='table-block'><div class='mini-title'>{title}</div>"
        f"<table class='compact-table'><tbody>{rows}</tbody></table></div>"
    )


def trust_layer_html(is_corpus: bool, audio_aux_used: bool, clip_only: bool) -> str:
    poem_note = (
        "Poem-level aggregation is available because this input matches a corpus poem."
        if is_corpus
        else "This input is not from the corpus; showing clip-level outputs only."
    )
    audio_note = (
        "Audio emotion contributed to the final decision in this case."
        if audio_aux_used
        else "Audio emotion is auxiliary / experimental and is not used unless the reliability gate passes."
    )
    scope_note = "Showing clip-level outputs only." if clip_only else "Showing both clip-level and poem-level outputs."
    return (
        "<div class='card'><div class='section-title'>Model Notes</div>"
        f"<div class='small'>{poem_note}</div>"
        f"<div class='small'>{scope_note}</div>"
        "<div class='small'>Emotion is subjective, so the app shows top-3 predictions and uncertainty instead of a single hard answer only.</div>"
        f"<div class='small'>{audio_note}</div>"
        "<div class='small'>Confidence badge legend: High >= 70%, Medium 45-69%, Low < 45%.</div>"
        "</div>"
    )


def summary_card_html(card: dict[str, Any]) -> str:
    mismatch = "Mismatch" if card.get("dms_poem") else "Aligned"
    mismatch_tone = "warn" if card.get("dms_poem") else "good"
    return (
        "<div class='summary-card'>"
        "<div class='eyebrow'>Final Summary</div>"
        f"{card.get('origin_badge_html', '')}"
        f"<div class='summary-grid'>"
        f"<div><div class='metric-label'>Final Genre</div><div class='metric-value'>{card.get('final_genre', 'unknown')}</div>{card.get('genre_conf_badge', '')}</div>"
        f"<div><div class='metric-label'>Final Poem Emotion</div><div class='metric-value'>{card.get('final_poem_emotion', 'unknown')}</div>{card.get('final_emotion_conf_badge', '')}</div>"
        f"<div><div class='metric-label'>Raw Poem Emotion</div><div class='metric-value'>{card.get('raw_poem_emotion', 'unknown')}</div>{card.get('raw_emotion_conf_badge', '')}</div>"
        f"<div><div class='metric-label'>Delivery / Arousal</div><div class='metric-value'>{card.get('arousal_display', 'Not available')}</div></div>"
        f"<div><div class='metric-label'>Mismatch</div><div class='metric-value'>{pill(mismatch, mismatch_tone)}</div></div>"
        f"<div><div class='metric-label'>Latency</div><div class='metric-value'>{card.get('latency_ms', 'n/a')}</div></div>"
        "</div>"
        "</div>"
    )


def transcript_panel_html(card: dict[str, Any]) -> str:
    diff_block = ""
    if card.get("show_diff"):
        diff_block = (
            "<details class='details'><summary>Difference vs Whisper</summary>"
            f"<div class='small diff-box'>{card['diff_html']}</div></details>"
        )
    whisper_block = ""
    if card.get("whisper_text"):
        whisper_block = (
            "<details class='details'><summary>Whisper transcript used by the ASR model</summary>"
            f"<div class='mono-block'>{card['whisper_text']}</div></details>"
        )
    return (
        "<div class='card'>"
        "<div class='section-title'>Poem Preview</div>"
        f"{pill(card.get('transcript_source', 'Transcript'), 'soft')}"
        f"{pill('Corpus-verified poem', 'good') if card.get('is_corpus') else ''}"
        f"{pill('External audio (not in corpus)', 'warn') if card.get('is_external_audio') else ''}"
        f"{pill(card.get('input_gate_label', ''), 'warn') if card.get('input_gate_label') else ''}"
        f"<div class='transcript-block arabic-preview'>{html_text(card.get('display_text', ''))}</div>"
        f"<div class='small'><strong>English preview:</strong> {html_text(card.get('english_summary', 'No English gloss available.'))}</div>"
        f"{diff_block}{whisper_block}"
        "</div>"
    )


def details_html(card: dict[str, Any]) -> str:
    clip_table = topk_table_html("Selected clip emotion top-3", card.get("clip_topk", []))
    poem_table = topk_table_html("Poem emotion top-3 (raw)", card.get("poem_topk", []))
    final_poem_table = topk_table_html("Poem emotion top-3 (final)", card.get("poem_final_topk", []))
    audio_table = topk_table_html("Audio auxiliary top-3", card.get("audio_aux_topk", []))
    genre_table = topk_table_html("Genre top-3", card.get("genre_topk", []))
    explanation = card.get("final_reason") or "No extra explanation available."
    calm_note = card.get("scope_note", "")
    return (
        "<div class='card'>"
        "<div class='section-title'>Details</div>"
        f"<div class='small'>{calm_note}</div>"
        "<details class='details' open><summary>Predictions</summary>"
        f"{genre_table}{clip_table}{poem_table}{final_poem_table}{audio_table}"
        "</details>"
        "<details class='details'><summary>Delivery explanation</summary>"
        f"<div class='small'>{html_text(card.get('delivery_explanation', 'Delivery information is not available for this input.'))}</div>"
        "</details>"
        "<details class='details'><summary>Decision explanation</summary>"
        f"<div class='small'>{explanation}</div>"
        "</details>"
        "</div>"
    )


def empty_plots() -> tuple[go.Figure, go.Figure]:
    return build_emotion_mix_plot("none", []), build_tension_curve_plot("none", [])


def build_card_outputs(card: dict[str, Any]) -> tuple[str, str, str, go.Figure, go.Figure, str, str, str]:
    emotion_plot = build_emotion_mix_plot(card.get("poem_id") or "external", card.get("poem_topk", []))
    arousal_plot = build_tension_curve_plot(card.get("poem_id") or "external", card.get("arousal_curve", []))
    return (
        summary_card_html(card),
        transcript_panel_html(card),
        details_html(card),
        emotion_plot,
        arousal_plot,
        retrieval_html(card.get("retrieval_rows", [])),
        trust_layer_html(
            is_corpus=bool(card.get("is_corpus")),
            audio_aux_used=bool(card.get("audio_aux_used")),
            clip_only=bool(card.get("clip_only")),
        ),
        build_semiotic_html(card.get("imagery_tags", [])),
    )


def empty_card(message: str) -> tuple[str, str, str, go.Figure, go.Figure, str, str, str]:
    emotion_plot, arousal_plot = empty_plots()
    empty = (
        f"<div class='summary-card'><div class='eyebrow'>Final Summary</div><div class='small'>{message}</div></div>",
        "<div class='card'><div class='section-title'>Transcript</div><div class='small'>Waiting for input.</div></div>",
        "<div class='card'><div class='section-title'>Details</div><div class='small'>No predictions yet.</div></div>",
        emotion_plot,
        arousal_plot,
        "<div class='card'><div class='section-title'>Why these retrieval results?</div><div class='small'>No results yet.</div></div>",
        trust_layer_html(False, False, True),
        build_semiotic_html([]),
    )
    return empty


def offline_setup_notice_html() -> str:
    missing = get_missing_runtime_assets()
    if not missing:
        return ""
    missing_text = ", ".join(missing)
    return (
        "<div class='card'>"
        "<div class='section-title'>Offline Setup Required</div>"
        "<div class='small'>Models not cached. Run: <code>just cache-models</code> (one-time).</div>"
        f"<div class='small'>Missing local assets: {html_text(missing_text)}</div>"
        "</div>"
    )


def search_results_html(results: list[dict[str, Any]]) -> str:
    if not results:
        return "<div class='card'><div class='small'>No poems matched this search.</div></div>"
    rows = []
    for hit in results:
        rows.append(
            "<tr>"
            f"<td>{similarity_badge(float(hit['score']))}</td>"
            f"<td>{hit['poem_id']}</td><td>{hit['poet_en']}</td>"
            f"<td>{hit['n_clips']}</td><td>{hit['genre_en']}</td><td>{hit['emotion_text']}</td>"
            f"<td>{hit['matched_tags']}</td><td>{hit['snippet']}</td>"
            "</tr>"
        )
    return (
        "<div class='card'><div class='section-title'>Search Results</div>"
        "<div class='small'>This mode is retrieval, not classification. Similarity score is the main confidence signal.</div>"
        "<table class='compact-table'><thead><tr><th>Similarity</th><th>Poem</th><th>Poet</th><th>Clips</th><th>Genre</th>"
        "<th>Emotion</th><th>Matched tags</th><th>Snippet</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def card_from_demo_result(
    result,
    selected_clip_row: dict[str, Any] | None,
    imagery_filter: str | None,
    device: str,
) -> dict[str, Any]:
    poem_rows = get_poem_rows(result.poem_id)
    runtime_poem = {}
    if poem_rows and (not result.emotion_poem_raw_topk or result.emotion_poem_final_reason == "Poem cache unavailable; falling back to clip-level emotion."):
        runtime_poem = build_runtime_poem_analysis(poem_rows, device, selected_clip_audio=result.audio_file)

    display_text = normalise_text(
        (selected_clip_row or {}).get("text_corrected") or result.transcription
    )
    whisper_text = result.transcription if selected_clip_row else ""
    clip_topk = [
        {"label": item.label, "prob": item.prob}
        for item in result.emotion_clip_topk
    ]
    poem_topk = [
        {"label": item.label, "prob": item.prob}
        for item in result.emotion_poem_raw_topk
    ]
    if runtime_poem:
        poem_topk = runtime_poem["raw_summary"].get("poem_emotion_raw_topk", poem_topk)

    audio_aux_topk = [
        {"label": item.label, "prob": item.prob}
        for item in result.audio_emotion_poem_aux_topk
    ]
    if runtime_poem and runtime_poem.get("audio_aux_topk"):
        audio_aux_topk = runtime_poem["audio_aux_topk"]

    arousal_curve = runtime_poem.get("arousal_curve", []) if runtime_poem else []
    poem_final_topk = (runtime_poem.get("final", {}) or {}).get("emotion_poem_conditioned_top3", poem_topk)
    poem_tags = collect_poem_tags(poem_rows) if poem_rows else [tag.strip() for tag in str((selected_clip_row or {}).get("imagery_tags_en") or "").split(",") if tag.strip()]
    translation_lines = collect_translation_lines(poem_rows) if poem_rows else [normalise_text((selected_clip_row or {}).get("translation_en") or "")]

    query_text = display_text
    retrieval_rows_payload = retrieval_rows(
        [
            {
                "score": p.score,
                "poet_en": p.poet_en,
                "source_poem": p.source_poem,
                "genre_en": p.genre_en,
                "emotion_text": p.emotion_text,
                "text_corrected": p.text_corrected,
                "imagery_tags_en": p.imagery_tags_en,
            }
            for p in result.similar_poems
        ],
        query_text=query_text,
        current_genre=result.manual_genre or result.genre,
        current_emotion=(runtime_poem.get("final", {}) or {}).get("emotion_poem_final", result.emotion_poem_final),
        exclude_poem_id=result.poem_id,
    )
    final_poem_emotion = (runtime_poem.get("final", {}) or {}).get("emotion_poem_final", result.emotion_poem_final)
    raw_poem_emotion = (runtime_poem.get("raw_summary", {}) or {}).get("poem_emotion_raw_top1", result.emotion_poem_raw_top1)
    final_reason = (runtime_poem.get("final", {}) or {}).get("emotion_poem_final_reason", result.emotion_poem_final_reason)
    english_summary = build_poem_description(
        poem_id=result.poem_id,
        arabic_text=display_text,
        genre=result.manual_genre or result.genre,
        top_emotions=poem_final_topk or poem_topk,
        imagery_tags=poem_tags,
        translations=translation_lines,
    )
    validation = validate_poem_text(display_text)
    delivery_explanation = (runtime_poem.get("final", {}) or {}).get(
        "delivery_nuance_tag",
        "Delivery explanation is not available for this clip.",
    )
    raw_prob = float((poem_topk or [{}])[0].get("prob", result.emotion_poem_raw_confidence or 0.0))
    final_prob = float((poem_final_topk or [{}])[0].get("prob", raw_prob))

    scope_note = (
        "This is a corpus clip, so the card shows corrected transcript text and poem-level aggregation."
        if selected_clip_row
        else "This input is not from the corpus; showing clip-level outputs only."
    )
    return {
        "is_corpus": bool(selected_clip_row),
        "is_external_audio": not bool(selected_clip_row),
        "clip_only": not bool(selected_clip_row),
        "poem_id": result.poem_id,
        "origin_badge_html": pill("Corpus-verified poem", "good") if selected_clip_row else pill("External audio (not in corpus)", "warn"),
        "final_genre": result.manual_genre or result.genre,
        "genre_conf_badge": confidence_badge(result.genre_confidence),
        "final_poem_emotion": final_poem_emotion,
        "final_emotion_conf_badge": confidence_badge(final_prob),
        "raw_poem_emotion": raw_poem_emotion,
        "raw_emotion_conf_badge": confidence_badge(raw_prob),
        "arousal_display": runtime_poem.get("arousal_poem") or result.arousal_poem or result.arousal or "Not available",
        "dms_poem": runtime_poem.get("dms_poem", result.dms_poem),
        "latency_ms": f"{result.inference_ms:.0f} ms",
        "display_text": display_text,
        "english_summary": english_summary,
        "whisper_text": whisper_text,
        "transcript_source": "Corrected transcript (corpus)" if selected_clip_row else "Whisper transcript",
        "show_diff": bool(selected_clip_row and normalise_text(display_text) != normalise_text(whisper_text)),
        "diff_html": diff_html(display_text, whisper_text) if selected_clip_row else "",
        "input_gate_label": validation["message"] if validation["warn"] else "",
        "genre_topk": [{"label": item.label, "prob": item.prob} for item in result.genre_topk],
        "clip_topk": clip_topk,
        "poem_topk": poem_topk,
        "poem_final_topk": poem_final_topk,
        "audio_aux_topk": audio_aux_topk,
        "audio_aux_used": runtime_poem.get("final", {}).get("audio_emotion_used_in_decision", result.audio_emotion_used_in_decision),
        "final_reason": final_reason,
        "delivery_explanation": delivery_explanation,
        "scope_note": scope_note,
        "arousal_curve": arousal_curve,
        "retrieval_rows": retrieval_rows_payload,
        "imagery_tags": poem_tags,
    }


def card_from_text_input(text: str, top_k: int, imagery_filter: str, device: str) -> dict[str, Any]:
    poem_id, clip_row = match_text_to_corpus(text)
    query_text = normalise_text(text)
    validation = validate_poem_text(text)
    if validation["hard_reject"]:
        return {
            "is_corpus": False,
            "is_external_audio": False,
            "clip_only": True,
            "poem_id": None,
            "origin_badge_html": "",
            "final_genre": "Input rejected",
            "genre_conf_badge": "",
            "final_poem_emotion": "Not analyzed",
            "final_emotion_conf_badge": "",
            "raw_poem_emotion": "Not analyzed",
            "raw_emotion_conf_badge": "",
            "arousal_display": "Not available",
            "dms_poem": None,
            "latency_ms": "n/a",
            "display_text": query_text,
            "english_summary": validation["message"],
            "whisper_text": "",
            "transcript_source": "Text input",
            "show_diff": False,
            "diff_html": "",
            "input_gate_label": "Input rejected",
            "genre_topk": [],
            "clip_topk": [],
            "poem_topk": [],
            "poem_final_topk": [],
            "audio_aux_topk": [],
            "audio_aux_used": False,
            "final_reason": validation["message"],
            "delivery_explanation": "No delivery analysis was run because the text was rejected before inference.",
            "scope_note": validation["message"],
            "arousal_curve": [],
            "retrieval_rows": [],
        }

    if poem_id:
        poem_rows = get_poem_rows(poem_id)
        try:
            runtime_poem = build_runtime_poem_analysis(poem_rows, device, selected_clip_audio=(clip_row or poem_rows[0]).get("audio_filename", ""))
        except RuntimeError as exc:
            return {
                "is_corpus": True,
                "is_external_audio": False,
                "clip_only": False,
                "poem_id": poem_id,
                "origin_badge_html": pill("Corpus-verified poem", "good"),
                "final_genre": "unavailable",
                "genre_conf_badge": "",
                "final_poem_emotion": "unavailable",
                "final_emotion_conf_badge": "",
                "raw_poem_emotion": "unavailable",
                "raw_emotion_conf_badge": "",
                "arousal_display": "Not available",
                "dms_poem": None,
                "latency_ms": "n/a",
                "display_text": query_text,
                "english_summary": "Offline text models are not cached yet.",
                "whisper_text": "",
                "transcript_source": "Corrected transcript (corpus)",
                "show_diff": False,
                "diff_html": "",
                "input_gate_label": validation["message"] if validation["warn"] else "",
                "genre_topk": [],
                "clip_topk": [],
                "poem_topk": [],
                "poem_final_topk": [],
                "audio_aux_topk": [],
                "audio_aux_used": False,
                "final_reason": str(exc),
                "delivery_explanation": "Delivery analysis is unavailable until offline models are cached.",
                "scope_note": str(exc),
                "arousal_curve": [],
                "retrieval_rows": [],
            }
        selected_clip = clip_row or poem_rows[0]
        retriever = get_retriever_cached(device)
        hits = retriever.search_poems(query=get_full_corrected_text(poem_rows), top_k=top_k + 1, imagery_filter=imagery_filter or None, tag_boost=0.15) if retriever else []
        retrieval_rows_payload = retrieval_rows(
            hits,
            query_text=get_full_corrected_text(poem_rows),
            current_genre=poem_rows[0].get("genre_en", ""),
            current_emotion=runtime_poem.get("final", {}).get("emotion_poem_final"),
            exclude_poem_id=poem_id,
        )
        return {
            "is_corpus": True,
            "is_external_audio": False,
            "clip_only": False,
            "poem_id": poem_id,
            "origin_badge_html": pill("Corpus-verified poem", "good"),
            "final_genre": poem_rows[0].get("genre_en", "unknown"),
            "genre_conf_badge": "",
            "final_poem_emotion": runtime_poem.get("final", {}).get("emotion_poem_final"),
            "final_emotion_conf_badge": confidence_badge(float((runtime_poem.get("final", {}).get("emotion_poem_conditioned_top3", [{}])[0]).get("prob", 0.0))),
            "raw_poem_emotion": runtime_poem.get("raw_summary", {}).get("poem_emotion_raw_top1"),
            "raw_emotion_conf_badge": confidence_badge(float((runtime_poem.get("raw_summary", {}).get("poem_emotion_raw_topk", [{}])[0]).get("prob", 0.0))),
            "arousal_display": runtime_poem.get("arousal_poem") or "Not available",
            "dms_poem": runtime_poem.get("dms_poem"),
            "latency_ms": "text-only",
            "display_text": normalise_text(selected_clip.get("text_corrected") or query_text),
            "english_summary": build_poem_description(
                poem_id=poem_id,
                arabic_text=selected_clip.get("text_corrected") or query_text,
                genre=poem_rows[0].get("genre_en", ""),
                top_emotions=runtime_poem.get("final", {}).get("emotion_poem_conditioned_top3", []),
                imagery_tags=collect_poem_tags(poem_rows),
                translations=collect_translation_lines(poem_rows),
            ),
            "whisper_text": normalise_text(selected_clip.get("text_whisper", "")),
            "transcript_source": "Corrected transcript (corpus)",
            "show_diff": normalise_text(selected_clip.get("text_corrected", "")) != normalise_text(selected_clip.get("text_whisper", "")),
            "diff_html": diff_html(selected_clip.get("text_corrected", ""), selected_clip.get("text_whisper", "")),
            "input_gate_label": validation["message"] if validation["warn"] else "",
            "genre_topk": [],
            "clip_topk": runtime_poem.get("selected_clip_emotion", {}).get("topk", []),
            "poem_topk": runtime_poem.get("raw_summary", {}).get("poem_emotion_raw_topk", []),
            "poem_final_topk": runtime_poem.get("final", {}).get("emotion_poem_conditioned_top3", []),
            "audio_aux_topk": runtime_poem.get("audio_aux_topk", []),
            "audio_aux_used": runtime_poem.get("final", {}).get("audio_emotion_used_in_decision", False),
            "final_reason": runtime_poem.get("final", {}).get("emotion_poem_final_reason"),
            "delivery_explanation": runtime_poem.get("final", {}).get("delivery_nuance_tag", "Delivery information is not available."),
            "scope_note": "Matched a corpus poem; showing corrected text and poem-level aggregation.",
            "arousal_curve": runtime_poem.get("arousal_curve", []),
            "retrieval_rows": retrieval_rows_payload,
            "imagery_tags": collect_poem_tags(poem_rows),
        }

    resolved_text = query_text
    resolved_device, _ = resolve_device(device)
    try:
        genre_tok, genre_model = get_genre_assets(resolved_device)
        emo_tok, emo_model = get_emotion_assets(resolved_device)
    except RuntimeError as exc:
        return {
            "is_corpus": False,
            "is_external_audio": False,
            "clip_only": True,
            "poem_id": None,
            "origin_badge_html": "",
            "final_genre": "unavailable",
            "genre_conf_badge": "",
            "final_poem_emotion": "unavailable",
            "final_emotion_conf_badge": "",
            "raw_poem_emotion": "unavailable",
            "raw_emotion_conf_badge": "",
            "arousal_display": "Not available",
            "dms_poem": None,
            "latency_ms": "n/a",
            "display_text": resolved_text,
            "english_summary": "Offline text models are not cached yet.",
            "whisper_text": "",
            "transcript_source": "Text input",
            "show_diff": False,
            "diff_html": "",
            "input_gate_label": validation["message"] if validation["warn"] else "",
            "genre_topk": [],
            "clip_topk": [],
            "poem_topk": [],
            "poem_final_topk": [],
            "audio_aux_topk": [],
            "audio_aux_used": False,
            "final_reason": str(exc),
            "delivery_explanation": "Delivery analysis is not available for text-only input.",
            "scope_note": str(exc),
            "arousal_curve": [],
            "retrieval_rows": [],
        }
    if genre_model is None or emo_model is None:
        return {
            "is_corpus": False,
            "is_external_audio": False,
            "clip_only": True,
            "poem_id": None,
            "origin_badge_html": "",
            "final_genre": "unavailable",
            "genre_conf_badge": "",
            "final_poem_emotion": "unavailable",
            "final_emotion_conf_badge": "",
            "raw_poem_emotion": "unavailable",
            "raw_emotion_conf_badge": "",
            "arousal_display": "Not available",
            "dms_poem": None,
            "latency_ms": "n/a",
            "display_text": resolved_text,
            "english_summary": "The local text models are unavailable, so the app cannot generate a reliable poem preview.",
            "whisper_text": "",
            "transcript_source": "Text input",
            "show_diff": False,
            "diff_html": "",
            "input_gate_label": validation["message"] if validation["warn"] else "",
            "genre_topk": [],
            "clip_topk": [],
            "poem_topk": [],
            "poem_final_topk": [],
            "audio_aux_topk": [],
            "audio_aux_used": False,
            "final_reason": "Text models are not available in the current local checkpoint folder.",
            "delivery_explanation": "Delivery analysis is not available for text-only input.",
            "scope_note": "This input is not from the corpus; showing clip-level outputs only.",
            "arousal_curve": [],
            "retrieval_rows": [],
        }

    genre_label, _, genre_topk = predict_genre(resolved_text, genre_tok, genre_model, resolved_device)
    emo_logits, emo_probs = run_text_logits(resolved_text, emo_tok, emo_model, resolved_device)
    clip_topk = ranked_pairs(EMOTION_TEXT_CLASSES, emo_probs, 3)
    conditioned = apply_genre_prior(emo_probs, EMOTION_TEXT_CLASSES, genre_label, get_genre_priors(), 1.0)
    text_summary = build_poem_emotion_summary(
        poem_id="external_text",
        probs=emo_probs,
        probs_by_clip=[emo_probs],
        labels=EMOTION_TEXT_CLASSES,
        method="clip_only",
        clip_conf=[float(emo_probs.max())],
    )
    final = decide_final_emotion(
        text_summary=text_summary,
        conditioned_probs=conditioned,
        labels=EMOTION_TEXT_CLASSES,
        genre=genre_label,
        poem_arousal=None,
        profile="rare_merge_v1",
        strategy_name="genre_prior",
    )
    retriever = get_retriever_cached(resolved_device)
    hits = retriever.search_poems(query=resolved_text, top_k=top_k, imagery_filter=imagery_filter or None, tag_boost=0.15) if retriever else []
    retrieval_rows_payload = retrieval_rows(
        hits,
        query_text=resolved_text,
        current_genre=genre_label,
        current_emotion=final["emotion_poem_final"],
        exclude_poem_id=None,
    )
    return {
        "is_corpus": False,
        "is_external_audio": False,
        "clip_only": True,
        "poem_id": None,
        "origin_badge_html": "",
        "final_genre": genre_label,
        "genre_conf_badge": confidence_badge(float(genre_topk[0].prob if genre_topk else 0.0)),
        "final_poem_emotion": final["emotion_poem_final"],
        "final_emotion_conf_badge": confidence_badge(float((final["emotion_poem_conditioned_top3"][0]).get("prob", 0.0))),
        "raw_poem_emotion": clip_topk[0]["label"],
        "raw_emotion_conf_badge": confidence_badge(float(clip_topk[0]["prob"])),
        "arousal_display": "Not available for text-only input",
        "dms_poem": None,
        "latency_ms": "text-only",
        "display_text": resolved_text,
        "english_summary": build_poem_description(
            poem_id=None,
            arabic_text=resolved_text,
            genre=genre_label,
            top_emotions=final["emotion_poem_conditioned_top3"],
            imagery_tags=imagery_filter,
            translations=None,
        ),
        "whisper_text": "",
        "transcript_source": "Text input",
        "show_diff": False,
        "diff_html": "",
        "input_gate_label": validation["message"] if validation["warn"] else "",
        "genre_topk": [{"label": item.label, "prob": item.prob} for item in genre_topk],
        "clip_topk": clip_topk,
        "poem_topk": final["emotion_poem_raw_topk"],
        "poem_final_topk": final["emotion_poem_conditioned_top3"],
        "audio_aux_topk": [],
        "audio_aux_used": False,
        "final_reason": final["emotion_poem_final_reason"],
        "delivery_explanation": "Delivery analysis is only available for audio inputs.",
        "scope_note": "This input is not from the corpus; showing clip-level outputs only.",
        "arousal_curve": [],
        "retrieval_rows": retrieval_rows_payload,
    }


def build_poem_card_from_id(poem_id: str, top_k: int, imagery_filter: str, device: str) -> tuple[str, str, str, go.Figure, go.Figure, str, str]:
    poem_rows = get_poem_rows(poem_id)
    if not poem_rows:
        return empty_card("Select a poem to open its card.")
    try:
        runtime_poem = build_runtime_poem_analysis(poem_rows, device)
    except RuntimeError as exc:
        return empty_card(str(exc))
    selected_clip = poem_rows[0]
    try:
        retriever = get_retriever_cached(device)
    except RuntimeError as exc:
        return empty_card(str(exc))
    query_text = get_full_corrected_text(poem_rows)
    hits = retriever.search_poems(query=query_text, top_k=top_k + 1, imagery_filter=imagery_filter or None, tag_boost=0.15) if retriever else []
    card = {
        "is_corpus": True,
        "is_external_audio": False,
        "clip_only": False,
        "poem_id": poem_id,
        "origin_badge_html": pill("Corpus-verified poem", "good"),
        "final_genre": poem_rows[0].get("genre_en", "unknown"),
        "genre_conf_badge": "",
        "final_poem_emotion": runtime_poem.get("final", {}).get("emotion_poem_final"),
        "final_emotion_conf_badge": confidence_badge(float((runtime_poem.get("final", {}).get("emotion_poem_conditioned_top3", [{}])[0]).get("prob", 0.0))),
        "raw_poem_emotion": runtime_poem.get("raw_summary", {}).get("poem_emotion_raw_top1"),
        "raw_emotion_conf_badge": confidence_badge(float((runtime_poem.get("raw_summary", {}).get("poem_emotion_raw_topk", [{}])[0]).get("prob", 0.0))),
        "arousal_display": runtime_poem.get("arousal_poem") or "Not available",
        "dms_poem": runtime_poem.get("dms_poem"),
        "latency_ms": "corpus view",
        "display_text": normalise_text(selected_clip.get("text_corrected") or ""),
        "english_summary": build_poem_description(
            poem_id=poem_id,
            arabic_text=selected_clip.get("text_corrected") or "",
            genre=poem_rows[0].get("genre_en", ""),
            top_emotions=runtime_poem.get("final", {}).get("emotion_poem_conditioned_top3", []),
            imagery_tags=collect_poem_tags(poem_rows),
            translations=collect_translation_lines(poem_rows),
        ),
        "whisper_text": normalise_text(selected_clip.get("text_whisper") or ""),
        "transcript_source": "Corrected transcript (corpus)",
        "show_diff": normalise_text(selected_clip.get("text_corrected", "")) != normalise_text(selected_clip.get("text_whisper", "")),
        "diff_html": diff_html(selected_clip.get("text_corrected", ""), selected_clip.get("text_whisper", "")),
        "input_gate_label": "",
        "genre_topk": [],
        "clip_topk": runtime_poem.get("selected_clip_emotion", {}).get("topk", []),
        "poem_topk": runtime_poem.get("raw_summary", {}).get("poem_emotion_raw_topk", []),
        "poem_final_topk": runtime_poem.get("final", {}).get("emotion_poem_conditioned_top3", []),
        "audio_aux_topk": runtime_poem.get("audio_aux_topk", []),
        "audio_aux_used": runtime_poem.get("final", {}).get("audio_emotion_used_in_decision", False),
        "final_reason": runtime_poem.get("final", {}).get("emotion_poem_final_reason"),
        "delivery_explanation": runtime_poem.get("final", {}).get("delivery_nuance_tag", "Delivery information is not available."),
        "scope_note": "Corpus poem view: corrected text plus runtime poem-level fusion.",
        "arousal_curve": runtime_poem.get("arousal_curve", []),
        "retrieval_rows": retrieval_rows(
            hits,
            query_text=query_text,
            current_genre=poem_rows[0].get("genre_en", ""),
            current_emotion=runtime_poem.get("final", {}).get("emotion_poem_final"),
            exclude_poem_id=poem_id,
        ),
        "imagery_tags": collect_poem_tags(poem_rows),
    }
    return build_card_outputs(card)


def analyse_audio(
    audio_path: str | None,
    top_k: int,
    imagery_filter: str,
    device: str,
    use_lora: bool,
) -> tuple[str, str, str, go.Figure, go.Figure, str, str]:
    if audio_path is None:
        return empty_card("Upload or record an audio clip to begin.")
    resolved_device, device_note = resolve_device(device)
    try:
        result = run_demo(
            audio_path=Path(audio_path),
            top_k=int(top_k),
            imagery_filter=imagery_filter.strip() or None,
            out_path=PROJECT_ROOT / "outputs/demo_result.json",
            device=resolved_device,
            use_lora=use_lora,
        )
    except Exception as exc:
        logger.exception(exc)
        if "Models not cached. Run: just cache-models (one-time)." in str(exc):
            return empty_card(str(exc))
        return empty_card(f"Could not analyze this input locally: {exc}")

    clip_row = get_clip_row(audio_path)
    transcript_for_gate = normalise_text((clip_row or {}).get("text_corrected") or result.transcription)
    validation = validate_poem_text(transcript_for_gate)
    if validation["hard_reject"] and not clip_row:
        return empty_card(
            "ASR may have failed on this clip, so the app stopped before full presentation. Please try another clip or paste the poem text."
        )
    card = card_from_demo_result(result, clip_row, imagery_filter, resolved_device)
    if device_note:
        card["scope_note"] = f"{device_note} {card['scope_note']}"
    if validation["warn"]:
        card["input_gate_label"] = validation["message"]
    return build_card_outputs(card)


def analyse_text_mode(
    text: str,
    top_k: int,
    imagery_filter: str,
    device: str,
) -> tuple[str, str, str, go.Figure, go.Figure, str, str]:
    if not text.strip():
        return empty_card("Enter text to analyze.")
    resolved_device, device_note = resolve_device(device)
    card = card_from_text_input(text, top_k, imagery_filter, resolved_device)
    if device_note:
        card["scope_note"] = f"{device_note} {card['scope_note']}"
    return build_card_outputs(card)


def search_query_preview_html(
    query_text: str,
    transcript_source: str,
    english_summary: str,
    corpus_match: bool,
    note: str,
) -> str:
    return (
        "<div class='card'>"
        "<div class='section-title'>Search Query Preview</div>"
        f"{pill(transcript_source, 'soft')}"
        f"{pill('Corpus-verified poem', 'good') if corpus_match else ''}"
        f"{pill('External audio (not in corpus)', 'warn') if not corpus_match and query_text else ''}"
        f"<div class='transcript-block arabic-preview'>{html_text(query_text)}</div>"
        f"<div class='small'><strong>English preview:</strong> {html_text(english_summary)}</div>"
        f"<div class='small'>{html_text(note)}</div>"
        "</div>"
    )


def run_audio_search_query(
    audio_path: str | None,
    top_k: int,
    genre_filter: str,
    imagery_filter: str,
    poet_filter: str,
    device: str,
    use_lora: bool,
) -> dict[str, Any]:
    if audio_path is None:
        return {
            "ok": False,
            "error": "Upload or record audio to search the corpus.",
            "query_text": "",
            "transcript_source": "",
            "english_summary": "",
            "corpus_match": False,
            "note": "",
            "rows": [],
        }

    resolved_device, device_note = resolve_device(device)
    clip_row = get_clip_row(audio_path)
    audio_file = Path(audio_path)
    try:
        query_text, transcript_source = transcribe_audio_query(audio_file, resolved_device, use_lora=use_lora)
    except RuntimeError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "query_text": "",
            "transcript_source": "",
            "english_summary": "Offline ASR assets are not cached yet.",
            "corpus_match": False,
            "note": str(exc),
            "rows": [],
        }
    validation = validate_poem_text(query_text)
    if validation["hard_reject"] and not clip_row:
        return {
            "ok": False,
            "error": "ASR may have failed; please paste the poem text in Analyze mode or try another clip.",
            "query_text": query_text,
            "transcript_source": transcript_source,
            "english_summary": "The ASR output does not look like a usable Arabic poem.",
            "corpus_match": False,
            "note": "Search stopped before retrieval because the transcript failed the Arabic / poetry gate.",
            "rows": [],
        }

    retriever = get_retriever_cached(resolved_device)
    if retriever is None:
        return {
            "ok": False,
            "error": "Retrieval index is unavailable. Run `just evaluate-retrieval` first.",
            "query_text": query_text,
            "transcript_source": transcript_source,
            "english_summary": "Retrieval index is unavailable.",
            "corpus_match": bool(clip_row),
            "note": "Run `just evaluate-retrieval` to build the local index.",
            "rows": [],
        }

    fetch_k = max(int(top_k) * 8, int(top_k))
    hits = retriever.search_poems(
        query=query_text,
        top_k=fetch_k,
        genre_filter=genre_filter.strip() or None,
        imagery_filter=imagery_filter.strip() or None,
        tag_boost=0.15,
    )
    filtered_hits = []
    for hit in hits:
        if poet_filter.strip() and poet_filter.strip().lower() not in str(hit.get("poet_en", "")).lower():
            continue
        filtered_hits.append(hit)
        if len(filtered_hits) >= int(top_k):
            break

    search_rows = [
        {
            "poem_id": hit.get("source_poem", ""),
            "poet_en": hit.get("poet_en", ""),
            "score": float(hit.get("score", 0.0)),
            "genre_en": hit.get("genre_en", ""),
            "emotion_text": hit.get("emotion_text", ""),
            "n_clips": hit.get("n_clips", 1),
            "matched_tags": ", ".join(token_overlap(query_text, hit.get("imagery_tags_en"))) or "none",
            "snippet": ((hit.get("text_corrected", "") or "")[:90] + ("..." if len(hit.get("text_corrected", "") or "") > 90 else "")),
        }
        for hit in filtered_hits
    ]
    poem_rows = get_poem_rows(clip_row.get("source_poem")) if clip_row else []
    english_summary = build_poem_description(
        poem_id=clip_row.get("source_poem") if clip_row else None,
        arabic_text=query_text,
        genre=(clip_row or {}).get("genre_en", ""),
        top_emotions=[{"label": (clip_row or {}).get("emotion_text", ""), "prob": 1.0}] if clip_row else [],
        imagery_tags=collect_poem_tags(poem_rows) if poem_rows else (clip_row or {}).get("imagery_tags_en"),
        translations=collect_translation_lines(poem_rows) if poem_rows else [normalise_text((clip_row or {}).get("translation_en") or "")],
    )
    note = "Using the corpus-corrected transcript for retrieval." if clip_row else "Using the ASR transcript as the retrieval query."
    if validation["warn"]:
        note = f"{note} {validation['message']}"
    if device_note:
        note = f"{device_note} {note}"

    return {
        "ok": True,
        "error": "",
        "query_text": query_text,
        "transcript_source": transcript_source,
        "english_summary": english_summary,
        "corpus_match": bool(clip_row),
        "note": note,
        "rows": search_rows,
    }


def search_audio_candidates(
    audio_path: str | None,
    top_k: int,
    genre_filter: str,
    imagery_filter: str,
    poet_filter: str,
    device: str,
    use_lora: bool,
) -> tuple[str, str, Any]:
    try:
        payload = run_audio_search_query(
            audio_path=audio_path,
            top_k=top_k,
            genre_filter=genre_filter,
            imagery_filter=imagery_filter,
            poet_filter=poet_filter,
            device=device,
            use_lora=use_lora,
        )
    except Exception as exc:
        logger.exception(exc)
        return (
            "<div class='card'><div class='small'>Could not read this audio locally.</div></div>",
            f"<div class='card'><div class='small'>{html_text(str(exc))}</div></div>",
            gr.Dropdown(choices=[], value=None),
        )

    if not payload["ok"]:
        return (
            search_query_preview_html(
                query_text=payload["query_text"],
                transcript_source=payload["transcript_source"],
                english_summary=payload["english_summary"] or "The query could not be prepared.",
                corpus_match=bool(payload["corpus_match"]),
                note=payload["note"] or payload["error"],
            ),
            f"<div class='card'><div class='small'>{html_text(payload['error'])}</div></div>",
            gr.Dropdown(choices=[], value=None),
        )
    choices = [row["poem_id"] for row in payload["rows"] if row["poem_id"]]
    return (
        search_query_preview_html(
            query_text=payload["query_text"],
            transcript_source=payload["transcript_source"],
            english_summary=payload["english_summary"],
            corpus_match=bool(payload["corpus_match"]),
            note=payload["note"],
        ),
        search_results_html(payload["rows"]),
        gr.Dropdown(choices=choices, value=choices[0] if choices else None),
    )


def search_corpus(
    query: str,
    top_k: int,
    genre_filter: str,
    emotion_filter: str,
    imagery_filter: str,
    device: str,
) -> tuple[str, Any]:
    if not query.strip():
        return (
            "<div class='card'><div class='small'>Enter a query to search the corpus.</div></div>",
            gr.Dropdown(choices=[], value=None),
        )
    resolved_device, _ = resolve_device(device)
    retriever = get_retriever_cached(resolved_device)
    if retriever is None:
        return (
            "<div class='card'><div class='small'>Retrieval index is unavailable. Run `just evaluate-retrieval` first.</div></div>",
            gr.Dropdown(choices=[], value=None),
        )
    hits = retriever.search_poems(
        query=query.strip(),
        top_k=int(top_k),
        genre_filter=genre_filter.strip() or None,
        emotion_filter=emotion_filter.strip() or None,
        imagery_filter=imagery_filter.strip() or None,
        tag_boost=0.15,
    )
    search_rows = [
        {
            "poem_id": hit.get("source_poem", ""),
            "poet_en": hit.get("poet_en", ""),
            "score": float(hit.get("score", 0.0)),
            "genre_en": hit.get("genre_en", ""),
            "emotion_text": hit.get("emotion_text", ""),
            "snippet": (hit.get("text_corrected", "") or "")[:90],
        }
        for hit in hits
    ]
    choices = [row["poem_id"] for row in search_rows if row["poem_id"]]
    return search_results_html(search_rows), gr.Dropdown(choices=choices, value=choices[0] if choices else None)


@lru_cache(maxsize=1)
def load_map_points() -> pd.DataFrame:
    if MAP_CACHE_PATH.exists():
        return pd.DataFrame(json.loads(MAP_CACHE_PATH.read_text(encoding="utf-8")))

    corpus = corpus_indexes()["by_poem_id"]
    embeddings: dict[str, list[np.ndarray]] = defaultdict(list)
    for split in ("train", "val", "test"):
        emb_path = PROJECT_ROOT / f"data/processed/embeddings/{split}.pkl"
        if not emb_path.exists():
            continue
        import pickle

        with emb_path.open("rb") as handle:
            emb_dict = pickle.load(handle)
        for poem_id, items in emb_dict.items():
            for item in items:
                embeddings[poem_id].append(np.array(item["cls"], dtype=np.float32))

    points: list[dict[str, Any]] = []
    vectors: list[np.ndarray] = []
    poem_ids: list[str] = []
    for poem_id, poem_rows in corpus.items():
        if poem_id not in embeddings or not embeddings[poem_id]:
            continue
        poem_ids.append(poem_id)
        vectors.append(np.mean(np.stack(embeddings[poem_id], axis=0), axis=0))
        dominant_emotion = Counter(row.get("emotion_text", "") for row in poem_rows).most_common(1)[0][0]
        points.append(
            {
                "poem_id": poem_id,
                "poem_title": poem_rows[0].get("poem_title", ""),
                "poet_en": poem_rows[0].get("poet_en", ""),
                "genre_en": poem_rows[0].get("genre_en", ""),
                "emotion_text": dominant_emotion,
                "arousal": emotion_to_arousal(dominant_emotion) or "Unknown",
                "n_clips": len(poem_rows),
            }
        )

    X = StandardScaler().fit_transform(np.stack(vectors))
    perplexity = min(30, max(len(points) - 1, 2))
    Z = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        learning_rate="auto",
        init="pca",
        max_iter=1000,
    ).fit_transform(X)
    for idx, point in enumerate(points):
        point["x"] = float(Z[idx, 0])
        point["y"] = float(Z[idx, 1])
    MAP_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    MAP_CACHE_PATH.write_text(json.dumps(points, ensure_ascii=False), encoding="utf-8")
    return pd.DataFrame(points)


def build_map_figure(color_by: str, search_term: str) -> tuple[go.Figure, list[dict[str, Any]], Any]:
    df = load_map_points().copy()
    if search_term.strip():
        needle = search_term.strip().lower()
        df = df[
            df["poem_id"].str.lower().str.contains(needle)
            | df["poet_en"].str.lower().str.contains(needle)
            | df["poem_title"].fillna("").str.lower().str.contains(needle)
        ]
    color_field = {
        "Genre": "genre_en",
        "Emotion": "emotion_text",
        "Poet": "poet_en",
        "Arousal": "arousal",
    }.get(color_by, "genre_en")
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color=color_field,
        hover_data=["poem_id", "poet_en", "poem_title", "genre_en", "emotion_text", "n_clips"],
        custom_data=["poem_id"],
    )
    fig.update_traces(marker={"size": 12, "line": {"width": 0.5, "color": CARD_BG}})
    fig.update_layout(
        paper_bgcolor=PLOT_BG,
        plot_bgcolor=PLOT_BG,
        font={"family": "Aeonik Mono, Menlo, monospace", "color": TEXT},
        margin={"l": 12, "r": 12, "t": 30, "b": 12},
        height=460,
        showlegend=True,
        legend_title_text=color_by,
        xaxis={"visible": False},
        yaxis={"visible": False},
        title=f"Poetry Map colored by {color_by}",
    )
    map_points = df.to_dict("records")
    return fig, map_points, map_choice_update(map_points)


def map_choice_update(map_points: list[dict[str, Any]]) -> Any:
    choices = [point["poem_id"] for point in map_points if point.get("poem_id")]
    return gr.Dropdown(choices=choices, value=choices[0] if choices else None)


def open_map_poem(
    poem_id: str,
    top_k: int,
    imagery_filter: str,
    device: str,
) -> tuple[str, str, str, go.Figure, go.Figure, str, str]:
    if not poem_id:
        return empty_card("Choose a poem from the filtered map list to open its card.")
    resolved_device, _ = resolve_device(device)
    return build_poem_card_from_id(poem_id, top_k, imagery_filter, resolved_device)


def show_analyze_mode() -> tuple[Any, Any, Any]:
    return (
        gr.update(visible=False),
        gr.update(visible=True),
        gr.update(visible=False),
    )


def show_search_mode() -> tuple[Any, Any, Any]:
    return (
        gr.update(visible=False),
        gr.update(visible=False),
        gr.update(visible=True),
    )


def show_start_screen() -> tuple[Any, Any, Any]:
    return (
        gr.update(visible=True),
        gr.update(visible=False),
        gr.update(visible=False),
    )


APP_CSS = """
:root {
  --color-bg: rgb(244, 239, 234);
  --color-surface: #ffffff;
  --color-surface-up: #ede8e2;
  --color-border: #2b2b2b;
  --color-border-light: #c8bfb5;
  --color-text: rgb(56, 56, 56);
  --color-text-sec: #7a6e65;
  --color-text-muted: #a8998e;
  --color-accent: #c17d3c;
  --color-accent-soft: #f5e6d3;
  --color-green: #3a7d44;
  --color-green-soft: #d4edd8;
  --color-amber: #c17d3c;
  --color-amber-soft: #f5e6d3;
  --color-red: #b5363a;
  --color-red-soft: #f5d4d5;
  --color-cyan: #2b6e8a;
  --color-cyan-soft: #d4eaf5;
}
body, .gradio-container {
  background: var(--color-bg) !important;
  color: var(--color-text) !important;
  min-height: 100vh;
  font-family: "Aeonik Mono", ui-monospace, "SFMono-Regular", monospace !important;
}
* {
  box-sizing: border-box;
  font-family: "Aeonik Mono", ui-monospace, "SFMono-Regular", monospace !important;
}
.gradio-container {
  max-width: 1320px !important;
}
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-thumb { background: rgb(136, 136, 136); }
::-webkit-scrollbar-track { background: rgb(241, 241, 241); }
a { color: var(--color-accent); }

.gradio-container .block,
.gradio-container .gr-box,
.gradio-container .gr-panel,
.gradio-container .gr-form,
.gradio-container .gr-group,
.gradio-container .gr-accordion,
.gradio-container .gradio-group {
  border-radius: 0 !important;
  border: none !important;
  box-shadow: none !important;
  background: transparent !important;
}

.gradio-container input,
.gradio-container textarea,
.gradio-container select,
.gradio-container .wrap,
.gradio-container .gr-dropdown,
.gradio-container [data-testid="textbox"],
.gradio-container [data-testid="dropdown"],
.gradio-container [data-testid="number-input"] {
  background: var(--color-surface) !important;
  color: var(--color-text) !important;
  border: 2.5px solid var(--color-border) !important;
  border-radius: 0 !important;
  font-size: 13px !important;
  box-shadow: none !important;
}
.gradio-container input:focus,
.gradio-container textarea:focus,
.gradio-container select:focus {
  border-color: var(--color-accent) !important;
  box-shadow: 3px 3px 0 var(--color-accent) !important;
}

.gradio-container .gr-button,
.gradio-container button {
  background: var(--color-surface) !important;
  color: var(--color-text) !important;
  border: 2.5px solid var(--color-border) !important;
  border-radius: 0 !important;
  box-shadow: 3px 3px 0 var(--color-border) !important;
  font-size: 12px !important;
  font-weight: 700 !important;
  letter-spacing: 0.06em !important;
  text-transform: uppercase !important;
  transition: transform 0.08s, box-shadow 0.08s, background 0.08s !important;
}
.gradio-container .gr-button:hover,
.gradio-container button:hover {
  background: var(--color-accent-soft) !important;
  transform: translate(-1px, -1px);
  box-shadow: 5px 5px 0 var(--color-border) !important;
}
.gradio-container .gr-button:active,
.gradio-container button:active {
  transform: translate(2px, 2px);
  box-shadow: 1px 1px 0 var(--color-border) !important;
}

.summary-card, .card {
  background: var(--color-surface);
  border: 2.5px solid var(--color-border);
  border-radius: 0;
  padding: 20px;
  box-shadow: 4px 4px 0 var(--color-border);
}
.hero-card {
  background: var(--color-green) !important;
  color: #ffffff !important;
  border-color: var(--color-green) !important;
  box-shadow: 4px 4px 0 var(--color-border) !important;
}
.hero-card .eyebrow,
.hero-card .metric-value,
.hero-card .small {
  color: #ffffff !important;
}
.hero-card .eyebrow {
  opacity: 0.9;
}
.summary-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 14px;
  margin-top: 12px;
}
.metric-label,
.eyebrow,
.section-title,
.mini-title {
  font-size: 10px;
  font-weight: 700;
  color: var(--color-text-muted);
  letter-spacing: 0.14em;
  text-transform: uppercase;
}
.eyebrow,
.section-title {
  margin-bottom: 10px;
}
.metric-value {
  font-size: 14px;
  line-height: 1.5;
  margin-top: 4px;
}
.pill, .confidence {
  display: inline-block;
  padding: 3px 10px;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin: 2px 6px 6px 0;
  border: 1.5px solid currentColor;
  border-radius: 0;
  background: var(--color-surface-up);
}
.transcript-block, .mono-block, .diff-box {
  margin-top: 12px;
  padding: 12px 16px;
  border: 2px solid var(--color-border-light);
  border-radius: 0;
  background: var(--color-bg);
  white-space: pre-wrap;
  line-height: 1.8;
}
.small {
  font-size: 12px;
  line-height: 1.7;
  color: var(--color-text-sec);
  margin-top: 8px;
}
.details {
  margin-top: 12px;
}
.details summary {
  cursor: pointer;
  font-size: 10px;
  font-weight: 700;
  color: var(--color-text-muted);
  text-transform: uppercase;
  letter-spacing: 0.14em;
}
.compact-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
  margin-top: 12px;
}
.compact-table th {
  background: var(--color-surface-up);
  color: var(--color-text-muted);
  font-size: 10px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  padding: 10px 12px;
  text-align: left;
  border-bottom: 2.5px solid var(--color-border);
}
.compact-table td {
  padding: 10px 12px;
  border-bottom: 1px solid var(--color-border-light);
  color: var(--color-text);
  vertical-align: top;
}
.compact-table tr:hover td {
  background: var(--color-accent-soft);
}
.diff-add {
  color: var(--color-green);
  background: var(--color-green-soft);
  padding: 0 3px;
  font-weight: 700;
  border-bottom: 2px solid var(--color-green);
}
.diff-del {
  color: var(--color-red);
  text-decoration: line-through;
  background: var(--color-red-soft);
  padding: 0 3px;
}
.arabic-preview {
  direction: rtl;
  text-align: right;
  font-family: "Amiri", "Scheherazade New", serif !important;
  font-size: 22px;
  line-height: 2.2;
}

.start-shell {
  background: transparent;
  border: none;
  padding: 8px 0 4px;
  box-shadow: none;
}
#start-copy,
#start-copy > div,
#start-copy .html-container,
#start-copy .prose {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
  margin: 0 !important;
}
.mode-copy h2 {
  font-size: 28px;
  line-height: 1.15;
  font-weight: 400;
  margin: 6px 0 12px;
}
.mode-copy p {
  font-size: 13px;
  line-height: 1.75;
  color: var(--color-text-sec);
}
.mode-card {
  background: transparent;
  border: none;
  padding: 4px 0 0;
  box-shadow: none;
  min-height: 250px;
}
#analyze-mode-card,
#search-mode-card,
#analyze-mode-card > div,
#search-mode-card > div,
#analyze-mode-card .gr-block,
#search-mode-card .gr-block,
#analyze-mode-card .gradio-group,
#search-mode-card .gradio-group {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
}
.mode-card h3 {
  font-size: 15px;
  line-height: 1.45;
  margin-top: 14px;
}
.mode-card p {
  font-size: 12px;
  line-height: 1.75;
  color: var(--color-text-sec);
  margin-top: 10px;
}
.mode-tag {
  display: inline-block;
  padding: 3px 10px;
  font-size: 10px;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  margin-top: 10px;
  border: 1.5px solid currentColor;
  color: var(--color-cyan);
  background: var(--color-cyan-soft);
}

.start-icon,
.start-icon *,
.floating-icon,
.floating-icon *,
.icon-container,
.icon-container *,
img[src*="/icons/"] {
  background: transparent !important;
  background-color: transparent !important;
  border: none !important;
  border-width: 0 !important;
  border-style: none !important;
  border-color: transparent !important;
  box-shadow: none !important;
  outline: none !important;
  outline-width: 0 !important;
  filter: none !important;
}
.start-icon img {
  object-fit: contain !important;
}
.start-icon-img {
  display: block;
  width: 112px;
  max-width: 100%;
  height: 112px;
  object-fit: contain;
  pointer-events: none !important;
  user-select: none !important;
}
.icon-container {
  display: flex;
  align-items: center;
  justify-content: center;
  margin-bottom: 8px;
  pointer-events: none !important;
}
.start-icon,
.start-icon > div,
.start-icon [data-testid="image"] {
  padding: 0 !important;
  margin: 0 0 8px 0 !important;
}
.icon-fallback {
  width: 96px;
  height: 96px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 28px;
  font-weight: 700;
  color: var(--color-text);
  background: var(--color-surface-up);
}
.icon-fallback.analyze {
  background: var(--color-accent-soft);
}
.icon-fallback.search {
  background: var(--color-cyan-soft);
}

.mode-cta button,
.back-cta button,
.mode-cta-analyze button,
.mode-cta-search button {
  min-height: 56px !important;
  text-align: left !important;
  justify-content: flex-start !important;
  padding: 12px 18px !important;
}
.mode-cta-analyze button {
  background: var(--color-accent-soft) !important;
  color: var(--color-accent) !important;
  border-color: var(--color-accent) !important;
  box-shadow: 3px 3px 0 var(--color-accent) !important;
}
.mode-cta-analyze button:hover {
  background: var(--color-accent-soft) !important;
  box-shadow: 5px 5px 0 var(--color-accent) !important;
}
.mode-cta-search button {
  background: var(--color-cyan-soft) !important;
  color: var(--color-cyan) !important;
  border-color: var(--color-cyan) !important;
  box-shadow: 3px 3px 0 var(--color-cyan) !important;
}
.mode-cta-search button:hover {
  background: var(--color-cyan-soft) !important;
  box-shadow: 5px 5px 0 var(--color-cyan) !important;
}
.back-cta button {
  background: transparent !important;
  color: var(--color-text-sec) !important;
  box-shadow: none !important;
  border-color: var(--color-border-light) !important;
}
.back-cta button:hover {
  background: var(--color-surface-up) !important;
  box-shadow: none !important;
}
footer,
.footer,
[data-testid="footer"],
.built-with-gradio,
.gradio-container footer,
.gradio-container .footer,
.gradio-container [data-testid="footer"] {
  display: none !important;
  border: none !important;
  box-shadow: none !important;
  margin: 0 !important;
  padding: 0 !important;
  height: 0 !important;
  min-height: 0 !important;
  overflow: hidden !important;
}

.gradio-container .gr-tab-nav {
  display: flex;
  gap: 0;
  border-bottom: 3px solid var(--color-border) !important;
  background: var(--color-bg) !important;
}
.gradio-container .gr-tab-nav button {
  background: var(--color-surface) !important;
  color: var(--color-text-sec) !important;
  border: 2.5px solid var(--color-border) !important;
  border-bottom: none !important;
  border-radius: 0 !important;
  margin-right: 4px !important;
  padding: 12px 22px !important;
  box-shadow: none !important;
}
.gradio-container .gr-tab-nav button.selected {
  background: var(--color-accent) !important;
  color: #fff !important;
}
"""


# ── Imagery Explorer ──────────────────────────────────────────────────────────

import math as _math


@lru_cache(maxsize=1)
def _load_imagery_corpus() -> tuple[list[dict], list[str], list[str]]:
    """Return (records, sorted_poets, sorted_genres). Cached after first call."""
    master = PROJECT_ROOT / "data" / "processed" / "master_dataset.jsonl"
    records: list[dict] = []
    with open(master, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            tags = [t.strip().lower() for t in d.get("imagery_tags_en", "").split(",") if t.strip()]
            genre = d.get("genre_en", "").split("(")[0].strip()
            poet = d.get("poet_en", "unknown")
            if tags:
                records.append({"poet": poet, "genre": genre, "tags": tags})
    poets = sorted({r["poet"] for r in records})
    genres = sorted({r["genre"] for r in records})
    return records, poets, genres


def build_poet_fingerprint_plot(poet_name: str) -> go.Figure:
    """Horizontal bar chart of the 12 most TF-IDF-distinctive imagery tags for a poet."""
    records, _, _ = _load_imagery_corpus()

    poet_tags: dict[str, Counter] = defaultdict(Counter)
    for r in records:
        poet_tags[r["poet"]].update(r["tags"])

    if poet_name not in poet_tags:
        fig = go.Figure()
        fig.add_annotation(text="Poet not found", xref="paper", yref="paper",
                           x=0.5, y=0.5, showarrow=False, font=dict(size=16))
        return fig

    N = len(poet_tags)
    # IDF: how many poets use each tag
    tag_df: Counter = Counter()
    for p_tags in poet_tags.values():
        for tag in p_tags:
            tag_df[tag] += 1

    poet_total = sum(poet_tags[poet_name].values())
    scores: dict[str, float] = {}
    for tag, count in poet_tags[poet_name].items():
        tf = count / max(poet_total, 1)
        idf = _math.log((N + 1) / (tag_df[tag] + 1))
        scores[tag] = tf * idf

    top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:12]
    tags_plot = [t[0].title() for t in reversed(top)]
    vals_plot = [t[1] for t in reversed(top)]
    counts_plot = [poet_tags[poet_name][t[0]] for t in reversed(top)]

    fig = go.Figure(go.Bar(
        x=vals_plot,
        y=tags_plot,
        orientation="h",
        marker_color="#3a7d44",
        text=[f"×{c}" for c in counts_plot],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>TF-IDF: %{x:.4f}<br>Clip count: %{text}<extra></extra>",
    ))
    fig.update_layout(
        title=dict(text=f"✦ Imagery Fingerprint — {poet_name}", font=dict(size=16, color=TEXT)),
        xaxis_title="TF-IDF distinctiveness score",
        yaxis_title="",
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=CARD_BG,
        font=dict(color=TEXT, size=13),
        margin=dict(l=180, r=100, t=70, b=40),
        height=440,
        xaxis=dict(gridcolor=BORDER),
    )
    return fig


def build_genre_heatmap_plot() -> go.Figure:
    """Heatmap: genres (rows) × top-25 imagery tags (cols), normalised by clips per genre."""
    records, _, genres = _load_imagery_corpus()

    genre_tags: dict[str, Counter] = defaultdict(Counter)
    genre_clip_n: dict[str, int] = defaultdict(int)
    all_tags: Counter = Counter()

    for r in records:
        genre_tags[r["genre"]].update(r["tags"])
        genre_clip_n[r["genre"]] += 1
        all_tags.update(r["tags"])

    top_tags = [t.title() for t, _ in all_tags.most_common(25)]
    top_tags_lower = [t.lower() for t in top_tags]

    matrix: list[list[float]] = []
    text_matrix: list[list[str]] = []
    for genre in genres:
        n = max(genre_clip_n[genre], 1)
        row = []
        trow = []
        for tag_lower in top_tags_lower:
            pct = round(genre_tags[genre].get(tag_lower, 0) / n * 100, 1)
            row.append(pct)
            trow.append(f"{pct}%" if pct > 0 else "")
        matrix.append(row)
        text_matrix.append(trow)

    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=top_tags,
        y=genres,
        colorscale="YlOrBr",
        text=text_matrix,
        texttemplate="%{text}",
        textfont={"size": 10},
        hovertemplate="<b>%{y}</b><br>Tag: %{x}<br>Frequency: %{text}<extra></extra>",
        colorbar=dict(title="% clips", thickness=14),
    ))
    fig.update_layout(
        title=dict(
            text="Imagery Tags by Genre — Top 25 tags (% of clips per genre)",
            font=dict(size=15, color=TEXT),
        ),
        xaxis=dict(tickangle=-40, tickfont=dict(size=11), gridcolor=BORDER),
        yaxis=dict(tickfont=dict(size=12)),
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=CARD_BG,
        font=dict(color=TEXT),
        height=520,
        margin=dict(l=140, r=60, t=70, b=130),
    )
    return fig


def build_semiotic_html(tags: list[str]) -> str:
    """Return a styled HTML card unpacking the semiotic meaning of each imagery tag."""
    from data.semiotics import lookup_semiotics  # lazy import to keep module-level clean

    if not tags:
        return (
            "<div class='card'>"
            "<div class='section-title'>🔣 Semiotic Reading</div>"
            "<div class='small'>No imagery tags were detected for this poem — "
            "semiotic analysis requires at least one imagery tag.</div></div>"
        )

    entries: list[tuple[str, dict]] = []
    seen: set[str] = set()
    for raw_tag in tags:
        tag = raw_tag.strip()
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        info = lookup_semiotics(key)
        if info:
            entries.append((tag, info))

    unmatched = [t.strip() for t in tags if t.strip() and t.strip().lower() not in {k for k, _ in entries}]

    if not entries:
        tag_list = ", ".join(t.strip() for t in tags if t.strip())
        return (
            "<div class='card'><div class='section-title'>🔣 Semiotic Reading</div>"
            f"<div class='small'>Imagery tags detected: <em>{html_text(tag_list)}</em>.<br>"
            "None of these tags have entries in the current semiotic lexicon — "
            "the lexicon covers the 36 most common Gulf poetic symbols.</div></div>"
        )

    cards_html = ""
    for tag, info in entries:
        cards_html += (
            f"<div style='background:{CARD_BG};border:1px solid {BORDER};"
            "border-radius:8px;padding:0.75rem 1rem;margin-bottom:0.6rem;'>"
            "<div style='display:flex;align-items:center;gap:0.5rem;margin-bottom:0.4rem;'>"
            f"<span style='font-size:1rem;font-weight:700;color:{TEXT};'>{html_text(tag)}</span>"
            f"<span style='font-size:0.72rem;color:{MUTED};background:#ede8e3;"
            f"border-radius:4px;padding:1px 7px;'>{html_text(info.get('category', ''))}</span>"
            f"<span style='font-size:0.88rem;color:{ACCENT};font-style:italic;"
            f"font-family:serif;margin-right:auto;'>{html_text(info.get('arabic', ''))}</span>"
            "</div>"
            f"<div style='font-size:0.8rem;font-weight:600;color:{ACCENT};margin-bottom:0.3rem;'>"
            f"&#8594; {html_text(info.get('signified', ''))}</div>"
            f"<div style='font-size:0.78rem;color:{MUTED};line-height:1.55;'>"
            f"{html_text(info.get('connotation', ''))}</div>"
            "</div>"
        )

    unmatched_note = ""
    if unmatched:
        unmatched_note = (
            f"<div class='small' style='margin-top:0.5rem;color:{MUTED};'>"
            f"Additional tags (no lexicon entry): {html_text(', '.join(unmatched))}</div>"
        )

    dukhon_img = render_inline_icon(_UI_ICON_MAP.get("semiotics"), "Dukhon – Semiotic Reading", size=20)
    return (
        "<div class='card'>"
        f"<div class='section-title'>{dukhon_img}Semiotic Reading</div>"
        f"<div class='small' style='margin-bottom:0.75rem;'>"
        "Each imagery tag unpacked through Khaleeji/Gulf cultural semiotics — "
        "what it <em>signifies</em> (the concept it encodes) and what it "
        "<em>connotes</em> (how it resonates in Nabati poetic tradition)."
        "</div>"
        f"{cards_html}"
        f"{unmatched_note}"
        "</div>"
    )


def build_result_panel(title: str) -> tuple[gr.HTML, gr.HTML, gr.HTML, gr.Plot, gr.Plot, gr.HTML, gr.HTML, gr.HTML]:
    with gr.Column(scale=2):
        summary = gr.HTML(label=f"{title} summary")
        transcript = gr.HTML(label=f"{title} transcript")
        details = gr.HTML(label=f"{title} details")
        with gr.Row():
            emotion_plot = gr.Plot(label="Poem Emotion Mix")
            arousal_plot = gr.Plot(label="Delivery Curve")
        retrieval = gr.HTML(label=f"{title} retrieval")
        notes = gr.HTML(label=f"{title} notes")
        with gr.Accordion("🔣 Semiotic Reading", open=False):
            semiotic_out = gr.HTML(label=f"{title} semiotics")
    return summary, transcript, details, emotion_plot, arousal_plot, retrieval, notes, semiotic_out


def build_ui(default_device: str = "cpu") -> gr.Blocks:
    global _UI_ICON_MAP
    initial_map, initial_points, _ = build_map_figure("Genre", "")
    icon_map = discover_icons()
    _UI_ICON_MAP.update(icon_map)
    with gr.Blocks(title="Nabat-AI") as ui:
        gr.HTML(f"<style>{APP_CSS}</style>")
        gr.HTML(
            """
            <div class="summary-card hero-card" style="background:#3a7d44;color:#fff;border-color:#3a7d44;">
              <div class="eyebrow" style="color:#fff;">Nabat-AI Demo</div>
              <div class="metric-value" style="color:#fff;">A local, offline Nabati poetry demo for analysis and corpus search.</div>
              <div class="small" style="color:#fff;">The app is built for an English-speaking reader: concise summaries, confidence badges, and corrected corpus text when available.</div>
            </div>
            """
        )
        setup_notice = offline_setup_notice_html()
        if setup_notice:
            gr.HTML(setup_notice)

        with gr.Column(visible=True) as start_screen:
            gr.HTML(
                """
                <div class="start-shell">
                  <div class="mode-copy">
                    <div class="eyebrow">Start Here</div>
                    <h2>Choose one path.</h2>
                    <p>Pick analysis when you want an interpretable reading of the poem. Pick search when you already have audio and mainly want the closest corpus match.</p>
                  </div>
                </div>
                """,
                elem_id="start-copy",
            )
            with gr.Row(equal_height=True):
                with gr.Column(elem_classes=["mode-card"], elem_id="analyze-mode-card"):
                    gr.HTML(render_start_icon(icon_map.get("analyze"), "analyze", "~", "Analyze mode icon"))
                    gr.HTML(
                        """
                        <div class="mode-tag">Analysis</div>
                        <h3>Analyze a poem / clip</h3>
                        <p>Upload audio, record audio, or paste Arabic poem text. This path explains genre, poem emotion, delivery, and similar poems in English-friendly language.</p>
                        """
                    )
                    analyze_start = gr.Button(
                        "Open Analyze Mode",
                        elem_classes=["mode-cta", "mode-cta-analyze"],
                    )
                with gr.Column(elem_classes=["mode-card"], elem_id="search-mode-card"):
                    gr.HTML(render_start_icon(icon_map.get("search"), "search", "?", "Search mode icon"))
                    gr.HTML(
                        """
                        <div class="mode-tag">Retrieval</div>
                        <h3>Search for a poem in the corpus</h3>
                        <p>Upload or record audio to find the most similar poem in the corpus. This path emphasizes ranking, similarity, and candidate review instead of classification.</p>
                        """
                    )
                    search_start = gr.Button(
                        "Open Search Mode",
                        elem_classes=["mode-cta", "mode-cta-search"],
                    )

        with gr.Column(visible=False) as analyze_group:
            analyze_back = gr.Button("Back to Start Screen", elem_classes=["back-cta"])
            gr.HTML(
                f"""
                <div class="card">
                  <div class="section-title">{render_inline_icon(icon_map.get("analyze"), "Dallah – Analyze mode")}Analyze a poem / clip</div>
                  <div class="small">Use this when you want a full reading of a poem: genre, poem-level emotions, delivery, and similar poems.</div>
                </div>
                """
            )
            with gr.Row():
                analyze_top_k = gr.Slider(1, 10, value=5, step=1, label="Retrieval depth")
                analyze_imagery = gr.Textbox(label="Imagery tag filter", placeholder="heart, desert, night")
                analyze_device = gr.Radio(choices=["cpu", "mps", "cuda"], value=default_device, label="Compute device")

            with gr.Tabs():
                with gr.Tab("Upload audio"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            upload_audio = gr.Audio(type="filepath", sources=["upload"], label="Upload clip")
                            upload_lora = gr.Checkbox(label="Use LoRA (experimental)", value=False)
                            upload_btn = gr.Button("Analyze upload", variant="primary")
                        upload_outputs = build_result_panel("Upload")
                    upload_btn.click(
                        fn=analyse_audio,
                        inputs=[upload_audio, analyze_top_k, analyze_imagery, analyze_device, upload_lora],
                        outputs=list(upload_outputs),
                    )

                with gr.Tab("Record audio"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            record_audio = gr.Audio(type="filepath", sources=["microphone"], label="Record clip")
                            record_lora = gr.Checkbox(label="Use LoRA (experimental)", value=False)
                            record_btn = gr.Button("Analyze recording", variant="primary")
                        record_outputs = build_result_panel("Record")
                    record_btn.click(
                        fn=analyse_audio,
                        inputs=[record_audio, analyze_top_k, analyze_imagery, analyze_device, record_lora],
                        outputs=list(record_outputs),
                    )

                with gr.Tab("Paste Arabic poem text"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            text_in = gr.Textbox(
                                label="Arabic poem text",
                                lines=10,
                                placeholder="Paste a verse, a full poem, or a known poem_id.",
                            )
                            text_btn = gr.Button("Analyze text", variant="primary")
                        text_outputs = build_result_panel("Text")
                    text_btn.click(
                        fn=analyse_text_mode,
                        inputs=[text_in, analyze_top_k, analyze_imagery, analyze_device],
                        outputs=list(text_outputs),
                    )

            with gr.Accordion("Poetry Map", open=False):
                gr.HTML(
                    f"""
                    <div class="card">
                      <div class="section-title">{render_inline_icon(icon_map.get("map"), "Dihn Oud – Poetry Map")}What am I looking at?</div>
                      <div class="small">Each point is a poem-level embedding projected with t-SNE. Use the color switch to inspect how genre, dominant emotion, poet, or arousal proxy organize the corpus. Search narrows the visible set, then use the selector to open a poem card.</div>
                    </div>
                    """
                )
                with gr.Row():
                    map_color = gr.Dropdown(choices=["Genre", "Emotion", "Poet", "Arousal"], value="Genre", label="Color by")
                    map_search = gr.Textbox(label="Search poem_id / poet", placeholder="e.g. poem0070_ku or Saad")
                    map_refresh = gr.Button("Refresh map")
                map_plot = gr.Plot(label="Interactive Poetry Map", value=initial_map)
                map_state = gr.State(initial_points)
                map_choice = gr.Dropdown(
                    choices=[point["poem_id"] for point in initial_points if point.get("poem_id")],
                    value=initial_points[0]["poem_id"] if initial_points else None,
                    label="Open visible poem",
                )
                map_open = gr.Button("Open map selection")
                map_refresh.click(
                    fn=build_map_figure,
                    inputs=[map_color, map_search],
                    outputs=[map_plot, map_state, map_choice],
                )
                map_outputs = build_result_panel("Map selection")
                map_open.click(
                    fn=open_map_poem,
                    inputs=[map_choice, analyze_top_k, analyze_imagery, analyze_device],
                    outputs=list(map_outputs),
                )

            with gr.Accordion("🌸 Imagery Explorer", open=False):
                gr.HTML(
                    f"""
                    <div class="card">
                      <div class="section-title">{render_inline_icon(icon_map.get("imagery"), "Dates – Imagery Explorer")}Imagery Explorer</div>
                      <div class="small">Visualise the distinctive imagery vocabulary of each poet and discover which
                      visual themes define each genre — computed via TF-IDF over 3,340 annotated clips.</div>
                    </div>
                    """
                )
                with gr.Tabs():
                    with gr.Tab("🧬 Poet Fingerprint"):
                        gr.HTML(
                            """<div class="small" style="margin-bottom:0.75rem;">
                            Select a poet to see their 12 most <em>distinctive</em> imagery tags —
                            scored by TF-IDF so common corpus-wide tags (e.g. "heart") are down-weighted
                            in favour of tags that uniquely characterise this poet's voice.
                            Clip count (×N) shows how often each tag appears in their corpus.
                            </div>"""
                        )
                        _records, _poets, _genres = _load_imagery_corpus()
                        with gr.Row():
                            with gr.Column(scale=1):
                                poet_dropdown = gr.Dropdown(
                                    choices=_poets,
                                    value=_poets[0] if _poets else None,
                                    label="Select poet",
                                    interactive=True,
                                )
                                poet_btn = gr.Button("Show fingerprint", variant="primary")
                            with gr.Column(scale=3):
                                poet_plot = gr.Plot(
                                    label="Imagery Fingerprint",
                                    value=build_poet_fingerprint_plot(_poets[0] if _poets else ""),
                                )
                        poet_btn.click(
                            fn=build_poet_fingerprint_plot,
                            inputs=[poet_dropdown],
                            outputs=[poet_plot],
                        )
                        poet_dropdown.change(
                            fn=build_poet_fingerprint_plot,
                            inputs=[poet_dropdown],
                            outputs=[poet_plot],
                        )

                    with gr.Tab("🗺️ Genre Heatmap"):
                        gr.HTML(
                            """<div class="small" style="margin-bottom:0.75rem;">
                            Each cell shows what percentage of clips in that genre contain each imagery tag.
                            Darker = more frequent. Hover a cell for the exact value.
                            Columns are the 25 most common tags corpus-wide; genres are rows.
                            Use this to see which imagery motifs define each genre.
                            </div>"""
                        )
                        genre_heatmap = gr.Plot(
                            label="Genre × Imagery Heatmap",
                            value=build_genre_heatmap_plot(),
                        )

        with gr.Column(visible=False) as search_group:
            search_back = gr.Button("Back to Start Screen", elem_classes=["back-cta"])
            gr.HTML(
                f"""
                <div class="card">
                  <div class="section-title">{render_inline_icon(icon_map.get("search"), "Finjan – Search mode")}Search for a poem (audio) in the corpus</div>
                  <div class="small">Use this when your main goal is retrieval: find which corpus poem is closest to the input audio. Similarity score is the main confidence signal here.</div>
                </div>
                """
            )
            with gr.Row():
                search_top_k = gr.Slider(1, 10, value=5, step=1, label="Number of candidate poems")
                search_genre = gr.Textbox(label="Genre filter", placeholder="optional")
                search_imagery = gr.Textbox(label="Imagery tag filter", placeholder="optional")
                search_poet = gr.Textbox(label="Poet filter", placeholder="optional")
                search_device = gr.Radio(choices=["cpu", "mps", "cuda"], value=default_device, label="Compute device")

            with gr.Tabs():
                with gr.Tab("Upload audio"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            search_upload_audio = gr.Audio(type="filepath", sources=["upload"], label="Upload search clip")
                            search_upload_lora = gr.Checkbox(label="Use LoRA (experimental)", value=False)
                            search_upload_btn = gr.Button("Search corpus with upload", variant="primary")
                        with gr.Column(scale=2):
                            search_preview = gr.HTML()
                            search_results = gr.HTML()
                            search_choice = gr.Dropdown(label="Open candidate")
                            search_open = gr.Button("Open candidate poem")
                    search_upload_btn.click(
                        fn=search_audio_candidates,
                        inputs=[
                            search_upload_audio,
                            search_top_k,
                            search_genre,
                            search_imagery,
                            search_poet,
                            search_device,
                            search_upload_lora,
                        ],
                        outputs=[search_preview, search_results, search_choice],
                    )

                with gr.Tab("Record audio"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            search_record_audio = gr.Audio(type="filepath", sources=["microphone"], label="Record search clip")
                            search_record_lora = gr.Checkbox(label="Use LoRA (experimental)", value=False)
                            search_record_btn = gr.Button("Search corpus with recording", variant="primary")
                        with gr.Column(scale=2):
                            search_preview_record = gr.HTML()
                            search_results_record = gr.HTML()
                            search_choice_record = gr.Dropdown(label="Open candidate")
                            search_open_record = gr.Button("Open candidate poem")
                    search_record_btn.click(
                        fn=search_audio_candidates,
                        inputs=[
                            search_record_audio,
                            search_top_k,
                            search_genre,
                            search_imagery,
                            search_poet,
                            search_device,
                            search_record_lora,
                        ],
                        outputs=[search_preview_record, search_results_record, search_choice_record],
                    )

            search_outputs = build_result_panel("Selected candidate")
            search_open.click(
                fn=build_poem_card_from_id,
                inputs=[search_choice, search_top_k, search_imagery, search_device],
                outputs=list(search_outputs),
            )
            search_open_record.click(
                fn=build_poem_card_from_id,
                inputs=[search_choice_record, search_top_k, search_imagery, search_device],
                outputs=list(search_outputs),
            )
        # ── Button wiring ──────────────────────────────────────────────────────
        _all_screens = [start_screen, analyze_group, search_group]

        analyze_start.click(fn=show_analyze_mode, outputs=_all_screens)
        search_start.click(fn=show_search_mode, outputs=_all_screens)
        analyze_back.click(fn=show_start_screen, outputs=_all_screens)
        search_back.click(fn=show_start_screen, outputs=_all_screens)

    return ui


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Nabat-AI MotherDuck-style app")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
    parser.add_argument("--port", type=int, default=7860, help="Local port")
    parser.add_argument("--device", type=str, default="cpu", help="Default device shown in UI")
    return parser.parse_args()


def choose_server_port(preferred_port: int, max_attempts: int = 10) -> int:
    for port in range(preferred_port, preferred_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred_port


if __name__ == "__main__":
    args = parse_args()
    app = build_ui(default_device=args.device)
    launch_port = choose_server_port(args.port)
    if launch_port != args.port:
        logger.warning("Port {} is busy; launching on {} instead.", args.port, launch_port)
    logger.info("Launching app at http://127.0.0.1:{}", launch_port)
    icon_dir = PROJECT_ROOT / "src" / "icons"
    allowed = [str(icon_dir)] if icon_dir.exists() else []
    app.launch(server_port=launch_port, share=args.share, allowed_paths=allowed)

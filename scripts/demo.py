"""
scripts/demo.py

End-to-end Nabat-AI inference demo.

Pipeline for a single audio clip (.mp3):
    1. Whisper-small  → Arabic transcription
    2. AraPoemBERT    → genre prediction (8 classes)
    3. AraPoemBERT    → emotion_text prediction (12 classes, from transcript)
    4. ArousalMLP     → delivery arousal Low/Medium/High (from audio, scratch model)
    5. Delivery Mismatch → text-implied arousal ≠ audio arousal → cultural signal
    6. NabatiRetriever→ top-5 similar poems (semantic + imagery-tag boost)

Usage:
    uv run python scripts/demo.py path/to/clip.mp3
    uv run python scripts/demo.py path/to/clip.mp3 --top-k 10
    uv run python scripts/demo.py path/to/clip.mp3 --imagery-filter heart
    uv run python scripts/demo.py path/to/clip.mp3 --out outputs/result.json

Output:
    - Formatted table printed to terminal
    - InferenceResult saved as JSON to --out (default: outputs/demo_result.json)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

# Enforce fully local runtime: submitted demo/app must not fetch from the network.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_VERBOSITY", "error")

# ── Suppress noisy HuggingFace messages before any transformers import ─────────
# LOAD REPORT (UNEXPECTED/MISSING keys) is printed by from_pretrained when loading
# a base model before the fine-tuned state_dict is applied — harmless.
# attention_mask and SuppressTokens warnings are internal HF generation details.
import transformers as _tf

_tf.logging.set_verbosity_error()
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("peft").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*attention mask.*")
warnings.filterwarnings("ignore", message=".*SuppressTokens.*")
warnings.filterwarnings("ignore", message=".*unauthenticated.*")

from transformers import AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402

# ── project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.labels import EMOTION_CLASSES, GENRE_CLASSES, get_merged_emotion_classes  # noqa: E402
from data.arousal_labels import AROUSAL_CLASSES, emotion_to_arousal  # noqa: E402
from data.schema import InferenceResult, RankedPrediction, SimilarPoem  # noqa: E402
from models.audio_cnn import Emotion1DCNN  # noqa: E402
from models.retrieval import NabatiRetriever  # noqa: E402


# ── paths ─────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
LOCAL_MODEL_DIR = PROJECT_ROOT / "models" / "local"
WHISPER_MODEL_DIR = LOCAL_MODEL_DIR / "whisper-small"
ARAPOEM_MODEL_DIR = LOCAL_MODEL_DIR / "arapoembert"
GENRE_CKPT = PROJECT_ROOT / "outputs/models/arapoem_genre/arapoem_genre_best.pt"
EMOTION_CKPT = (
    PROJECT_ROOT / "outputs/models/arapoem_emotion/arapoem_emotion_text_best.pt"
)
AROUSAL_CKPT = PROJECT_ROOT / "outputs/models/arousal_mlp/arousal_mlp_arousal_best.pt"
AROUSAL_SCALER = PROJECT_ROOT / "outputs/models/arousal_mlp/arousal_scaler.pkl"
CNN_CKPT = PROJECT_ROOT / "outputs/models/audio_cnn/audio_cnn_emotion_best.pt"
WHISPER_ADAPTER = PROJECT_ROOT / "outputs/models/whisper_nabati/best"
RETRIEVAL_DIR = PROJECT_ROOT / "outputs/retrieval"
POEM_PREDICTION_FILES = [
    PROJECT_ROOT / "outputs/reports/poem_emotion_predictions_test.json",
    PROJECT_ROOT / "outputs/reports/poem_emotion_predictions_val.json",
]

ARAPOEM_MODEL = "faisalq/bert-base-arapoembert"
WHISPER_MODEL = "openai/whisper-small"
SAMPLE_RATE = 16_000
MAX_AUDIO_SEC = 30
MAX_SEQ_LEN = 32  # AraPoemBERT hard limit
N_MFCC = 13
EMOTION_TEXT_CLASSES = get_merged_emotion_classes("rare_merge_v1")

os.environ.setdefault("NABAT_ARAPOEM_MODEL_DIR", str(ARAPOEM_MODEL_DIR))
os.environ.setdefault("NABAT_WHISPER_MODEL_DIR", str(WHISPER_MODEL_DIR))


def _has_model_weights(model_dir: Path) -> bool:
    return any(
        (model_dir / name).exists()
        for name in (
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
        )
    )


def _runtime_asset_error() -> str:
    return "Models not cached. Run: just cache-models (one-time)."


def get_missing_runtime_assets() -> list[str]:
    missing: list[str] = []

    whisper_ok = (
        WHISPER_MODEL_DIR.exists()
        and (WHISPER_MODEL_DIR / "config.json").exists()
        and (WHISPER_MODEL_DIR / "preprocessor_config.json").exists()
        and _has_model_weights(WHISPER_MODEL_DIR)
    )
    if not whisper_ok:
        missing.append("whisper-small")

    arapoem_ok = (
        ARAPOEM_MODEL_DIR.exists()
        and (ARAPOEM_MODEL_DIR / "config.json").exists()
        and (
            (ARAPOEM_MODEL_DIR / "tokenizer.json").exists()
            or (ARAPOEM_MODEL_DIR / "tokenizer_config.json").exists()
        )
        and _has_model_weights(ARAPOEM_MODEL_DIR)
    )
    if not arapoem_ok:
        missing.append("arapoembert")

    return missing


def ensure_runtime_assets(
    *, require_whisper: bool = False, require_text_models: bool = False
) -> None:
    missing = get_missing_runtime_assets()
    needed: list[str] = []
    if require_whisper:
        needed.append("whisper-small")
    if require_text_models:
        needed.append("arapoembert")

    missing_needed = [asset for asset in needed if asset in missing]
    if missing_needed:
        joined = ", ".join(missing_needed)
        raise RuntimeError(f"{_runtime_asset_error()} Missing local assets: {joined}.")


# ── Corpus metadata lookup ─────────────────────────────────────────────────────


def _is_arabic(text: str, threshold: float = 0.4) -> bool:
    """Return True if at least `threshold` fraction of chars are Arabic."""
    if not text or not text.strip():
        return False
    arabic = sum(
        1 for c in text if "\u0600" <= c <= "\u06ff" or "\u0750" <= c <= "\u077f"
    )
    return arabic / max(len(text), 1) >= threshold


def lookup_poem_metadata(audio_path: Path) -> dict:
    """
    Look up poem title, poet, and ground-truth labels for the given audio clip
    by scanning master_dataset.jsonl (or train/val/test splits).

    Returns a dict with keys: poem_title, poet_en, genre_en, emotion_text,
    source_poem, text_corrected, and text_whisper.
    Falls back to empty strings if not found.
    """
    filename = audio_path.name
    candidates = [
        PROJECT_ROOT / "data" / "processed" / "master_dataset.jsonl",
        PROJECT_ROOT / "data" / "processed" / "train.jsonl",
        PROJECT_ROOT / "data" / "processed" / "val.jsonl",
        PROJECT_ROOT / "data" / "processed" / "test.jsonl",
    ]
    for jsonl_path in candidates:
        if not jsonl_path.exists():
            continue
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                if Path(rec.get("audio_filename", "")).name == filename:
                    return {
                        "poem_title": rec.get("poem_title") or "",
                        "poet_en": rec.get("poet_en") or "",
                        "genre_en": rec.get("genre_en") or "",
                        "emotion_text": rec.get("emotion_text") or "",
                        "source_poem": rec.get("source_poem") or "",
                        "text_corrected": rec.get("text_corrected") or "",
                        "text_whisper": rec.get("text_whisper") or "",
                    }
            except (json.JSONDecodeError, KeyError):
                continue
    return {
        "poem_title": "",
        "poet_en": "",
        "genre_en": "",
        "emotion_text": "",
        "source_poem": "",
        "text_corrected": "",
        "text_whisper": "",
    }


def load_poem_prediction_lookup() -> dict[str, dict[str, dict]]:
    lookup: dict[str, dict[str, dict]] = {}
    for report_path in POEM_PREDICTION_FILES:
        if not report_path.exists():
            continue
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        for variant_name, poems in payload.items():
            lookup.setdefault(variant_name, {}).update(poems)
    return lookup


def _topk_predictions(
    probs: np.ndarray | torch.Tensor, labels: list[str], k: int = 3
) -> list[RankedPrediction]:
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    ranked = np.argsort(probs)[::-1][:k]
    return [
        RankedPrediction(label=labels[int(idx)], prob=float(probs[int(idx)]))
        for idx in ranked
    ]


# ── ASR ───────────────────────────────────────────────────────────────────────


def load_whisper(device: str, use_lora: bool = False):
    """Load Whisper-small baseline by default; LoRA is opt-in."""
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    ensure_runtime_assets(require_whisper=True)
    try:
        processor = WhisperProcessor.from_pretrained(
            str(WHISPER_MODEL_DIR), local_files_only=True
        )
        base = WhisperForConditionalGeneration.from_pretrained(
            str(WHISPER_MODEL_DIR), local_files_only=True
        )
    except OSError as exc:
        raise RuntimeError(
            f"{_runtime_asset_error()} Missing local assets: whisper-small."
        ) from exc

    if use_lora and WHISPER_ADAPTER.exists():
        try:
            from peft import PeftModel

            model = PeftModel.from_pretrained(
                base, str(WHISPER_ADAPTER), local_files_only=True
            )
            logger.info(f"Whisper loaded with LoRA adapter → {WHISPER_ADAPTER}")
        except Exception as e:
            logger.warning(f"LoRA adapter load failed ({e}), using zero-shot Whisper")
            model = base
    else:
        logger.info("Using whisper-small zero-shot baseline")
        model = base

    model = model.to(device).eval()
    return processor, model


def transcribe(audio_path: Path, processor, model, device: str) -> str:
    """Load audio and run Whisper inference → Arabic transcription string."""
    wav, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    wav = wav[: MAX_AUDIO_SEC * SAMPLE_RATE]

    inputs = processor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt")
    input_features = inputs["input_features"].to(device)
    # Explicit attention_mask of all ones (no padding in single-clip inference)
    attention_mask = torch.ones(
        input_features.shape[:2], dtype=torch.long, device=device
    )

    # Pass forced_decoder_ids=None to avoid duplicate SuppressTokens logits processors
    # (language/task args already create those internally in generate())
    forced_ids = processor.get_decoder_prompt_ids(language="ar", task="transcribe")
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features=input_features,
            attention_mask=attention_mask,
            forced_decoder_ids=forced_ids,
        )
    return processor.tokenizer.decode(
        predicted_ids[0], skip_special_tokens=True
    ).strip()


# ── Genre classifier ──────────────────────────────────────────────────────────


def load_genre_model(device: str):
    """Load AraPoemBERT fine-tuned for genre classification."""
    if not GENRE_CKPT.exists():
        logger.warning(
            f"Genre checkpoint not found at {GENRE_CKPT} — skipping genre prediction"
        )
        return None, None

    ensure_runtime_assets(require_text_models=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(ARAPOEM_MODEL_DIR), local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(ARAPOEM_MODEL_DIR),
        num_labels=len(GENRE_CLASSES),
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    state = torch.load(GENRE_CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device).eval()
    logger.info(f"Genre model loaded ({len(GENRE_CLASSES)} classes)")
    return tokenizer, model


def _classify_text(
    text: str, tokenizer, model, device: str, classes: list[str]
) -> tuple[str, float, list[RankedPrediction]]:
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
        ).logits
    probs = F.softmax(logits, dim=-1)[0].cpu()
    idx = int(probs.argmax())
    return classes[idx], float(probs[idx]), _topk_predictions(probs, classes)


def predict_genre(
    text: str, tokenizer, model, device: str
) -> tuple[str, float, list[RankedPrediction]]:
    return _classify_text(text, tokenizer, model, device, GENRE_CLASSES)


# ── Emotion-text classifier ───────────────────────────────────────────────────


def load_emotion_model(device: str):
    """Load AraPoemBERT fine-tuned for emotion_text classification (12 classes)."""
    if not EMOTION_CKPT.exists():
        logger.warning(f"Emotion checkpoint not found at {EMOTION_CKPT}")
        return None, None
    ensure_runtime_assets(require_text_models=True)
    tokenizer = AutoTokenizer.from_pretrained(
        str(ARAPOEM_MODEL_DIR), local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        str(ARAPOEM_MODEL_DIR),
        num_labels=len(EMOTION_TEXT_CLASSES),
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    state = torch.load(EMOTION_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=False)
    model = model.to(device).eval()
    logger.info(f"Emotion model loaded ({len(EMOTION_TEXT_CLASSES)} classes)")
    return tokenizer, model


def predict_emotion_text(
    text: str, tokenizer, model, device: str
) -> tuple[str, float, list[RankedPrediction]]:
    return _classify_text(text, tokenizer, model, device, EMOTION_TEXT_CLASSES)


# ── Arousal MLP (scratch model) ───────────────────────────────────────────────


class ArousalMLP(nn.Module):
    """Mirrors the architecture in scripts/train_arousal.py."""

    def __init__(
        self,
        input_dim: int = 34,
        hidden_dim: int = 128,
        n_layers: int = 2,
        dropout: float = 0.3,
        n_classes: int = 3,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _extract_arousal_features(
    audio_path: Path, sr: int = 16000, max_sec: int = 30, n_mfcc: int = 13
) -> np.ndarray | None:
    """Extract 34-dim librosa feature vector for arousal prediction."""
    try:
        y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
        y = y[: max_sec * sr]
        if len(y) == 0:
            return None

        feats: list[float] = []
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        feats.extend(mfcc.mean(axis=1).tolist())
        feats.extend(mfcc.std(axis=1).tolist())
        rms = librosa.feature.rms(y=y)[0]
        feats += [float(rms.mean()), float(rms.std())]
        zcr = librosa.feature.zero_crossing_rate(y=y)[0]
        feats.append(float(zcr.mean()))
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        feats += [float(centroid.mean()), float(centroid.std())]
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        feats.append(float(rolloff.mean()))
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        feats.append(float(np.squeeze(tempo)))
        silence_threshold = rms.mean() * 0.1
        pause_ratio = float((rms < silence_threshold).sum()) / max(len(rms), 1)
        feats.append(pause_ratio)
        return np.array(feats, dtype=np.float32)
    except Exception as e:
        logger.warning(f"Arousal feature extraction failed: {e}")
        return None


def load_arousal_model(device: str):
    """Load ArousalMLP + StandardScaler."""
    if not AROUSAL_CKPT.exists() or not AROUSAL_SCALER.exists():
        logger.warning(
            "Arousal checkpoint or scaler not found — skipping arousal prediction"
        )
        return None, None
    import pickle

    scaler = pickle.load(open(AROUSAL_SCALER, "rb"))
    model = ArousalMLP()
    state = torch.load(AROUSAL_CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device).eval()
    logger.info("Arousal MLP loaded (3-class Low/Medium/High)")
    return scaler, model


def predict_arousal(audio_path: Path, scaler, model, device: str) -> tuple[str, float]:
    """Extract librosa features and predict arousal level → (label, confidence)."""
    feats = _extract_arousal_features(audio_path)
    if feats is None:
        return "unknown", 0.0
    feats_scaled = scaler.transform(feats.reshape(1, -1))
    x = torch.tensor(feats_scaled, dtype=torch.float32).to(device)
    with torch.no_grad():
        logits = model(x)
    probs = F.softmax(logits, dim=-1)[0].cpu()
    idx = int(probs.argmax())
    return AROUSAL_CLASSES[idx], float(probs[idx])


# ── Audio emotion CNN ─────────────────────────────────────────────────────────


def load_cnn(device: str):
    """Load Emotion1DCNN from scratch checkpoint."""
    if not CNN_CKPT.exists():
        logger.warning(
            f"CNN checkpoint not found at {CNN_CKPT} — skipping audio emotion prediction"
        )
        return None

    model = Emotion1DCNN(num_classes=len(EMOTION_CLASSES))
    state = torch.load(CNN_CKPT, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model = model.to(device).eval()
    logger.info(f"Audio CNN loaded ({len(EMOTION_CLASSES)} classes)")
    return model


def predict_emotion_audio(
    audio_path: Path, cnn_model, device: str
) -> tuple[str, float, list[RankedPrediction]]:
    """Extract mel-spec and predict audio emotion → (label, confidence)."""
    wav, _ = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    wav = wav[: 8 * SAMPLE_RATE]  # CNN trained on 8s clips

    mel = librosa.feature.melspectrogram(y=wav, sr=SAMPLE_RATE, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Pad / trim to fixed width (CNN expects consistent T dimension)
    target_len = 8 * SAMPLE_RATE // 512  # ~250 frames for hop_length=512
    if mel_db.shape[1] < target_len:
        mel_db = np.pad(mel_db, ((0, 0), (0, target_len - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :target_len]

    x = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 128, T)
    with torch.no_grad():
        logits = cnn_model(x)
    probs = F.softmax(logits, dim=-1)[0].cpu()
    idx = int(probs.argmax())
    return (
        EMOTION_CLASSES[idx],
        float(probs[idx]),
        _topk_predictions(probs, EMOTION_CLASSES),
    )


# ── Retrieval ─────────────────────────────────────────────────────────────────


def load_retriever(device: str) -> NabatiRetriever | None:
    """Load the saved FAISS retrieval index."""
    if not RETRIEVAL_DIR.exists() or not (RETRIEVAL_DIR / "meta.pkl").exists():
        logger.warning(f"Retrieval index not found at {RETRIEVAL_DIR}")
        return None
    ensure_runtime_assets(require_text_models=True)
    retriever = NabatiRetriever.load(RETRIEVAL_DIR, device=device)
    return retriever


# ── Main pipeline ─────────────────────────────────────────────────────────────


def run_demo(
    audio_path: Path,
    top_k: int = 5,
    imagery_filter: str | None = None,
    out_path: Path = PROJECT_ROOT / "outputs/demo_result.json",
    device: str = "cpu",
    use_lora: bool = False,
) -> InferenceResult:
    """Run the full Nabat-AI pipeline on a single audio clip."""

    # ── Poem metadata lookup (title, poet, ground-truth labels) ──────────
    meta = lookup_poem_metadata(audio_path)
    poet_display = meta["poet_en"] if meta["poet_en"] else ""

    logger.info(f"{'=' * 60}")
    logger.info(f"Nabat-AI Demo  |  {audio_path.name}")
    if meta["poem_title"]:
        logger.info(f"Poem title     |  {meta['poem_title']}")
    if poet_display:
        logger.info(f"Poet           |  {poet_display}")
    logger.info(f"{'=' * 60}")

    t0 = time.perf_counter()

    # ── Step 1: ASR / corpus transcript override ──────────────────────────
    corrected_text = (meta.get("text_corrected") or "").strip()
    corpus_whisper_text = (meta.get("text_whisper") or "").strip()
    analysis_text = ""
    transcription = ""
    asr_model_name = WHISPER_MODEL

    if corrected_text:
        logger.info("Step 1/6 — Corpus transcript override (skipping Whisper)")
        analysis_text = corrected_text
        transcription = corpus_whisper_text or corrected_text
        asr_model_name = "corpus_corrected_text"
    else:
        logger.info("Step 1/6 — ASR (Whisper)")
        processor, whisper = load_whisper(device, use_lora=use_lora)
        transcription = transcribe(audio_path, processor, whisper, device)
        analysis_text = transcription
        asr_model_name = f"{WHISPER_MODEL}+LoRA" if use_lora else WHISPER_MODEL
        logger.info(f"  Transcription: {transcription}")
        del whisper  # free memory before loading next model

    # ── Step 2: Genre ─────────────────────────────────────────────────────
    logger.info("Step 2/6 — Genre classification (AraPoemBERT)")
    tokenizer, genre_model = load_genre_model(device)
    if genre_model is not None:
        genre, genre_conf, genre_topk = predict_genre(
            analysis_text, tokenizer, genre_model, device
        )
        del genre_model
    else:
        genre, genre_conf, genre_topk = "unknown", 0.0, []
    logger.info(f"  Genre: {genre}  (conf={genre_conf:.3f})")

    # ── Step 3: Emotion from text ─────────────────────────────────────────
    logger.info("Step 3/6 — Emotion (text) classification (AraPoemBERT)")
    emo_tok, emo_model = load_emotion_model(device)
    if emo_model is not None:
        emo_text, emo_text_conf, emotion_clip_topk = predict_emotion_text(
            analysis_text, emo_tok, emo_model, device
        )
        del emo_model
    else:
        emo_text, emo_text_conf, emotion_clip_topk = None, None, []
    if emo_text:
        logger.info(f"  Emotion (text): {emo_text}  (conf={emo_text_conf:.3f})")

    # ── Step 4: Arousal from audio (scratch model) ────────────────────────
    logger.info("Step 4/6 — Delivery arousal (ArousalMLP, from scratch)")
    scaler, arousal_model = load_arousal_model(device)
    if arousal_model is not None:
        arousal, arousal_conf = predict_arousal(
            audio_path, scaler, arousal_model, device
        )
        del arousal_model
    else:
        arousal, arousal_conf = None, None
    if arousal:
        logger.info(f"  Arousal: {arousal}  (conf={arousal_conf:.3f})")

    # ── Step 5: Delivery mismatch ─────────────────────────────────────────
    delivery_mismatch: bool | None = None
    if emo_text and arousal:
        text_implied_arousal = emotion_to_arousal(emo_text)
        if text_implied_arousal:
            delivery_mismatch = text_implied_arousal != arousal
            if delivery_mismatch:
                logger.info(
                    f"  Delivery mismatch: {emo_text} implies {text_implied_arousal} "
                    f"but audio is {arousal} — culturally significant performance choice"
                )

    # ── Audio emotion CNN (auxiliary) ────────────────────────────────────
    # Emotion1DCNN provides a clip-level audio emotion prediction that is
    # gated into the poem-level emotion fusion pipeline as an auxiliary signal.
    # Arousal (above) is the primary from-scratch audio branch; this CNN
    # contributes the emotion_audio field and is down-weighted by the gate
    # when audio and text emotion signals disagree (high DMS clips).
    cnn = load_cnn(device)
    if cnn is not None:
        emo_audio, emo_audio_conf, audio_clip_topk = predict_emotion_audio(
            audio_path, cnn, device
        )
        del cnn
    else:
        emo_audio, emo_audio_conf, audio_clip_topk = "unknown", 0.0, []

    poem_predictions = load_poem_prediction_lookup()
    poem_id = meta["source_poem"] or None
    raw_poem = poem_predictions.get("raw", {}).get(poem_id or "", {})
    full_poem = poem_predictions.get("full_fusion", {}).get(poem_id or "", {})

    if raw_poem:
        poem_raw_topk = [
            RankedPrediction(label=item["label"], prob=float(item["prob"]))
            for item in raw_poem.get("poem_emotion_raw_topk", [])
        ]
        poem_raw_top1 = raw_poem.get("poem_emotion_raw_top1")
        poem_raw_conf = raw_poem.get("poem_emotion_raw_confidence")
        poem_secondary = raw_poem.get("poem_emotion_secondary", [])
        poem_uncertainty = raw_poem.get("uncertainty", {})
        poem_clip_support = raw_poem.get("clip_support", [])
        manual_genre = raw_poem.get("manual_genre") or meta["genre_en"] or None
    else:
        poem_raw_topk = emotion_clip_topk
        poem_raw_top1 = emo_text
        poem_raw_conf = emo_text_conf
        poem_secondary = [item.label for item in emotion_clip_topk[1:]]
        poem_uncertainty = {}
        poem_clip_support = []
        manual_genre = meta["genre_en"] or None

    if full_poem:
        audio_poem_probs = full_poem.get("audio_emotion_poem_probs") or []
        audio_poem_topk = (
            _topk_predictions(
                np.array(audio_poem_probs, dtype=np.float64), EMOTION_TEXT_CLASSES
            )
            if audio_poem_probs
            else []
        )
        emotion_poem_final = full_poem.get("emotion_poem_final")
        emotion_poem_final_reason = full_poem.get("emotion_poem_final_reason")
        arousal_poem = full_poem.get("poem_arousal")
        arousal_poem_conf = full_poem.get("poem_arousal_confidence")
        dms_poem = full_poem.get("dms_poem")
        delivery_nuance_tag = full_poem.get("delivery_nuance_tag")
        audio_emotion_poem_aux = full_poem.get("audio_emotion_poem_aux")
        audio_emotion_poem_aux_conf = full_poem.get("audio_emotion_poem_aux_confidence")
        audio_emotion_used = full_poem.get("audio_emotion_used_in_decision")
    else:
        audio_poem_topk = audio_clip_topk
        emotion_poem_final = poem_raw_top1 or emo_text
        emotion_poem_final_reason = (
            "Poem cache unavailable; falling back to clip-level emotion."
        )
        arousal_poem = arousal
        arousal_poem_conf = arousal_conf
        dms_poem = delivery_mismatch
        delivery_nuance_tag = None
        audio_emotion_poem_aux = emo_audio if emo_audio != "unknown" else None
        audio_emotion_poem_aux_conf = emo_audio_conf if emo_audio != "unknown" else None
        audio_emotion_used = False

    # ── Step 6: Retrieval ─────────────────────────────────────────────────
    logger.info(f"Step 6/6 — Retrieval (top-{top_k} similar poems)")
    retriever = load_retriever(device)
    similar: list[SimilarPoem] = []
    if retriever is not None:
        hits = retriever.search_poems(
            query=analysis_text,
            top_k=top_k,
            imagery_filter=imagery_filter,
            tag_boost=0.15,
        )
        similar = [
            SimilarPoem(
                score=h["score"],
                poet_en=h.get("poet_en", ""),
                source_poem=h.get("source_poem", ""),
                genre_en=h.get("genre_en", ""),
                emotion_text=h.get("emotion_text", ""),
                text_corrected=h.get("text_corrected", ""),
                imagery_tags_en=h.get("imagery_tags_en"),
                audio_filename=h.get("audio_filename", ""),
                n_clips=h.get("n_clips"),
            )
            for h in hits
        ]

    inference_ms = (time.perf_counter() - t0) * 1000

    result = InferenceResult(
        audio_file=str(audio_path),
        poem_id=poem_id,
        transcription=transcription,
        asr_model=asr_model_name,
        genre=genre,
        genre_confidence=genre_conf,
        genre_topk=genre_topk,
        manual_genre=manual_genre,
        emotion_text=emo_text,
        emotion_text_confidence=emo_text_conf,
        emotion_clip_topk=emotion_clip_topk,
        emotion_poem_raw_topk=poem_raw_topk,
        emotion_poem_raw_top1=poem_raw_top1,
        emotion_poem_raw_confidence=poem_raw_conf,
        emotion_poem_secondary=poem_secondary,
        emotion_poem_final=emotion_poem_final,
        emotion_poem_final_reason=emotion_poem_final_reason,
        poem_emotion_uncertainty=poem_uncertainty,
        poem_emotion_clip_support=poem_clip_support,
        emotion_audio=emo_audio,
        emotion_audio_confidence=emo_audio_conf,
        audio_emotion_poem_aux=audio_emotion_poem_aux,
        audio_emotion_poem_aux_confidence=audio_emotion_poem_aux_conf,
        audio_emotion_poem_aux_topk=audio_poem_topk,
        audio_emotion_used_in_decision=audio_emotion_used,
        arousal=arousal,
        arousal_confidence=arousal_conf,
        delivery_mismatch=delivery_mismatch,
        arousal_poem=arousal_poem,
        arousal_poem_confidence=arousal_poem_conf,
        dms_poem=dms_poem,
        delivery_nuance_tag=delivery_nuance_tag,
        similar_poems=similar,
        inference_ms=round(inference_ms, 1),
    )

    # ── Print results ─────────────────────────────────────────────────────
    _print_result(result, poem_title=meta["poem_title"], poet_en=meta["poet_en"])

    # ── Save JSON ─────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    logger.success(f"Result saved → {out_path}")

    return result


def _print_result(
    r: InferenceResult,
    poem_title: str = "",
    poet_en: str = "",
) -> None:
    """Pretty-print InferenceResult to terminal."""
    sep = "─" * 62
    logger.info(f"\n{'═' * 62}")
    logger.info("  NABAT-AI INFERENCE RESULT")
    logger.info(f"{'═' * 62}")
    logger.info(f"  File    : {Path(r.audio_file).name}")
    if r.poem_id:
        logger.info(f"  Poem ID : {r.poem_id}")
    if poem_title:
        logger.info(f"  Title   : {poem_title}")
    if poet_en:
        logger.info(f"  Poet    : {poet_en}")
    logger.info(f"  Latency : {r.inference_ms:.0f} ms")
    logger.info(sep)
    logger.info(f"  Transcription  : {r.transcription}")
    logger.info(sep)
    logger.info(f"  Genre          : {r.genre:<30} (conf={r.genre_confidence:.2%})")
    if r.manual_genre:
        logger.info(f"  Genre (manual) : {r.manual_genre}")
    if r.emotion_text:
        logger.info(
            f"  Emotion (text) : {r.emotion_text:<30} (conf={r.emotion_text_confidence:.2%})"
        )
    else:
        logger.info("  Emotion (text) : not available — train with: just train-emotion")
    if r.emotion_poem_raw_top1:
        logger.info(
            f"  Poem raw emo   : {r.emotion_poem_raw_top1:<30} (conf={r.emotion_poem_raw_confidence or 0:.2%})"
        )
    if r.emotion_poem_final:
        logger.info(f"  Poem final emo : {r.emotion_poem_final}")
        if r.emotion_poem_final_reason:
            logger.info(f"  Decision note  : {r.emotion_poem_final_reason}")
    if r.arousal:
        logger.info(
            f"  Delivery Arousal: {r.arousal:<29} (conf={r.arousal_confidence:.2%})"
        )
        if r.delivery_mismatch is not None:
            text_ar = emotion_to_arousal(r.emotion_text) if r.emotion_text else "?"
            flag = "MISMATCH" if r.delivery_mismatch else "aligned"
            logger.info(
                f"  Delivery Bias  : text→{text_ar:<8} audio={r.arousal:<8} [{flag}]"
            )
    if r.arousal_poem:
        logger.info(
            f"  Poem arousal   : {r.arousal_poem:<29} (conf={(r.arousal_poem_confidence or 0):.2%})"
        )
    if r.dms_poem is not None:
        logger.info(f"  DMS            : {'MISMATCH' if r.dms_poem else 'aligned'}")
    if r.delivery_nuance_tag:
        logger.info(f"  Delivery tag   : {r.delivery_nuance_tag}")
    if r.audio_emotion_poem_aux:
        used = "used" if r.audio_emotion_used_in_decision else "aux-only"
        logger.info(f"  Audio aux emo  : {r.audio_emotion_poem_aux:<30} [{used}]")
    logger.info(sep)
    logger.info(f"  Top-{len(r.similar_poems)} Similar Poems (poem-level):")
    for i, p in enumerate(r.similar_poems, 1):
        n = getattr(p, "n_clips", "?")
        logger.info(
            f"    {i}. [{p.score:.3f}] {p.poet_en} — {p.source_poem}  ({n} clips)"
        )
        logger.info(f"       Genre: {p.genre_en}  |  Tags: {p.imagery_tags_en}")
        # Show best line only if it looks like Arabic text; skip garbage transcriptions
        best = p.text_corrected[:80] if p.text_corrected else ""
        if best and _is_arabic(best):
            logger.info(
                f"       Sample: {best}{'…' if len(p.text_corrected) > 80 else ''}"
            )
        elif best:
            logger.info("       Sample: [Arabic text]")
    logger.info(f"{'═' * 62}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Nabat-AI end-to-end demo: audio → transcription + genre + emotion + similar poems"
    )
    p.add_argument("audio", type=Path, help="Path to input .mp3 audio file")
    p.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of similar poems to retrieve (default: 5)",
    )
    p.add_argument(
        "--imagery-filter",
        type=str,
        default=None,
        help="Only retrieve poems whose imagery tags contain this word (e.g. 'heart')",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "outputs/demo_result.json",
        help="Path to save InferenceResult JSON (default: outputs/demo_result.json)",
    )
    p.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Compute device: cpu | cuda | mps (default: cpu)",
    )
    p.add_argument(
        "--use-lora",
        action="store_true",
        help="Use LoRA Whisper adapter instead of the baseline model",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not args.audio.exists():
        logger.error(f"Audio file not found: {args.audio}")
        sys.exit(1)

    run_demo(
        audio_path=args.audio,
        top_k=args.top_k,
        imagery_filter=args.imagery_filter,
        out_path=args.out,
        device=args.device,
        use_lora=args.use_lora,
    )

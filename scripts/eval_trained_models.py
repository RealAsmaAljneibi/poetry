"""
scripts/eval_trained_models.py

Re-evaluate all trained model checkpoints with Metrics v3:
  A) AraPoemBERT   — genre classification  → Macro-F1, Weighted-F1, κ
  B) Emotion1DCNN  — audio emotion          → Macro-F1, κ, Emotion Partial Credit
  C) Whisper-small LoRA fine-tuned         → WER, CER, Soft-CER v2

Produces a final comparison table vs. baselines from baseline_results.csv.

Usage:
    uv run python scripts/eval_trained_models.py           # all three
    uv run python scripts/eval_trained_models.py --skip-whisper
    uv run python scripts/eval_trained_models.py --only whisper
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import librosa
from loguru import logger
from sklearn.metrics import f1_score, cohen_kappa_score, classification_report

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.labels import (
    GENRE_CLASSES, EMOTION_CLASSES,
    encode_genre, encode_emotion,
)
from src.models.audio_cnn import Emotion1DCNN
from src.config import arapoem_genre_config, audio_cnn_config
from src.evaluation.metrics import (
    soft_cer, standard_cer, standard_wer,
    emotion_partial_credit,
)

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR   = Path("data/processed")
MODEL_DIR  = Path("outputs/models")
REPORT_DIR = Path("outputs/reports")
FIG_DIR    = Path("outputs/figures")

TEST_JSONL = DATA_DIR / "test.jsonl"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_test() -> list[dict]:
    return [
        json.loads(line)
        for line in TEST_JSONL.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ── A) AraPoemBERT genre ──────────────────────────────────────────────────────

def eval_arapoem_genre(records: list[dict], device: str) -> dict:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    cfg = arapoem_genre_config()
    ckpt = MODEL_DIR / "arapoem_genre" / "arapoem_genre_best.pt"
    if not ckpt.exists():
        logger.error(f"Checkpoint not found: {ckpt}")
        return {}

    logger.info("── A) AraPoemBERT genre ─────────────────────────────────")
    logger.info(f"  Loading {cfg.model_name} + {ckpt.name}")

    num_classes = len(GENRE_CLASSES)
    id2label    = {i: c for i, c in enumerate(GENRE_CLASSES)}
    label2id    = {c: i for i, c in enumerate(GENRE_CLASSES)}

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels   = num_classes,
        id2label     = id2label,
        label2id     = label2id,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    skipped = 0

    with torch.no_grad():
        for rec in records:
            text  = str(rec.get("text_corrected", "")).strip()
            label = rec.get("genre_en", "")
            if not text or not label:
                skipped += 1
                continue
            label_id = encode_genre(label)
            if label_id < 0:
                skipped += 1
                continue

            enc = tokenizer(
                text,
                return_tensors="pt",
                max_length=cfg.max_seq_len,
                truncation=True,
                padding="max_length",
            ).to(device)
            logits = model(**enc).logits
            pred_id = int(logits.argmax(-1).item())

            y_true.append(GENRE_CLASSES[label_id])
            y_pred.append(GENRE_CLASSES[pred_id])

    logger.info(f"  Evaluated {len(y_true)} records, skipped {skipped}")

    classes = sorted(set(y_true + y_pred))
    macro_f1    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    kappa       = cohen_kappa_score(y_true, y_pred)

    logger.info(f"  Macro-F1    : {macro_f1:.4f}   (mBERT baseline: 0.0907)")
    logger.info(f"  Weighted-F1 : {weighted_f1:.4f}")
    logger.info(f"  Cohen's κ   : {kappa:.4f}")
    logger.info("\n" + classification_report(
        y_true, y_pred,
        labels=[c for c in GENRE_CLASSES if c in classes],
        target_names=[c[:28] for c in GENRE_CLASSES if c in classes],
        zero_division=0,
    ))

    _save_confusion_matrix(y_true, y_pred, classes,
                           f"AraPoemBERT | Genre (Macro-F1={macro_f1:.3f})",
                           "cm_arapoem_genre_v3.png")

    return {
        "model": "AraPoemBERT | Genre",
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "kappa": kappa,
        "n": len(y_true),
        "baseline_macro_f1": 0.0907,   # mBERT
        "baseline_label":    "mBERT",
    }


# ── B) Emotion1DCNN audio emotion ─────────────────────────────────────────────

def _load_mel(path: str, sr: int = 16_000, n_mels: int = 128,
              max_sec: float = 8.0) -> torch.Tensor:
    try:
        wav, _ = librosa.load(path, sr=sr, mono=True)
        max_len = int(max_sec * sr)
        if len(wav) > max_len:
            wav = wav[:max_len]
        mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=n_mels,
                                             fmax=8000)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_t  = torch.from_numpy(mel_db).float()   # (n_mels, T)
        # Pad / trim time axis to fixed length
        target_frames = int(max_sec * sr / 512) + 1  # hop_length default=512
        if mel_t.shape[1] < target_frames:
            mel_t = torch.nn.functional.pad(
                mel_t, (0, target_frames - mel_t.shape[1]))
        else:
            mel_t = mel_t[:, :target_frames]
        return mel_t.unsqueeze(0)   # (1, n_mels, T)
    except Exception as e:
        logger.warning(f"  Mel failed for {path}: {e}")
        return None


def eval_audio_cnn(records: list[dict], device: str) -> dict:
    audio_cnn_config()
    ckpt = MODEL_DIR / "audio_cnn" / "audio_cnn_emotion_best.pt"
    if not ckpt.exists():
        logger.error(f"Checkpoint not found: {ckpt}")
        return {}

    logger.info("── B) Emotion1DCNN (audio emotion) ─────────────────────")
    logger.info(f"  Loading Emotion1DCNN + {ckpt.name}")

    model = Emotion1DCNN(num_classes=len(EMOTION_CLASSES))
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()

    y_true, y_pred = [], []
    y_true_str, y_pred_str = [], []
    audio_refs, text_refs, genres = [], [], []
    skipped = 0

    with torch.no_grad():
        for rec in records:
            audio_path   = rec.get("audio_filename", "")
            label_audio  = rec.get("emotion_audio",  "")
            label_text   = rec.get("emotion_text",   "")
            genre        = rec.get("genre_en",        "")

            if not audio_path or not label_audio:
                skipped += 1
                continue
            label_id = encode_emotion(label_audio)
            if label_id < 0:
                skipped += 1
                continue

            mel = _load_mel(audio_path)
            if mel is None:
                skipped += 1
                continue

            mel = mel.to(device)
            logits  = model(mel)
            pred_id = int(logits.argmax(-1).item())

            y_true.append(label_id)
            y_pred.append(pred_id)
            y_true_str.append(EMOTION_CLASSES[label_id])
            y_pred_str.append(EMOTION_CLASSES[pred_id])
            audio_refs.append(label_audio)
            text_refs.append(label_text)
            genres.append(genre)

    logger.info(f"  Evaluated {len(y_true)} records, skipped {skipped}")

    classes     = sorted(set(y_true_str + y_pred_str))
    macro_f1    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    kappa       = cohen_kappa_score(y_true, y_pred)

    # Emotion partial credit (Metrics v3)
    pc_scores = [
        emotion_partial_credit(p, a, t, g)
        for p, a, t, g in zip(y_pred_str, audio_refs, text_refs, genres)
    ]
    mean_pc = float(np.mean(pc_scores))

    logger.info(f"  Macro-F1        : {macro_f1:.4f}   (MFCC+SVM baseline: 0.0977)")
    logger.info(f"  Weighted-F1     : {weighted_f1:.4f}")
    logger.info(f"  Cohen's κ       : {kappa:.4f}")
    logger.info(f"  Emotion PC      : {mean_pc:.4f}   (MFCC+SVM baseline: 0.5273)")
    logger.info("\n" + classification_report(
        y_true_str, y_pred_str,
        labels=[c for c in EMOTION_CLASSES if c in classes],
        target_names=[c[:25] for c in EMOTION_CLASSES if c in classes],
        zero_division=0,
    ))

    _save_confusion_matrix(y_true_str, y_pred_str, classes,
                           f"Emotion1DCNN | Audio Emotion (Macro-F1={macro_f1:.3f})",
                           "cm_audio_cnn_v3.png")

    return {
        "model": "Emotion1DCNN | Emotion (audio)",
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "kappa": kappa,
        "emotion_partial_credit": mean_pc,
        "n": len(y_true),
        "baseline_macro_f1": 0.0977,
        "baseline_pc":       0.5273,
        "baseline_label":    "MFCC+SVM",
    }


# ── C) Whisper-small LoRA fine-tuned ─────────────────────────────────────────

def eval_whisper_finetuned(records: list[dict], device: str) -> dict:
    adapter_dir = MODEL_DIR / "whisper_nabati" / "best"
    if not adapter_dir.exists():
        logger.error(f"LoRA adapter not found: {adapter_dir}")
        return {}

    logger.info("── C) Whisper-small LoRA fine-tuned ────────────────────")
    logger.info(f"  Loading openai/whisper-small + LoRA adapter from {adapter_dir}")

    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from peft import PeftModel

    base_name = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(
        str(adapter_dir), language="Arabic", task="transcribe", local_files_only=True,
    )
    base_model = WhisperForConditionalGeneration.from_pretrained(base_name, local_files_only=True)
    model = PeftModel.from_pretrained(base_model, str(adapter_dir), local_files_only=True)
    model = model.merge_and_unload()   # merge LoRA weights into base for fast inference
    model.to(device)
    model.eval()

    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="arabic", task="transcribe"
    )

    wers, cers, soft_cers = [], [], []
    skipped = 0

    with torch.no_grad():
        for rec in records:
            audio_path = rec.get("audio_filename", "")
            reference  = str(rec.get("text_corrected", "")).strip()
            if not audio_path or not reference:
                skipped += 1
                continue

            try:
                wav, sr = librosa.load(audio_path, sr=16_000, mono=True)
                inputs  = processor.feature_extractor(
                    wav, sampling_rate=16_000, return_tensors="pt"
                ).to(device)
                pred_ids = model.generate(
                    inputs["input_features"],
                    max_new_tokens=128,
                )
                hypothesis = processor.tokenizer.decode(
                    pred_ids[0], skip_special_tokens=True
                ).strip()
            except Exception as e:
                logger.warning(f"  Inference failed for {audio_path}: {e}")
                skipped += 1
                continue

            wers.append(standard_wer(hypothesis, reference))
            cers.append(standard_cer(hypothesis, reference))
            soft_cers.append(soft_cer(hypothesis, reference))

    logger.info(f"  Evaluated {len(wers)} records, skipped {skipped}")

    mean_wer  = float(np.mean(wers))
    mean_cer  = float(np.mean(cers))
    mean_soft = float(np.mean(soft_cers))

    logger.info(f"  WER             : {mean_wer:.4f}   (zero-shot baseline: 0.272)")
    logger.info(f"  CER             : {mean_cer:.4f}   (zero-shot baseline: 0.085)")
    logger.info(f"  Soft-CER        : {mean_soft:.4f}  (zero-shot baseline: 0.083)")
    logger.info(
        f"  Soft/CER ratio  : {mean_soft/mean_cer:.3f}  "
        "(<1 = normalization-aware weighting lowers apparent error)"
    )

    return {
        "model": "Whisper-small LoRA | ASR",
        "wer":      mean_wer,
        "cer":      mean_cer,
        "soft_cer": mean_soft,
        "n": len(wers),
        "baseline_wer":      0.272,   # Whisper-small zero-shot, fixed poet-disjoint split, n=333
        "baseline_cer":      0.085,
        "baseline_soft_cer": 0.083,
        "baseline_label":    "Whisper-small zero-shot",
    }


# ── Confusion matrix helper ───────────────────────────────────────────────────

def _save_confusion_matrix(y_true, y_pred, classes, title, fname):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    short = [c.split("(")[0].strip()[:18] for c in classes]
    cm      = confusion_matrix(y_true, y_pred, labels=classes)
    cm_norm = np.nan_to_num(cm.astype(float) / cm.sum(axis=1, keepdims=True))

    n = len(classes)
    fig, ax = plt.subplots(figsize=(max(10, n), max(8, n - 1)))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=short, yticklabels=short,
                linewidths=0.3, ax=ax, cbar_kws={"label": "Row %"})
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True",      fontsize=10)
    ax.set_title(title, fontsize=10)
    plt.xticks(rotation=40, ha="right", fontsize=7)
    plt.yticks(rotation=0,  fontsize=7)
    fig.tight_layout()
    out = FIG_DIR / fname
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"  Confusion matrix → {out}")


# ── Final comparison table ────────────────────────────────────────────────────

def print_comparison(results: list[dict], baseline_csv: Path) -> None:
    logger.info("=" * 65)
    logger.info("  FINAL COMPARISON: TRAINED MODELS vs BASELINES")
    logger.info("=" * 65)

    for r in results:
        logger.info(f"\n  {r['model']}")
        if "macro_f1" in r:
            delta = r["macro_f1"] - r["baseline_macro_f1"]
            sign  = "+" if delta >= 0 else ""
            logger.info(f"  Macro-F1    : {r['macro_f1']:.4f}  "
                        f"({r['baseline_label']} baseline: {r['baseline_macro_f1']:.4f}, "
                        f"Δ={sign}{delta:.4f})")
        if "emotion_partial_credit" in r:
            delta = r["emotion_partial_credit"] - r["baseline_pc"]
            sign  = "+" if delta >= 0 else ""
            logger.info(f"  Emotion PC  : {r['emotion_partial_credit']:.4f}  "
                        f"(baseline: {r['baseline_pc']:.4f}, Δ={sign}{delta:.4f})")
        if "wer" in r:
            for metric, baseline_key, label in [
                ("wer",      "baseline_wer",      "WER     "),
                ("cer",      "baseline_cer",      "CER     "),
                ("soft_cer", "baseline_soft_cer", "Soft-CER"),
            ]:
                delta = r[metric] - r[baseline_key]
                sign  = "+" if delta >= 0 else ""
                logger.info(f"  {label}: {r[metric]:.4f}  "
                            f"(baseline: {r[baseline_key]:.4f}, Δ={sign}{delta:.4f})")

    # Save combined JSON
    out_path = REPORT_DIR / "trained_models_eval.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.success(f"\n  Full results → {out_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    logger.add("logs/eval_trained_models.log", rotation="10 MB")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"Device: {device}")

    records = load_test()
    logger.info(f"Test records: {len(records)}")

    only = args.only
    results = []

    if only in (None, "arapoem"):
        r = eval_arapoem_genre(records, device)
        if r:
            results.append(r)

    if only in (None, "audio") and not args.skip_audio:
        r = eval_audio_cnn(records, device)
        if r:
            results.append(r)

    if only in (None, "whisper") and not args.skip_whisper:
        r = eval_whisper_finetuned(records, device)
        if r:
            results.append(r)

    if results:
        print_comparison(results, REPORT_DIR / "baseline_results.csv")
    else:
        logger.warning("No results produced.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model checkpoints (Metrics v3)")
    parser.add_argument("--only", choices=["arapoem", "audio", "whisper"],
                        default=None, help="Run only one model evaluation")
    parser.add_argument("--skip-whisper", action="store_true",
                        help="Skip Whisper inference (slow on CPU)")
    parser.add_argument("--skip-audio", action="store_true",
                        help="Skip Audio CNN inference")
    parser.add_argument("--device", default=None,
                        help="Override device (cpu/mps/cuda)")
    main(parser.parse_args())

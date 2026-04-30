"""
Canonical poem-level emotion aggregation + fusion evaluation.

Outputs:
  - outputs/reports/poem_emotion_aggregation_eval.json
  - outputs/reports/emotion_fusion_eval.json
  - outputs/reports/poem_emotion_predictions_val.json
  - outputs/reports/poem_emotion_predictions_test.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from sklearn.metrics import f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.arousal_labels import AROUSAL_CLASSES
from src.data.labels import get_genre_expected_emotions, get_merged_emotion_classes
from src.models.emotion.aggregate import (
    aggregate_confidence_weighted,
    aggregate_logits_mean,
    aggregate_probs_mean,
    aggregate_topk_vote,
    build_poem_emotion_summary,
    group_by_poem_id,
)
from src.models.emotion.fusion import (
    apply_genre_constrained,
    apply_genre_prior,
    compute_delivery_metadata,
    decide_final_emotion,
    estimate_genre_emotion_prior,
    map_audio_emotion_to_core,
    map_text_emotion_to_core,
)
from src.evaluation.metrics import (
    emotion_partial_credit,
    expected_calibration_error,
    top_k_accuracy,
    emotion_ndcg_at_3,
    log_loss_safe,
    balanced_accuracy,
)
from src.models.audio_cnn import Emotion1DCNN
from scripts.demo import ArousalMLP, _extract_arousal_features


PROJECT_ROOT = Path(__file__).parent.parent
MODEL_NAME = "faisalq/bert-base-arapoembert"
GENRE_MODEL_NAME = "faisalq/bert-base-arapoembert"
EMOTION_CKPT = (
    PROJECT_ROOT / "outputs/models/arapoem_emotion/arapoem_emotion_text_best.pt"
)
AROUSAL_CKPT = PROJECT_ROOT / "outputs/models/arousal_mlp/arousal_mlp_arousal_best.pt"
AROUSAL_SCALER = PROJECT_ROOT / "outputs/models/arousal_mlp/arousal_scaler.pkl"
CNN_CKPT = PROJECT_ROOT / "outputs/models/audio_cnn/audio_cnn_emotion_best.pt"
TRAIN_PATH = PROJECT_ROOT / "data/processed/train.jsonl"
VAL_PATH = PROJECT_ROOT / "data/processed/val.jsonl"
TEST_PATH = PROJECT_ROOT / "data/processed/test.jsonl"
REPORT_DIR = PROJECT_ROOT / "outputs/reports"
MAX_SEQ_LEN = 32
DEFAULT_RUN_ID = "K1_merge_v1"
DEFAULT_MERGE_PROFILE = "rare_merge_v1"
DEFAULT_CONTEXT_WINDOW = 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate poem-level emotion aggregation and fusion."
    )
    p.add_argument(
        "--merge-profile",
        default=DEFAULT_MERGE_PROFILE,
        choices=["none", "rare_merge_v1"],
    )
    p.add_argument("--context-window", type=int, default=DEFAULT_CONTEXT_WINDOW)
    p.add_argument("--checkpoint", type=Path, default=EMOTION_CKPT)
    p.add_argument("--run-id", default=DEFAULT_RUN_ID)
    p.add_argument(
        "--genre-conditioning",
        choices=["none", "constrained", "prior", "all"],
        default="all",
    )
    p.add_argument(
        "--lambda-sweep", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0]
    )
    p.add_argument(
        "--poem-aggregation",
        choices=["mean", "conf_weighted", "logit_mean", "vote", "all"],
        default="all",
    )
    p.add_argument("--tau-text", type=float, default=0.45)
    p.add_argument("--tau-audio", type=float, default=0.55)
    return p.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def build_windowed_rows(
    rows: list[dict[str, Any]],
    context_window: int,
    merge_profile: str,
) -> list[dict[str, Any]]:
    grouped = group_by_poem_id(rows)
    built: list[dict[str, Any]] = []
    half = context_window // 2
    for poem_id, clips in grouped.items():
        ordered = sorted(clips, key=lambda row: int(row.get("start") or 0))
        for i, rec in enumerate(ordered):
            text_i = (rec.get("text_corrected") or "").strip()
            label = (rec.get("emotion_text") or "").strip()
            if not text_i or not label:
                continue
            if context_window > 1:
                parts: list[str] = []
                for k in range(max(0, i - half), i):
                    prev_text = (ordered[k].get("text_corrected") or "").strip()
                    if prev_text:
                        parts.append(prev_text)
                parts.append(text_i)
                for k in range(i + 1, min(len(ordered), i + 1 + half)):
                    next_text = (ordered[k].get("text_corrected") or "").strip()
                    if next_text:
                        parts.append(next_text)
                model_text = " ".join(parts)
            else:
                model_text = text_i

            gold_emotion_core = map_text_emotion_to_core(
                rec.get("emotion_text"), merge_profile
            )
            if gold_emotion_core is None:
                continue
            built.append(
                {
                    **rec,
                    "poem_id": poem_id,
                    "model_text": model_text,
                    "gold_emotion_core": gold_emotion_core,
                    "manual_genre": rec.get("genre_en", ""),
                    "merge_profile": merge_profile,
                }
            )
    return built


def load_text_emotion_model(device: torch.device, checkpoint: Path, num_labels: int):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.eval()
    return tokenizer, model


def batched_text_emotion_inference(
    rows: list[dict[str, Any]],
    tokenizer,
    model,
    labels: list[str],
    device: torch.device,
    batch_size: int = 32,
) -> None:
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        texts = [row["model_text"] for row in batch_rows]
        enc = tokenizer(
            texts,
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
        probs = F.softmax(logits, dim=-1).cpu().numpy()
        logits_np = logits.cpu().numpy()
        for row, prob, logit in zip(batch_rows, probs, logits_np):
            row["text_emotion_probs"] = prob.tolist()
            row["text_emotion_logits"] = logit.tolist()
            row["text_emotion_top1"] = labels[int(np.argmax(prob))]
            row["text_emotion_conf"] = float(np.max(prob))


def load_arousal_assets(device: torch.device):
    import pickle

    if not AROUSAL_CKPT.exists() or not AROUSAL_SCALER.exists():
        logger.warning("Arousal assets missing; full fusion will skip arousal.")
        return None, None
    scaler = pickle.load(open(AROUSAL_SCALER, "rb"))
    model = ArousalMLP().to(device)
    state = torch.load(AROUSAL_CKPT, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return scaler, model


def predict_arousal_probs(
    audio_path: Path, scaler, model, device: torch.device
) -> list[float] | None:
    if scaler is None or model is None:
        return None
    feats = _extract_arousal_features(audio_path)
    if feats is None:
        return None
    x = torch.tensor(scaler.transform(feats.reshape(1, -1)), dtype=torch.float32).to(
        device
    )
    with torch.no_grad():
        logits = model(x)
    return F.softmax(logits, dim=-1)[0].cpu().tolist()


def load_audio_emotion_model(device: torch.device):
    if not CNN_CKPT.exists():
        logger.warning(
            "Audio CNN checkpoint missing; full fusion will skip audio-emotion."
        )
        return None
    model = Emotion1DCNN().to(device)
    state = torch.load(CNN_CKPT, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_audio_emotion_probs(
    audio_path: Path, model, device: torch.device
) -> list[float] | None:
    if model is None:
        return None
    wav, _ = librosa.load(str(audio_path), sr=16000, mono=True)
    wav = wav[: 8 * 16000]
    mel = librosa.feature.melspectrogram(y=wav, sr=16000, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    target_len = 8 * 16000 // 512
    if mel_db.shape[1] < target_len:
        mel_db = np.pad(mel_db, ((0, 0), (0, target_len - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :target_len]
    x = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
    return F.softmax(logits, dim=-1)[0].cpu().tolist()


def majority_label(labels: list[str]) -> str:
    if not labels:
        return ""
    counts = Counter(labels)
    return counts.most_common(1)[0][0]


def _build_poem_metrics(
    true_ids: list[int],
    pred_ids: list[int],
    probs: list[list[float]],
    ndcg_scores: list[float],
    partial_scores: list[float],
    plausibility_hits: int,
    dms_hits: int,
    n_labels: int,
) -> dict[str, Any]:
    n = max(len(true_ids), 1)
    all_label_ids = list(range(n_labels))
    return {
        "hard_macro_f1": round(
            float(f1_score(true_ids, pred_ids, average="macro", zero_division=0)), 4
        ),
        "accuracy": round(
            sum(int(p == t) for p, t in zip(pred_ids, true_ids)) / n, 4
        ),
        "balanced_accuracy": round(balanced_accuracy(true_ids, pred_ids), 4),
        "ndcg_at_3": round(float(np.mean(ndcg_scores)) if ndcg_scores else 0.0, 4),
        "top3_accuracy": round(_poem_topk_accuracy(probs, true_ids, k=3), 4),
        "top2_accuracy": round(_poem_topk_accuracy(probs, true_ids, k=2), 4),
        "log_loss": round(log_loss_safe(true_ids, probs, labels=all_label_ids), 4),
        "ece": round(expected_calibration_error(probs, true_ids), 4),
        "mean_partial_credit": round(float(np.mean(partial_scores)), 4),
        "genre_plausibility_rate": round(plausibility_hits / n, 4),
        "dms_rate": round(dms_hits / n, 4),
        "n_poems": len(true_ids),
    }


def _poem_topk_accuracy(
    prob_rows: list[list[float]], true_ids: list[int], k: int
) -> float:
    if not prob_rows:
        return 0.0
    return sum(
        top_k_accuracy(prob, true, k=k) for prob, true in zip(prob_rows, true_ids)
    ) / len(true_ids)


def evaluate_aggregation_methods(
    rows: list[dict[str, Any]],
    labels: list[str],
    methods: list[str],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    grouped = group_by_poem_id(rows)
    report: dict[str, Any] = {
        "poem_level": {},
    }
    summaries_by_method: dict[str, dict[str, Any]] = {}
    for method in methods:
        poem_true_ids: list[int] = []
        poem_pred_ids: list[int] = []
        poem_prob_rows: list[list[float]] = []
        poem_summaries: dict[str, Any] = {}
        for poem_id, poem_rows in grouped.items():
            probs_by_clip = [
                np.array(row["text_emotion_probs"], dtype=np.float64)
                for row in poem_rows
            ]
            logits_by_clip = [
                np.array(row["text_emotion_logits"], dtype=np.float64)
                for row in poem_rows
            ]
            conf_by_clip = [float(row["text_emotion_conf"]) for row in poem_rows]
            if method == "mean":
                poem_probs = aggregate_probs_mean(probs_by_clip)
            elif method == "logit_mean":
                poem_probs = aggregate_logits_mean(logits_by_clip)
            elif method == "vote":
                poem_probs = aggregate_topk_vote(probs_by_clip, k=3)
            else:
                poem_probs = aggregate_confidence_weighted(probs_by_clip, conf_by_clip)

            summary = build_poem_emotion_summary(
                poem_id=poem_id,
                probs=poem_probs,
                probs_by_clip=probs_by_clip,
                labels=labels,
                method=method,
                clip_conf=conf_by_clip,
            )
            gold_label = majority_label(
                [str(row["gold_emotion_core"]) for row in poem_rows]
            )
            summary["gold_poem_emotion"] = gold_label
            poem_true_ids.append(labels.index(gold_label))
            poem_pred_ids.append(labels.index(summary["poem_emotion_raw_top1"]))
            poem_prob_rows.append(poem_probs.tolist())
            poem_summaries[poem_id] = summary

        poem_f1 = float(
            f1_score(poem_true_ids, poem_pred_ids, average="macro", zero_division=0)
        )
        poem_acc = sum(int(p == t) for p, t in zip(poem_pred_ids, poem_true_ids)) / max(
            len(poem_true_ids), 1
        )
        all_label_ids = list(range(len(labels)))
        report["poem_level"][method] = {
            "hard_macro_f1": round(poem_f1, 4),
            "accuracy": round(poem_acc, 4),
            "balanced_accuracy": round(
                balanced_accuracy(poem_true_ids, poem_pred_ids), 4
            ),
            "top2_accuracy": round(
                _poem_topk_accuracy(poem_prob_rows, poem_true_ids, k=2), 4
            ),
            "top3_accuracy": round(
                _poem_topk_accuracy(poem_prob_rows, poem_true_ids, k=3), 4
            ),
            "log_loss": round(
                log_loss_safe(poem_true_ids, poem_prob_rows, labels=all_label_ids), 4
            ),
            "ece": round(expected_calibration_error(poem_prob_rows, poem_true_ids), 4),
            "n_poems": len(poem_true_ids),
        }
        summaries_by_method[method] = poem_summaries
    return report, summaries_by_method


def select_best_aggregation_method(report: dict[str, Any]) -> str:
    method_metrics = report["poem_level"]
    preference = {"logit_mean": 3, "conf_weighted": 2, "mean": 1, "vote": 0}
    return max(
        method_metrics,
        key=lambda method: (
            method_metrics[method]["hard_macro_f1"],
            preference.get(method, -1),
        ),
    )


def select_best_fusion_variant(report: dict[str, Any]) -> str:
    systems = report["systems"]
    preference = {"full_fusion": 3, "genre_prior": 2, "genre_constrained": 1, "raw": 0}
    return max(
        systems,
        key=lambda name: (
            systems[name]["poem_metrics"]["hard_macro_f1"],
            systems[name]["poem_metrics"]["mean_partial_credit"],
            preference.get(name, -1),
        ),
    )


def build_poem_audio_summaries(
    grouped_rows: dict[str, list[dict[str, Any]]],
    device: torch.device,
    merge_profile: str,
) -> dict[str, dict[str, Any]]:
    scaler, arousal_model = load_arousal_assets(device)
    audio_model = load_audio_emotion_model(device)
    audio_labels_12 = [
        "Longing (Shawq)",
        "Delicate Love (Hub Raqeeq)",
        "Sorrow (Huzn)",
        "Pride (Fakhr)",
        "Admiration (I'jab)",
        "Contemplation (Ta'ammul)",
        "Disappointment (Khayba)",
        "Defiance (Tahaddi)",
        "Hope (Amal)",
        "Compassion (Hanaan)",
        "Humor (Turfah)",
        "Neutral / Descriptive (Wasfi)",
    ]
    summaries: dict[str, dict[str, Any]] = {}
    core_labels = get_merged_emotion_classes(merge_profile)
    for poem_id, poem_rows in grouped_rows.items():
        arousal_probs_by_clip: list[np.ndarray] = []
        audio_core_probs_by_clip: list[np.ndarray] = []
        mapped_audio_votes: list[str] = []
        for row in poem_rows:
            audio_path = Path(str(row.get("audio_filename") or ""))
            if not audio_path.exists():
                continue
            arousal_probs = predict_arousal_probs(
                audio_path, scaler, arousal_model, device
            )
            if arousal_probs is not None:
                arousal_probs_by_clip.append(np.array(arousal_probs, dtype=np.float64))
            audio_probs = predict_audio_emotion_probs(audio_path, audio_model, device)
            if audio_probs is not None:
                core_probs = np.zeros(len(core_labels), dtype=np.float64)
                for label12, prob in zip(audio_labels_12, audio_probs):
                    mapped = map_audio_emotion_to_core(label12, merge_profile)
                    if mapped is None:
                        continue
                    core_probs[core_labels.index(mapped)] += float(prob)
                if core_probs.sum() > 0:
                    core_probs /= core_probs.sum()
                    audio_core_probs_by_clip.append(core_probs)
                    mapped_audio_votes.append(core_labels[int(np.argmax(core_probs))])

        if arousal_probs_by_clip:
            arousal_poem_probs = aggregate_confidence_weighted(arousal_probs_by_clip)
            arousal_label = AROUSAL_CLASSES[int(np.argmax(arousal_poem_probs))]
            arousal_conf = float(np.max(arousal_poem_probs))
        else:
            arousal_poem_probs = None
            arousal_label = None
            arousal_conf = 0.0

        if audio_core_probs_by_clip:
            audio_poem_probs = aggregate_confidence_weighted(audio_core_probs_by_clip)
            audio_label = core_labels[int(np.argmax(audio_poem_probs))]
            audio_conf = float(np.max(audio_poem_probs))
        else:
            audio_poem_probs = None
            audio_label = None
            audio_conf = 0.0

        summaries[poem_id] = {
            "poem_arousal": arousal_label,
            "poem_arousal_confidence": round(arousal_conf, 6),
            "poem_arousal_probs": None
            if arousal_poem_probs is None
            else arousal_poem_probs.tolist(),
            "audio_emotion_poem_aux": audio_label,
            "audio_emotion_poem_aux_confidence": round(audio_conf, 6),
            "audio_emotion_poem_probs": None
            if audio_poem_probs is None
            else audio_poem_probs.tolist(),
            "audio_clip_support": dict(Counter(mapped_audio_votes)),
        }
    return summaries


def evaluate_fusion_variants(
    split_name: str,
    grouped_rows: dict[str, list[dict[str, Any]]],
    text_summaries: dict[str, dict[str, Any]],
    merge_profile: str,
    priors: dict[str, dict[str, float]],
    lam: float,
    tau_text: float,
    tau_audio: float,
    device: torch.device,
    audio_summaries: dict[str, dict[str, Any]] | None = None,
    include_full_fusion: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    labels = get_merged_emotion_classes(merge_profile)
    if audio_summaries is None:
        audio_summaries = (
            build_poem_audio_summaries(grouped_rows, device, merge_profile)
            if include_full_fusion
            else {}
        )
    systems: dict[str, dict[str, Any]] = {}
    detailed_predictions: dict[str, Any] = {}

    variant_defs: list[tuple[str, Any]] = [
        ("raw", lambda probs, genre: probs),
        (
            "genre_constrained",
            lambda probs, genre: apply_genre_constrained(
                probs, labels, genre, merge_profile
            ),
        ),
        (
            "genre_prior",
            lambda probs, genre: apply_genre_prior(probs, labels, genre, priors, lam),
        ),
    ]

    for variant_name, transform in variant_defs:
        poem_true_ids: list[int] = []
        poem_pred_ids: list[int] = []
        poem_probs: list[list[float]] = []
        plausibility_hits = 0
        partial_scores: list[float] = []
        ndcg_scores: list[float] = []
        dms_hits = 0
        per_poem_variant: dict[str, Any] = {}
        for poem_id, poem_rows in grouped_rows.items():
            text_summary = text_summaries[poem_id]
            full_raw_probs = np.array(
                [float(text_summary["poem_probabilities"][label]) for label in labels],
                dtype=np.float64,
            )

            genre = str(poem_rows[0].get("manual_genre") or "")
            conditioned = transform(full_raw_probs, genre)
            audio_summary = audio_summaries.get(poem_id, {})
            gold_label = majority_label(
                [str(row["gold_emotion_core"]) for row in poem_rows]
            )
            poem_true_ids.append(labels.index(gold_label))
            pred_label = labels[int(np.argmax(conditioned))]
            poem_pred_ids.append(labels.index(pred_label))
            poem_probs.append(conditioned.tolist())
            plausibility_hits += int(
                pred_label in set(get_genre_expected_emotions(genre, merge_profile))
            )
            partial_scores.append(
                emotion_partial_credit(
                    pred_label,
                    audio_summary.get("audio_emotion_poem_aux"),
                    gold_label,
                    genre,
                )
            )
            ndcg_scores.append(
                emotion_ndcg_at_3(
                    conditioned.tolist(),
                    gold_label,
                    audio_summary.get("audio_emotion_poem_aux") or "",
                    genre,
                    labels,
                )
            )
            delivery = compute_delivery_metadata(
                pred_label, audio_summary.get("poem_arousal")
            )
            dms_hits += int(delivery["dms_poem"])
            per_poem_variant[poem_id] = {
                **text_summary,
                "manual_genre": genre,
                "gold_poem_emotion": gold_label,
                "conditioned_top3": [
                    {
                        "label": labels[int(idx)],
                        "prob": round(float(conditioned[int(idx)]), 6),
                    }
                    for idx in np.argsort(conditioned)[::-1][:3]
                ],
                **audio_summary,
                **delivery,
                "predicted_poem_emotion": pred_label,
            }

        systems[variant_name] = {
            "poem_metrics": _build_poem_metrics(
                true_ids=poem_true_ids,
                pred_ids=poem_pred_ids,
                probs=poem_probs,
                ndcg_scores=ndcg_scores,
                partial_scores=partial_scores,
                plausibility_hits=plausibility_hits,
                dms_hits=dms_hits,
                n_labels=len(labels),
            ),
        }
        detailed_predictions[variant_name] = per_poem_variant

    # Full fusion starts from genre-prior conditioned text.
    if not include_full_fusion:
        best_variant = max(
            systems,
            key=lambda key: systems[key]["poem_metrics"]["hard_macro_f1"],
        )
        for name, payload in systems.items():
            payload["decision"] = "adopt" if name == best_variant else "discard"
        return {
            "split": split_name,
            "merge_profile": merge_profile,
            "systems": systems,
            "adopted_variant": best_variant,
            "lambda_prior": lam,
            "tau_text": tau_text,
            "tau_audio": tau_audio,
        }, detailed_predictions

    full_fusion_poem_true: list[int] = []
    full_fusion_poem_pred: list[int] = []
    full_fusion_probs: list[list[float]] = []
    full_partial_scores: list[float] = []
    full_ndcg_scores: list[float] = []
    full_dms_hits = 0
    full_plausibility_hits = 0
    per_poem_full: dict[str, Any] = {}
    for poem_id, poem_rows in grouped_rows.items():
        text_summary = text_summaries[poem_id]
        raw_probs = np.array(
            [float(text_summary["poem_probabilities"][label]) for label in labels],
            dtype=np.float64,
        )
        genre = str(poem_rows[0].get("manual_genre") or "")
        conditioned = apply_genre_prior(raw_probs, labels, genre, priors, lam)
        audio_summary = audio_summaries.get(poem_id, {})
        final = decide_final_emotion(
            text_summary=text_summary,
            conditioned_probs=conditioned,
            labels=labels,
            genre=genre,
            poem_arousal=audio_summary.get("poem_arousal"),
            audio_aux_label=audio_summary.get("audio_emotion_poem_aux"),
            audio_aux_conf=float(
                audio_summary.get("audio_emotion_poem_aux_confidence") or 0.0
            ),
            profile=merge_profile,
            tau_text=tau_text,
            tau_audio=tau_audio,
            strategy_name="genre_prior",
        )
        gold_label = majority_label(
            [str(row["gold_emotion_core"]) for row in poem_rows]
        )
        full_fusion_poem_true.append(labels.index(gold_label))
        full_fusion_poem_pred.append(labels.index(final["emotion_poem_final"]))
        full_fusion_probs.append(conditioned.tolist())
        full_partial_scores.append(
            emotion_partial_credit(
                final["emotion_poem_final"],
                audio_summary.get("audio_emotion_poem_aux"),
                gold_label,
                genre,
            )
        )
        full_ndcg_scores.append(
            emotion_ndcg_at_3(
                conditioned.tolist(),
                gold_label,
                audio_summary.get("audio_emotion_poem_aux") or "",
                genre,
                labels,
            )
        )
        full_dms_hits += int(final["dms_poem"])
        full_plausibility_hits += int(
            final["emotion_poem_final"]
            in set(get_genre_expected_emotions(genre, merge_profile))
        )
        per_poem_full[poem_id] = {
            **text_summary,
            **audio_summary,
            **final,
            "manual_genre": genre,
            "gold_poem_emotion": gold_label,
        }

    systems["full_fusion"] = {
        "poem_metrics": _build_poem_metrics(
            true_ids=full_fusion_poem_true,
            pred_ids=full_fusion_poem_pred,
            probs=full_fusion_probs,
            ndcg_scores=full_ndcg_scores,
            partial_scores=full_partial_scores,
            plausibility_hits=full_plausibility_hits,
            dms_hits=full_dms_hits,
            n_labels=len(labels),
        ),
    }
    detailed_predictions["full_fusion"] = per_poem_full

    best_variant = max(
        systems,
        key=lambda key: systems[key]["poem_metrics"]["hard_macro_f1"],
    )
    for name, payload in systems.items():
        payload["decision"] = "adopt" if name == best_variant else "discard"
    return {
        "split": split_name,
        "merge_profile": merge_profile,
        "systems": systems,
        "adopted_variant": best_variant,
        "lambda_prior": lam,
        "tau_text": tau_text,
        "tau_audio": tau_audio,
    }, detailed_predictions


def main() -> None:
    args = parse_args()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    logger.add(PROJECT_ROOT / "logs/evaluate_emotion_fusion.log", rotation="10 MB")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    labels = get_merged_emotion_classes(args.merge_profile)
    tokenizer, model = load_text_emotion_model(device, args.checkpoint, len(labels))

    train_rows = load_jsonl(TRAIN_PATH)
    val_rows = build_windowed_rows(
        load_jsonl(VAL_PATH), args.context_window, args.merge_profile
    )
    test_rows = build_windowed_rows(
        load_jsonl(TEST_PATH), args.context_window, args.merge_profile
    )
    batched_text_emotion_inference(val_rows, tokenizer, model, labels, device)
    batched_text_emotion_inference(test_rows, tokenizer, model, labels, device)

    methods = (
        ["mean", "conf_weighted", "logit_mean", "vote"]
        if args.poem_aggregation == "all"
        else [args.poem_aggregation]
    )
    aggregation_eval = {
        "run_id": args.run_id,
        "merge_profile": args.merge_profile,
        "context_window": args.context_window,
        "default_method": "conf_weighted",
        "splits": {},
    }
    val_agg, val_summaries = evaluate_aggregation_methods(val_rows, labels, methods)
    test_agg, test_summaries = evaluate_aggregation_methods(test_rows, labels, methods)
    aggregation_eval["splits"]["val"] = val_agg
    aggregation_eval["splits"]["test"] = test_agg
    adopted_method = select_best_aggregation_method(val_agg)
    aggregation_eval["default_method"] = adopted_method
    (REPORT_DIR / "poem_emotion_aggregation_eval.json").write_text(
        json.dumps(aggregation_eval, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    priors = estimate_genre_emotion_prior(train_rows, args.merge_profile)
    val_grouped = group_by_poem_id(val_rows)
    test_grouped = group_by_poem_id(test_rows)
    best_lam = args.lambda_sweep[0]
    best_f1 = -1.0
    for lam in args.lambda_sweep:
        candidate_report, _ = evaluate_fusion_variants(
            split_name="val",
            grouped_rows=val_grouped,
            text_summaries=val_summaries[adopted_method],
            merge_profile=args.merge_profile,
            priors=priors,
            lam=lam,
            tau_text=args.tau_text,
            tau_audio=args.tau_audio,
            device=device,
            include_full_fusion=False,
        )
        poem_f1 = candidate_report["systems"]["genre_prior"]["poem_metrics"][
            "hard_macro_f1"
        ]
        if poem_f1 > best_f1:
            best_f1 = poem_f1
            best_lam = lam

    val_audio_summaries = build_poem_audio_summaries(
        val_grouped, device, args.merge_profile
    )
    test_audio_summaries = build_poem_audio_summaries(
        test_grouped, device, args.merge_profile
    )
    val_report, val_predictions = evaluate_fusion_variants(
        split_name="val",
        grouped_rows=val_grouped,
        text_summaries=val_summaries[adopted_method],
        merge_profile=args.merge_profile,
        priors=priors,
        lam=best_lam,
        tau_text=args.tau_text,
        tau_audio=args.tau_audio,
        device=device,
        audio_summaries=val_audio_summaries,
    )
    test_report, test_predictions = evaluate_fusion_variants(
        split_name="test",
        grouped_rows=test_grouped,
        text_summaries=test_summaries[adopted_method],
        merge_profile=args.merge_profile,
        priors=priors,
        lam=best_lam,
        tau_text=args.tau_text,
        tau_audio=args.tau_audio,
        device=device,
        audio_summaries=test_audio_summaries,
    )
    adopted_variant = select_best_fusion_variant(val_report)
    for split_report in (val_report, test_report):
        split_report["adopted_variant"] = adopted_variant
        for name, payload in split_report["systems"].items():
            payload["decision"] = "adopt" if name == adopted_variant else "discard"

    fusion_report = {
        "run_id": args.run_id,
        "merge_profile": args.merge_profile,
        "context_window": args.context_window,
        "adopted_poem_aggregation": adopted_method,
        "adopted_variant": adopted_variant,
        "best_lambda_from_val": best_lam,
        "metric_tiers": {
            "primary": ["poem_macro_f1", "ndcg_at_3", "top3_accuracy"],
            "secondary": ["log_loss", "balanced_accuracy"],
            "constraint": [
                "partial_credit",
                "genre_plausibility_rate",
                "dms_rate",
                "ece",
            ],
        },
        "val": val_report,
        "test": test_report,
    }
    (REPORT_DIR / "emotion_fusion_eval.json").write_text(
        json.dumps(fusion_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (REPORT_DIR / "poem_emotion_predictions_val.json").write_text(
        json.dumps(val_predictions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (REPORT_DIR / "poem_emotion_predictions_test.json").write_text(
        json.dumps(test_predictions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.success(
        "Aggregation report → {}", REPORT_DIR / "poem_emotion_aggregation_eval.json"
    )
    logger.success("Fusion report → {}", REPORT_DIR / "emotion_fusion_eval.json")
    logger.success("Poem predictions (val/test) written for demo lookup.")


if __name__ == "__main__":
    main()

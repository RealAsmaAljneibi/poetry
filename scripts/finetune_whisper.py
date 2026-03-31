"""
scripts/finetune_whisper.py

Fine-tune openai/whisper-small on Nabati Arabic using human-corrected transcripts.

═══════════════════════════════════════════════════════════════════
WHY THE FIRST LoRA RUN FAILED (corrected-only regression analysis)
═══════════════════════════════════════════════════════════════════
Run 1 used --corrected-only: trained only on ~1,887 clips where
text_whisper ≠ text_corrected (i.e., where Whisper made mistakes).

This caused distribution shift + catastrophic forgetting:
  • Model saw ONLY hard dialect-gap cases, never easy/already-correct ones
  • Learned "everything needs fixing" → applied corrections even to correct output
  • Result: test WER 0.272 → 0.589 (catastrophic regression; 0.272 = fixed-split zero-shot baseline)

THE FIX:
  • Train on the FULL train split (all 2669 clips: correct + incorrect)
  • Balanced sampler: 50/50 mix of already-correct/needs-correction per batch
    so the model sees what correct Arabic looks like alongside what to fix
  • Lower LR: 1e-5 (was 1e-4, too aggressive for preservation)
  • OCR (Over-Correction Rate): new metric — fraction of previously-correct
    val clips that the model broke. If OCR rises, model is forgetting.
  • Early stop on val Soft-CER with patience=2; also warn if OCR > threshold.

DATA PROTOCOL (non-negotiable):
  • Train: train.jsonl only (audio → text_corrected)
  • Val:   val.jsonl   (tune/stop here)
  • Test:  test.jsonl  (final numbers, report only once)
  • text_whisper is a REFERENCE baseline, never a model input

Architecture:
  • whisper-small (241.7M) — approved exception 2026-03-02
  • LoRA: ~1.77M trainable (rank=16, q_proj+v_proj)

Course techniques:
  • LR warmup + cosine decay    (Week 2)
  • Label smoothing             (Week 3)
  • Gradient accumulation + clipping (Week 4 SSL lab)
  • TensorBoard, Pydantic config, structured eval (Week 5)

Usage:
    # Recommended: full train split, balanced sampler, LoRA
    uv run python scripts/finetune_whisper.py --use-lora

    # Ablation: corrected-only (for comparison, expected to regress)
    uv run python scripts/finetune_whisper.py --use-lora --corrected-only

    # Adjust balance ratio (default 0.5 = 50/50 mix)
    uv run python scripts/finetune_whisper.py --use-lora --mix-ratio 0.3
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import librosa
import numpy as np
import torch
from loguru import logger
from torch.utils.data import WeightedRandomSampler
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

# ── Project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import ASRConfig
from src.evaluation.metrics import (
    soft_cer as compute_soft_cer,
    standard_cer as compute_standard_cer,
    standard_wer as compute_standard_wer,
)
from src.training.trainer import set_seed

# ── jiwer metrics ─────────────────────────────────────────────────────────────
try:
    from jiwer import wer as _jiwer_wer, cer as _jiwer_cer

    def compute_wer(refs: list[str], hyps: list[str]) -> float:
        return float(_jiwer_wer(refs, hyps))

    def compute_cer(refs: list[str], hyps: list[str]) -> float:
        return float(_jiwer_cer(refs, hyps))
except ImportError:
    def compute_wer(refs, hyps):
        logger.warning("jiwer not installed — WER unavailable")
        return float("nan")
    def compute_cer(refs, hyps):
        logger.warning("jiwer not installed — CER unavailable")
        return float("nan")

# ── Constants ──────────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(__file__).parent.parent
SAMPLING_RATE   = 16_000
MAX_AUDIO_SECS  = 30
FEATURE_MAX_LEN = MAX_AUDIO_SECS * SAMPLING_RATE


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATASET
# ══════════════════════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════════════════════
# 1a. SPEC-AUGMENT
# ══════════════════════════════════════════════════════════════════════════════

def apply_spec_augment(
    features: np.ndarray,
    n_time_masks:  int = 2,
    time_mask_max: int = 80,
    n_freq_masks:  int = 2,
    freq_mask_max: int = 20,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """
    Apply SpecAugment to a single Whisper log-mel feature array.

    Shape expected: (n_mels, time_steps) — Whisper default is (80, 3000).

    Parameters follow the 'LD' variant from [Park et al., 2019]:
      time_mask_max=80   ≈ 0.8 s masked per mask (Whisper 100 fps)
      freq_mask_max=20   ≈ 25% of 80 mel bins per mask

    Validated for Arabic broadcast ASR in the QASR corpus [Al-Ali et al., 2021].
    Used here without SpecAugment's time-warping (W=0) to avoid distorting
    the prosodic timing cues that are important for Gulf poetry meter.
    """
    rng = rng or np.random.default_rng()
    x = features.copy()
    n_mels, n_frames = x.shape

    # Frequency masking
    for _ in range(n_freq_masks):
        f = int(rng.integers(0, freq_mask_max + 1))
        f0 = int(rng.integers(0, max(1, n_mels - f)))
        x[f0: f0 + f, :] = 0.0

    # Time masking
    for _ in range(n_time_masks):
        t = int(rng.integers(0, time_mask_max + 1))
        t0 = int(rng.integers(0, max(1, n_frames - t)))
        x[:, t0: t0 + t] = 0.0

    return x


class NabatiASRDataset(torch.utils.data.Dataset):
    """
    Loads (audio → Whisper log-mel features, text_corrected → token ids).

    Each record tracks:
      - is_correct=True/False to support:
      - Balanced sampling (WeightedRandomSampler)
      - OCR metric (Over-Correction Rate) on already-correct clips
      - source_name to support mixed-domain replay sampling

    corrected_only=True: ABLATION ONLY — causes distribution shift and
    catastrophic forgetting. Do NOT use for main training. See module docstring.

    Important safety rule:
      - examples whose known duration would exceed Whisper's 30s limit are skipped
        rather than truncated, because truncating audio while keeping the full
        transcript creates impossible supervision targets.

    spec_augment=True: applies SpecAugment (time + frequency masking) to each
    training sample's log-mel features. Proven for Arabic broadcast ASR in
    QASR [Al-Ali et al., 2021] and recommended by the deep-research report.
    """

    def __init__(
        self,
        jsonl_path:     Path,
        processor:      WhisperProcessor,
        corrected_only: bool = False,
        speed_factors:  Sequence[float] | None = None,
        spec_augment:   bool = False,
        spec_time_masks: int = 2,
        spec_time_max:   int = 80,
        spec_freq_masks: int = 2,
        spec_freq_max:   int = 20,
        source_name: str = "nabati",
    ):
        self.processor = processor
        self.records: list[dict] = []
        self.speed_factors   = tuple(float(x) for x in (speed_factors or (1.0,)))
        self.spec_augment    = spec_augment
        self.spec_time_masks = spec_time_masks
        self.spec_time_max   = spec_time_max
        self.spec_freq_masks = spec_freq_masks
        self.spec_freq_max   = spec_freq_max
        self._rng = np.random.default_rng()

        n_correct = 0
        n_incorrect = 0
        n_skipped_overlong = 0

        for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            audio_path     = Path(rec.get("audio_filename", ""))
            text_corrected = (rec.get("text_corrected") or "").strip()
            text_whisper   = (rec.get("text_whisper")   or "").strip()
            audio_offset_sec = float(rec.get("audio_offset_sec") or 0.0)
            audio_duration_sec = rec.get("audio_duration_sec")
            if audio_duration_sec is not None:
                audio_duration_sec = float(audio_duration_sec)

            if not audio_path.exists() or not text_corrected:
                continue

            is_correct = (text_whisper == text_corrected)

            if corrected_only and is_correct:
                continue

            for speed_factor in self.speed_factors:
                effective_duration_sec = (
                    (audio_duration_sec / speed_factor)
                    if audio_duration_sec is not None else None
                )
                if effective_duration_sec is not None and effective_duration_sec > MAX_AUDIO_SECS:
                    n_skipped_overlong += 1
                    continue
                self.records.append({
                    "audio_path":     audio_path,
                    "audio_offset_sec": audio_offset_sec,
                    "audio_duration_sec": audio_duration_sec,
                    "text_corrected": text_corrected,
                    "is_correct":     is_correct,
                    "speed_factor":   speed_factor,
                    "source_name":    source_name,
                })
            if is_correct:
                n_correct += 1
            else:
                n_incorrect += 1

        logger.info(
            f"  {jsonl_path.name} [{source_name}]: {len(self.records)} clips — "
            f"{n_correct} already-correct, {n_incorrect} needs-correction"
            + (f" | speed={self.speed_factors}" if self.speed_factors != (1.0,) else "")
            + (" | SpecAugment=ON" if self.spec_augment else "")
            + (f" | skipped_overlong={n_skipped_overlong}" if n_skipped_overlong else "")
        )
        if corrected_only:
            logger.warning(
                "  --corrected-only active: training on BIASED subset. "
                "This is an ablation mode — expect catastrophic forgetting."
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        audio, _ = librosa.load(
            str(rec["audio_path"]),
            sr=SAMPLING_RATE,
            mono=True,
            offset=float(rec.get("audio_offset_sec", 0.0)),
            duration=rec.get("audio_duration_sec"),
        )
        speed_factor = float(rec.get("speed_factor", 1.0))
        if speed_factor != 1.0:
            audio = librosa.effects.time_stretch(audio, rate=speed_factor)
        if len(audio) > FEATURE_MAX_LEN:
            raise ValueError(
                "Loaded audio exceeds Whisper 30s limit after dataset filtering. "
                "This would create an audio/transcript mismatch; skip or segment such clips."
            )

        inputs = self.processor.feature_extractor(
            audio, sampling_rate=SAMPLING_RATE, return_tensors="np",
        )
        feats = inputs.input_features[0]   # shape (80, 3000)

        if self.spec_augment:
            feats = apply_spec_augment(
                feats,
                n_time_masks  = self.spec_time_masks,
                time_mask_max = self.spec_time_max,
                n_freq_masks  = self.spec_freq_masks,
                freq_mask_max = self.spec_freq_max,
                rng           = self._rng,
            )

        labels = self.processor.tokenizer(
            rec["text_corrected"], return_tensors="np",
        ).input_ids[0]

        return {
            "input_features": feats,
            "labels":         labels,
        }

    def make_balanced_sampler(self, mix_ratio: float = 0.5) -> WeightedRandomSampler:
        """
        WeightedRandomSampler so each batch contains ~mix_ratio already-correct
        clips and ~(1-mix_ratio) needs-correction clips.

        mix_ratio=0.5: equal representation of both types per batch.
        This prevents the model from learning "always correct" (forgetting) or
        "always fix" (over-correction). Both error types are equally represented.
        """
        correct_idx   = [i for i, r in enumerate(self.records) if r["is_correct"]]
        incorrect_idx = [i for i, r in enumerate(self.records) if not r["is_correct"]]

        n_correct   = len(correct_idx)
        n_incorrect = len(incorrect_idx)

        if n_correct == 0 or n_incorrect == 0:
            logger.warning("Only one type of sample — balanced sampler disabled.")
            return None

        # Weight each sample so its expected count in a batch matches mix_ratio
        # weight_correct * n_correct  = mix_ratio * N
        # weight_incorrect * n_incorrect = (1-mix_ratio) * N
        N = len(self.records)
        w_correct   = (mix_ratio   * N) / n_correct
        w_incorrect = ((1 - mix_ratio) * N) / n_incorrect

        weights = torch.zeros(N)
        for i in correct_idx:
            weights[i] = w_correct
        for i in incorrect_idx:
            weights[i] = w_incorrect

        logger.info(
            f"  Balanced sampler: mix_ratio={mix_ratio:.2f} | "
            f"w_correct={w_correct:.2f} (×{n_correct}) | "
            f"w_incorrect={w_incorrect:.2f} (×{n_incorrect})"
        )
        return WeightedRandomSampler(
            weights=weights,
            num_samples=N,
            replacement=True,
        )

    def make_replay_sampler(
        self,
        primary_ratio: float,
        primary_source: str = "nabati",
        replay_source: str = "sada_replay",
        primary_correct_ratio: float | None = 0.5,
        epoch_num_samples: int | None = None,
    ) -> WeightedRandomSampler | None:
        """
        WeightedRandomSampler that mixes domains explicitly:

        - `primary_ratio` of samples come from the main Nabati dataset
        - `1-primary_ratio` come from the replay dataset (e.g. filtered SADA)

        Within the primary domain, optionally preserve the existing already-correct vs
        needs-correction balancing via `primary_correct_ratio`.
        """
        if not (0.0 < primary_ratio < 1.0):
            raise ValueError("primary_ratio must be in (0, 1) for replay sampling.")

        primary_correct_idx = [
            i for i, r in enumerate(self.records)
            if r.get("source_name") == primary_source and r.get("is_correct")
        ]
        primary_incorrect_idx = [
            i for i, r in enumerate(self.records)
            if r.get("source_name") == primary_source and not r.get("is_correct")
        ]
        replay_idx = [
            i for i, r in enumerate(self.records)
            if r.get("source_name") == replay_source
        ]

        n_primary_correct = len(primary_correct_idx)
        n_primary_incorrect = len(primary_incorrect_idx)
        n_replay = len(replay_idx)
        n_primary = n_primary_correct + n_primary_incorrect

        if n_primary == 0 or n_replay == 0:
            logger.warning("Replay sampler requested but one domain is empty — replay disabled.")
            return None

        N = len(self.records)
        replay_ratio = 1.0 - primary_ratio
        weights = torch.zeros(N)

        if (
            primary_correct_ratio is not None
            and 0.0 < primary_correct_ratio < 1.0
            and n_primary_correct > 0
            and n_primary_incorrect > 0
        ):
            w_primary_correct = (primary_ratio * primary_correct_ratio * N) / n_primary_correct
            w_primary_incorrect = (primary_ratio * (1.0 - primary_correct_ratio) * N) / n_primary_incorrect
            for i in primary_correct_idx:
                weights[i] = w_primary_correct
            for i in primary_incorrect_idx:
                weights[i] = w_primary_incorrect
        else:
            w_primary = (primary_ratio * N) / n_primary
            for i in primary_correct_idx + primary_incorrect_idx:
                weights[i] = w_primary
            w_primary_correct = w_primary
            w_primary_incorrect = w_primary

        w_replay = (replay_ratio * N) / n_replay
        for i in replay_idx:
            weights[i] = w_replay

        if epoch_num_samples is None:
            epoch_num_samples = n_primary

        logger.info(
            "  Replay sampler: "
            f"primary_ratio={primary_ratio:.2f} replay_ratio={replay_ratio:.2f} | "
            f"primary_correct={n_primary_correct} primary_incorrect={n_primary_incorrect} replay={n_replay} | "
            f"w_primary_correct={w_primary_correct:.2f} "
            f"w_primary_incorrect={w_primary_incorrect:.2f} "
            f"w_replay={w_replay:.2f} | "
            f"epoch_samples={epoch_num_samples}"
        )
        return WeightedRandomSampler(
            weights=weights,
            num_samples=epoch_num_samples,
            replacement=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 2. DATA COLLATOR
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DataCollatorSpeechSeq2Seq:
    """Pads features and labels; labels use -100 for padding (ignored by loss)."""
    processor: WhisperProcessor

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ══════════════════════════════════════════════════════════════════════════════
# 3. LoRA SETUP
# ══════════════════════════════════════════════════════════════════════════════

def apply_lora(
    model,
    rank: int,
    alpha: int,
    dropout: float,
    target_mods: list[str] | str,
):
    """Apply LoRA adapters. Only adapter params (~1.77M) are trained."""
    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        task_type      = TaskType.SEQ_2_SEQ_LM,
        r              = rank,
        lora_alpha     = alpha,
        lora_dropout   = dropout,
        target_modules = target_mods,
        bias           = "none",
    )
    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    logger.info(
        f"  LoRA: trainable={trainable/1e6:.2f}M / {total/1e6:.1f}M  "
        f"rank={rank}  alpha={alpha}  targets={target_mods}"
    )
    return model


def resolve_lora_targets(
    model,
    requested_targets: Sequence[str],
    decoder_only: bool = False,
) -> list[str]:
    """Expand LoRA targets to exact module names when using decoder-only adapters."""
    if not decoder_only:
        return list(requested_targets)

    matched = [
        name
        for name, _ in model.named_modules()
        if name.startswith("model.decoder.layers.")
        and any(name.endswith(f".{target}") for target in requested_targets)
    ]
    if not matched:
        raise ValueError(
            f"No decoder attention modules matched requested LoRA targets: {requested_targets}"
        )

    logger.info(f"  Decoder-only LoRA: targeting {len(matched)} decoder projection modules")
    return matched


def freeze_whisper_encoder(model) -> None:
    """Freeze all encoder parameters to preserve acoustic representations."""
    frozen_params = 0
    for name, param in model.named_parameters():
        if name.startswith("model.encoder."):
            param.requires_grad = False
            frozen_params += param.numel()
    logger.info(f"  Encoder frozen: {frozen_params / 1e6:.2f}M parameters")


def load_whisper_assets(model_name: str, language: str):
    """Load Whisper processor/model strictly from local files for offline reproducibility."""
    kwargs = {"language": language, "task": "transcribe"}
    try:
        processor = WhisperProcessor.from_pretrained(
            model_name,
            local_files_only=True,
            **kwargs,
        )
        model = WhisperForConditionalGeneration.from_pretrained(
            model_name,
            local_files_only=True,
        )
        logger.info("  Loaded Whisper from local cache")
        return processor, model
    except Exception as exc:
        raise RuntimeError(
            f"Could not load Whisper locally from '{model_name}': {exc}. "
            "Run `just cache-models` first and keep inference/training fully offline."
        ) from exc


# ══════════════════════════════════════════════════════════════════════════════
# 4. BALANCED TRAINER
# ══════════════════════════════════════════════════════════════════════════════

class BalancedSeq2SeqTrainer(Seq2SeqTrainer):
    """
    Seq2SeqTrainer subclass that injects a WeightedRandomSampler so each
    training batch contains a balanced mix of already-correct and
    needs-correction clips.

    Without this, batches drawn from the full train split are 70% needs-correction
    (1887 clips) and 30% already-correct (782 clips), which still risks mild
    over-correction bias. With mix_ratio=0.5, both types appear equally.
    """
    def __init__(self, *args, balanced_sampler=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._balanced_sampler = balanced_sampler

    def get_train_dataloader(self):
        if self._balanced_sampler is None:
            return super().get_train_dataloader()

        from torch.utils.data import DataLoader
        return DataLoader(
            self.train_dataset,
            batch_size  = self.args.per_device_train_batch_size,
            sampler     = self._balanced_sampler,
            collate_fn  = self.data_collator,
            num_workers = self.args.dataloader_num_workers,
            pin_memory  = self.args.dataloader_pin_memory,
        )


# ══════════════════════════════════════════════════════════════════════════════
# 5. OCR CALLBACK
# ══════════════════════════════════════════════════════════════════════════════

class OCRCallback(TrainerCallback):
    """
    Over-Correction Rate (OCR) monitor.

    After each eval, runs inference on val clips that were ALREADY CORRECT
    before fine-tuning (text_whisper == text_corrected). If the fine-tuned
    model makes these worse, OCR rises — a direct signal of catastrophic
    forgetting / over-correction.

    OCR = fraction of previously-correct clips now transcribed beyond a small
    tolerance, using WER/CER thresholds instead of exact string equality.

    Threshold: warn if OCR > ocr_threshold (default 0.15 = 15%).
    If OCR rises consistently, consider stopping early.
    """

    def __init__(
        self,
        processor:      WhisperProcessor,
        val_jsonl:      Path,
        device:         str,
        ocr_threshold:  float = 0.15,
        ocr_wer_threshold: float = 0.10,
        ocr_cer_threshold: float = 0.05,
        stop_on_threshold: bool = False,
    ):
        self.processor     = processor
        self.device        = device
        self.ocr_threshold = ocr_threshold
        self.ocr_wer_threshold = ocr_wer_threshold
        self.ocr_cer_threshold = ocr_cer_threshold
        self.stop_on_threshold = stop_on_threshold

        # Load already-correct val clips (text_whisper == text_corrected)
        self.correct_clips: list[dict] = []
        for line in val_jsonl.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            tw = (rec.get("text_whisper")   or "").strip()
            tc = (rec.get("text_corrected") or "").strip()
            ap = Path(rec.get("audio_filename", ""))
            offset_sec = float(rec.get("audio_offset_sec") or 0.0)
            duration_sec = rec.get("audio_duration_sec")
            if duration_sec is not None:
                duration_sec = float(duration_sec)
            if tw == tc and tc and ap.exists():
                self.correct_clips.append({
                    "audio_path": ap,
                    "audio_offset_sec": offset_sec,
                    "audio_duration_sec": duration_sec,
                    "text_corrected": tc,
                })

        logger.info(
            f"OCRCallback: monitoring {len(self.correct_clips)} "
            f"already-correct val clips (OCR threshold={ocr_threshold:.0%}, "
            f"WER<={ocr_wer_threshold:.2f} or CER<={ocr_cer_threshold:.2f})"
        )

    def on_evaluate(self, args, state, control, model, **kwargs):
        if not self.correct_clips:
            return

        model.eval()
        n_broken = 0

        for clip in self.correct_clips:
            try:
                audio, _ = librosa.load(
                    str(clip["audio_path"]),
                    sr=SAMPLING_RATE,
                    mono=True,
                    offset=float(clip.get("audio_offset_sec", 0.0)),
                    duration=clip.get("audio_duration_sec"),
                )
                audio = audio[:FEATURE_MAX_LEN]
                feats = self.processor.feature_extractor(
                    audio, sampling_rate=SAMPLING_RATE, return_tensors="pt"
                ).input_features.to(self.device)

                with torch.no_grad():
                    pred_ids = model.generate(
                        input_features=feats, language="ar", task="transcribe"
                    )
                hyp = self.processor.tokenizer.decode(
                    pred_ids[0], skip_special_tokens=True
                ).strip()

                wer = compute_standard_wer(hyp, clip["text_corrected"])
                cer = compute_standard_cer(hyp, clip["text_corrected"])
                if not (wer <= self.ocr_wer_threshold or cer <= self.ocr_cer_threshold):
                    n_broken += 1
            except Exception:
                pass

        ocr = n_broken / max(len(self.correct_clips), 1)
        state.log_history[-1]["ocr"] = round(ocr, 4)

        msg = (
            f"OCR={ocr:.1%}  ({n_broken}/{len(self.correct_clips)} "
            f"previously-correct clips exceeded WER/CER tolerance)"
        )
        if ocr > self.ocr_threshold:
            logger.warning(f"  [OCR WARNING] {msg} — model is over-correcting!")
            if self.stop_on_threshold:
                control.should_training_stop = True
                logger.warning(
                    f"  [OCR STOP] stopping early because OCR exceeded {self.ocr_threshold:.0%}"
                )
        else:
            logger.info(f"  [OCR OK]      {msg}")
        return control


class BaselineWERGateCallback(TrainerCallback):
    """
    Stops training if eval WER degrades more than `max_delta` above the zero-shot baseline.

    Hypothesis: if fine-tuning worsens WER beyond the baseline by a meaningful margin at the
    first eval point, continuing is unlikely to recover. This is a conservative early-exit
    to avoid wasting compute on runs that are already regressing.

    This does NOT guarantee that runs that pass the gate will converge; it only terminates
    clearly regressing runs early.
    """

    def __init__(self, baseline_val_wer: float, max_delta: float = 0.05):
        self.baseline_val_wer = baseline_val_wer
        self.max_delta = max_delta

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return control
        eval_wer = metrics.get("eval_wer", None)
        if eval_wer is None:
            return control
        threshold = self.baseline_val_wer + self.max_delta
        if eval_wer > threshold:
            logger.warning(
                f"  [WER-GATE STOP] eval_wer={eval_wer:.3f} > baseline+delta "
                f"({self.baseline_val_wer:.3f}+{self.max_delta:.3f}={threshold:.3f}). "
                "Stopping training — run is regressing beyond acceptable tolerance."
            )
            control.should_training_stop = True
        else:
            logger.info(
                f"  [WER-GATE OK]  eval_wer={eval_wer:.3f} within delta of baseline "
                f"({threshold:.3f})."
            )
        return control


# ══════════════════════════════════════════════════════════════════════════════
# 6. METRICS
# ══════════════════════════════════════════════════════════════════════════════

def make_compute_metrics(tokenizer, cfg: ASRConfig):
    """Returns compute_metrics closure for Seq2SeqTrainer (WER + CER + Soft-CER)."""

    def compute_metrics(pred) -> dict[str, float]:
        pred_ids  = pred.predictions
        label_ids = pred.label_ids.copy()
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        hyps = [s.strip() for s in tokenizer.batch_decode(pred_ids,  skip_special_tokens=True)]
        refs = [s.strip() for s in tokenizer.batch_decode(label_ids, skip_special_tokens=True)]

        val_wer      = compute_wer(refs, hyps)
        val_cer      = compute_cer(refs, hyps)
        val_soft_cer = float(np.mean([compute_soft_cer(h, r) for h, r in zip(hyps, refs)]))

        if val_wer > cfg.wer_warn_threshold:
            logger.warning(f"  val WER={val_wer:.3f} > threshold {cfg.wer_warn_threshold}")
        if val_cer > cfg.cer_warn_threshold:
            logger.warning(f"  val CER={val_cer:.3f} > threshold {cfg.cer_warn_threshold}")

        logger.info(
            f"  val WER={val_wer:.4f}  CER={val_cer:.4f}  "
            f"Soft-CER={val_soft_cer:.4f} [research-informed diagnostic]"
        )
        return {"wer": val_wer, "cer": val_cer, "soft_cer": val_soft_cer}

    return compute_metrics


# ══════════════════════════════════════════════════════════════════════════════
# 7. BASELINE EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_baseline(jsonl_path: Path) -> dict[str, float]:
    """WER/CER/Soft-CER of vanilla Whisper-small from stored text_whisper field."""
    refs, hyps = [], []
    for line in Path(jsonl_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        ref = (rec.get("text_corrected") or "").strip()
        hyp = (rec.get("text_whisper")   or "").strip()
        if ref and hyp:
            refs.append(ref)
            hyps.append(hyp)

    if not refs:
        return {"wer": float("nan"), "cer": float("nan"), "soft_cer": float("nan"), "n": 0}

    return {
        "wer":      compute_wer(refs, hyps),
        "cer":      compute_cer(refs, hyps),
        "soft_cer": float(np.mean([compute_soft_cer(h, r) for h, r in zip(hyps, refs)])),
        "n": len(refs),
    }


def resolve_jsonl_arg(raw_path: str) -> Path:
    """Resolve a CLI JSONL path relative to the project root when needed."""
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


# ══════════════════════════════════════════════════════════════════════════════
# 8. ARGUMENT PARSING
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Fine-tune Whisper on Nabati Arabic. "
            "Train on full TRAIN split (not corrected-only) to prevent over-correction."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    p.add_argument("--model", default="openai/whisper-small",
                   help="whisper-small=241.7M (approved exception 2026-03-02). "
                        "Do NOT use whisper-medium or larger.")
    p.add_argument("--train-jsonl", default="data/processed/train.jsonl",
                   help="Training split JSONL. Override for SADA or other curricula.")
    p.add_argument("--val-jsonl", default="data/processed/val.jsonl",
                   help="Validation split JSONL.")
    p.add_argument("--test-jsonl", default="data/processed/test.jsonl",
                   help="Held-out evaluation JSONL.")
    p.add_argument(
        "--replay-jsonl",
        default=None,
        help=(
            "Optional replay-domain JSONL mixed into training only "
            "(e.g. filtered SADA for anti-forgetting replay training)."
        ),
    )
    p.add_argument(
        "--replay-ratio",
        type=float,
        default=0.0,
        help=(
            "Fraction of training samples drawn from --replay-jsonl. "
            "Example: 0.25 gives 75%% primary Nabati + 25%% replay."
        ),
    )

    # ── LoRA ──────────────────────────────────────────────────────────────────
    p.add_argument("--use-lora",     action="store_true")
    p.add_argument("--lora-rank",    type=int,   default=16)
    p.add_argument("--lora-alpha",   type=int,   default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-targets", nargs="+",  default=["q_proj", "v_proj"])
    p.add_argument(
        "--decoder-only-lora",
        action="store_true",
        help="Restrict LoRA adapters to decoder attention projections only.",
    )
    p.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="Freeze all Whisper encoder parameters before training.",
    )
    p.add_argument(
        "--speed-perturb",
        nargs="+",
        type=float,
        default=None,
        help="Optional in-memory speed factors applied to each training clip, e.g. 0.9 1.0 1.1.",
    )
    p.add_argument(
        "--spec-augment",
        action="store_true",
        help=(
            "Apply SpecAugment (time+frequency masking) to training log-mel features. "
            "Validated for Arabic ASR in QASR corpus [Al-Ali et al., 2021]. "
            "Defaults: 2 time masks (max 80 frames), 2 freq masks (max 20 bins). "
            "Applied to train set only; val/test are not augmented."
        ),
    )
    p.add_argument("--spec-time-masks", type=int, default=2,
                   help="Number of time masks for SpecAugment (default: 2).")
    p.add_argument("--spec-time-max",   type=int, default=80,
                   help="Max time frames masked per mask for SpecAugment (default: 80 ≈ 0.8s).")
    p.add_argument("--spec-freq-masks", type=int, default=2,
                   help="Number of frequency masks for SpecAugment (default: 2).")
    p.add_argument("--spec-freq-max",   type=int, default=20,
                   help="Max frequency bins masked per mask for SpecAugment (default: 20/80 bins).")

    # ── Training ──────────────────────────────────────────────────────────────
    p.add_argument("--epochs",        type=int,   default=5,
                   help="Fewer epochs + early stopping avoids overfitting on 2669 clips.")
    p.add_argument("--batch-size",    type=int,   default=4)
    p.add_argument("--grad-accum",    type=int,   default=8,
                   help="Effective batch = 4×8=32.")
    p.add_argument("--learning-rate", type=float, default=1e-5,
                   help="1e-5 (not 1e-4): lower LR preserves Whisper's existing Arabic knowledge.")
    p.add_argument(
        "--optimizer",
        choices=["adamw_torch", "adafactor"],
        default="adamw_torch",
        help="Optimizer family for Seq2SeqTrainer.",
    )
    p.add_argument(
        "--lr-scheduler-type",
        choices=["cosine", "linear", "constant", "constant_with_warmup", "cosine_with_restarts", "polynomial"],
        default="cosine",
        help="Learning-rate scheduler family.",
    )
    p.add_argument("--weight-decay", type=float, default=0.01,
                   help="Weight decay / L2 regularization strength.")
    p.add_argument("--warmup-ratio",  type=float, default=0.15,
                   help="15%% warmup: more warmup steps → smoother adaptation.")
    p.add_argument("--max-grad-norm", type=float, default=1.0,
                   help="Gradient clipping. Prevents large updates from corrupting weights.")
    p.add_argument("--patience",      type=int,   default=2,
                   help="Early stop on val Soft-CER. Patience=2 stops quickly on regression.")
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--mix-ratio",     type=float, default=0.5,
                   help="Fraction of already-correct clips per batch (balanced sampler). "
                        "0.5=50/50, 0.0=disable balanced sampler.")
    p.add_argument("--ocr-threshold", type=float, default=0.15,
                   help="Warn if Over-Correction Rate exceeds this on val.")
    p.add_argument("--ocr-wer-threshold", type=float, default=0.10,
                   help="Treat a previously-correct clip as still acceptable if WER stays at or below this threshold.")
    p.add_argument("--ocr-cer-threshold", type=float, default=0.05,
                   help="Treat a previously-correct clip as still acceptable if CER stays at or below this threshold.")
    p.add_argument(
        "--stop-on-ocr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Stop training immediately when OCR exceeds --ocr-threshold (default: on). "
             "Use --no-stop-on-ocr to disable (not recommended).",
    )
    p.add_argument(
        "--wer-delta-gate",
        type=float,
        default=0.05,
        help="Stop training if val WER exceeds the zero-shot baseline by more than this amount "
             "(default: 0.05). Set to a large value (e.g. 99) to disable.",
    )
    p.add_argument(
        "--best-model-metric",
        choices=["soft_cer", "wer", "cer"],
        default="wer",
        help="Validation metric used to select the best checkpoint.",
    )
    p.add_argument("--eval-steps", type=int, default=100,
                   help="Evaluate every N optimizer steps instead of only once per epoch.")
    p.add_argument("--save-steps", type=int, default=100,
                   help="Save checkpoints every N optimizer steps.")

    # ── Ablation: corrected-only (DISABLED BY DEFAULT — causes regression) ────
    p.add_argument("--corrected-only", action="store_true",
                   help="ABLATION ONLY: train on dialect-gap clips only. "
                        "Known to cause catastrophic forgetting (WER 0.272→0.589 on fixed split). "
                        "Do not use for main training.")

    p.add_argument("--output-dir", default="outputs/models/whisper_nabati")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# 9. MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.decoder_only_lora and not args.use_lora:
        raise ValueError("--decoder-only-lora requires --use-lora")
    if args.save_steps % args.eval_steps != 0:
        raise ValueError("--save-steps must be a multiple of --eval-steps when load_best_model_at_end=True")
    if not (0.0 <= args.replay_ratio < 1.0):
        raise ValueError("--replay-ratio must be in [0, 1).")
    if args.replay_ratio > 0.0 and not args.replay_jsonl:
        raise ValueError("--replay-jsonl is required when --replay-ratio > 0.")
    if args.replay_jsonl and args.corrected_only:
        raise ValueError("--corrected-only and replay training should not be combined.")

    cfg = ASRConfig(
        model_name       = args.model,
        language         = "ar",
        max_audio_sec    = 30,
        epochs           = args.epochs,
        batch_size       = args.batch_size,
        learning_rate    = args.learning_rate,
        weight_decay     = args.weight_decay,
        warmup_ratio     = args.warmup_ratio,
        patience         = args.patience,
        train_jsonl      = resolve_jsonl_arg(args.train_jsonl),
        val_jsonl        = resolve_jsonl_arg(args.val_jsonl),
        test_jsonl       = resolve_jsonl_arg(args.test_jsonl),
        output_dir       = PROJECT_ROOT / args.output_dir,
        grad_accum_steps = args.grad_accum,
        label_smoothing  = args.label_smoothing,
    )

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = PROJECT_ROOT / "logs" / "finetune_whisper.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_file), level="INFO", rotation="10 MB")

    logger.info("=" * 65)
    logger.info(f"Whisper fine-tune | model={cfg.model_name}")
    logger.info(f"  corrected_only={args.corrected_only}  "
                f"(WARNING: ablation mode — see docstring)" if args.corrected_only
                else "  Training on FULL train split (corrected + already-correct)")
    logger.info(f"  LR={cfg.learning_rate:.0e}  batch={cfg.batch_size}×{cfg.grad_accum_steps}"
                f"  mix_ratio={args.mix_ratio:.2f}  max_grad_norm={args.max_grad_norm}")
    logger.info(
        f"  optimizer={args.optimizer}  scheduler={args.lr_scheduler_type}  "
        f"weight_decay={cfg.weight_decay:.3g}  label_smoothing={cfg.label_smoothing:.3g}"
    )
    logger.info(
        f"  data train={cfg.train_jsonl.name}  val={cfg.val_jsonl.name}  test={cfg.test_jsonl.name}"
    )
    if args.replay_jsonl and args.replay_ratio > 0.0:
        logger.info(
            f"  replay train={Path(args.replay_jsonl).name}  "
            f"replay_ratio={args.replay_ratio:.2f}  primary_ratio={1.0 - args.replay_ratio:.2f}"
        )
    if args.speed_perturb:
        logger.info(f"  Speed perturbation factors: {tuple(args.speed_perturb)}")
    logger.info(
        f"  freeze_encoder={args.freeze_encoder}  decoder_only_lora={args.decoder_only_lora}  "
        f"best_model_metric={args.best_model_metric}"
    )

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"  Device: {device}")

    # ── Baseline ──────────────────────────────────────────────────────────────
    logger.info("\n── Baseline (whisper-small zero-shot on Nabati) ──")
    baseline_val  = evaluate_baseline(cfg.val_jsonl)
    baseline_test = evaluate_baseline(cfg.test_jsonl)
    baseline_available = baseline_val["n"] > 0 and baseline_test["n"] > 0
    if baseline_available:
        logger.info(f"  val  WER={baseline_val['wer']:.3f}  CER={baseline_val['cer']:.3f}  "
                    f"Soft-CER={baseline_val['soft_cer']:.3f}  n={baseline_val['n']}")
        logger.info(f"  test WER={baseline_test['wer']:.3f}  CER={baseline_test['cer']:.3f}  "
                    f"Soft-CER={baseline_test['soft_cer']:.3f}  n={baseline_test['n']}")
    else:
        logger.info(
            "  Baseline comparison unavailable for this dataset "
            "(expected for SADA JSONL without stored text_whisper fields)."
        )

    # ── Load Processor + Model ─────────────────────────────────────────────────
    logger.info(f"Loading {cfg.model_name} …")
    processor, model = load_whisper_assets(cfg.model_name, cfg.language)
    model.generation_config.language           = cfg.language
    model.generation_config.task               = "transcribe"
    model.generation_config.forced_decoder_ids = None

    if args.freeze_encoder:
        freeze_whisper_encoder(model)

    if args.use_lora:
        lora_targets = resolve_lora_targets(
            model,
            requested_targets=args.lora_targets,
            decoder_only=args.decoder_only_lora,
        )
        model = apply_lora(
            model,
            args.lora_rank,
            args.lora_alpha,
            args.lora_dropout,
            lora_targets,
        )
    else:
        total = sum(p.numel() for p in model.parameters())
        logger.info(f"  Full fine-tuning: {total/1e6:.1f}M params")

    model.to(device)

    # ── Datasets ──────────────────────────────────────────────────────────────
    if args.spec_augment:
        logger.info(
            f"  SpecAugment=ON  time_masks={args.spec_time_masks}×{args.spec_time_max}fr  "
            f"freq_masks={args.spec_freq_masks}×{args.spec_freq_max}bins"
        )
    train_ds = NabatiASRDataset(
        cfg.train_jsonl,
        processor,
        corrected_only   = args.corrected_only,
        speed_factors    = args.speed_perturb,
        spec_augment     = args.spec_augment,
        spec_time_masks  = args.spec_time_masks,
        spec_time_max    = args.spec_time_max,
        spec_freq_masks  = args.spec_freq_masks,
        spec_freq_max    = args.spec_freq_max,
        source_name      = "nabati",
    )
    replay_jsonl = resolve_jsonl_arg(args.replay_jsonl) if args.replay_jsonl else None

    # ── Deterministic SADA subset + 10 K total-items guard ────────────────────
    # Constraint: total training items (primary + replay) ≤ 10,000.
    # If replay is enabled we sample deterministically (seed=42) and write a
    # reproducible subset file so the exact records used are auditable.
    if replay_jsonl and args.replay_ratio > 0.0:
        n_primary = len(train_ds.records)
        max_replay = max(0, 10_000 - n_primary)
        if max_replay == 0:
            logger.warning(
                f"  [10K-GUARD] Primary set already has {n_primary} records (≥10 000). "
                "Replay disabled to stay within the 10 K training-item budget."
            )
            replay_jsonl = None
        else:
            import random as _random
            _rng = _random.Random(42)
            with open(replay_jsonl, "r", encoding="utf-8") as _fh:
                _all_lines = [ln for ln in _fh if ln.strip()]
            _rng.shuffle(_all_lines)
            _N = min(max_replay, len(_all_lines))
            _subset_lines = _all_lines[:_N]
            _subset_path = (
                Path(__file__).parent.parent
                / "data" / "processed" / f"sada_replay_{_N}.jsonl"
            )
            _subset_path.parent.mkdir(parents=True, exist_ok=True)
            with open(_subset_path, "w", encoding="utf-8") as _out:
                _out.writelines(_subset_lines)
            logger.info(
                f"  [10K-GUARD] primary={n_primary}  sada_full={len(_all_lines)}  "
                f"sada_subset_N={_N}  seed=42  "
                f"total={n_primary + _N}  subset_path={_subset_path}"
            )
            replay_jsonl = _subset_path

    if replay_jsonl and args.replay_ratio > 0.0:
        replay_ds = NabatiASRDataset(
            replay_jsonl,
            processor,
            corrected_only   = False,
            speed_factors    = args.speed_perturb,
            spec_augment     = args.spec_augment,
            spec_time_masks  = args.spec_time_masks,
            spec_time_max    = args.spec_time_max,
            spec_freq_masks  = args.spec_freq_masks,
            spec_freq_max    = args.spec_freq_max,
            source_name      = "sada_replay",
        )
        train_ds.records.extend(replay_ds.records)
        logger.info(
            f"  Mixed-domain replay active: merged {len(replay_ds.records)} replay records "
            f"into primary training stream; total train records={len(train_ds.records)}"
        )
    # Val/test are never augmented
    val_ds   = NabatiASRDataset(cfg.val_jsonl,   processor, corrected_only=False)
    test_ds  = NabatiASRDataset(cfg.test_jsonl,  processor, corrected_only=False)
    collator = DataCollatorSpeechSeq2Seq(processor=processor)

    # ── Balanced sampler ──────────────────────────────────────────────────────
    balanced_sampler = None
    if replay_jsonl and args.replay_ratio > 0.0:
        balanced_sampler = train_ds.make_replay_sampler(
            primary_ratio=1.0 - args.replay_ratio,
            primary_source="nabati",
            replay_source="sada_replay",
            primary_correct_ratio=(args.mix_ratio if args.mix_ratio > 0.0 else None),
            epoch_num_samples=len([r for r in train_ds.records if r.get("source_name") == "nabati"]),
        )
    elif not args.corrected_only and args.mix_ratio > 0.0:
        balanced_sampler = train_ds.make_balanced_sampler(mix_ratio=args.mix_ratio)

    # ── Training Arguments ────────────────────────────────────────────────────
    effective_train_examples = (
        int(getattr(balanced_sampler, "num_samples", 0))
        if balanced_sampler is not None else len(train_ds)
    )
    total_steps  = (effective_train_examples // (cfg.batch_size * cfg.grad_accum_steps)) * cfg.epochs
    total_steps = max(total_steps, cfg.epochs)
    warmup_steps = max(1, int(cfg.warmup_ratio * total_steps))
    logger.info(
        f"  effective_train_examples_per_epoch={effective_train_examples}  "
        f"planned_total_steps={total_steps}  warmup_steps={warmup_steps}"
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir                  = str(output_dir),
        num_train_epochs            = cfg.epochs,
        per_device_train_batch_size = cfg.batch_size,
        per_device_eval_batch_size  = cfg.batch_size * 2,
        gradient_accumulation_steps = cfg.grad_accum_steps,
        learning_rate               = cfg.learning_rate,
        weight_decay                = cfg.weight_decay,
        warmup_steps                = warmup_steps,
        max_grad_norm               = args.max_grad_norm,
        optim                       = args.optimizer,
        lr_scheduler_type           = args.lr_scheduler_type,
        label_smoothing_factor      = args.label_smoothing,
        predict_with_generate       = True,
        generation_max_length       = 225,
        eval_strategy               = "steps",
        eval_steps                  = args.eval_steps,
        save_strategy               = "steps",
        save_steps                  = args.save_steps,
        load_best_model_at_end      = True,
        metric_for_best_model       = args.best_model_metric,
        greater_is_better           = False,
        save_total_limit            = 2,
        fp16                        = (device == "cuda"),
        dataloader_num_workers      = 0,
        report_to                   = "tensorboard",
        logging_dir                 = str(PROJECT_ROOT / "outputs/runs"),
        logging_steps               = max(1, total_steps // (cfg.epochs * 5)),
        seed                        = cfg.seed,
        remove_unused_columns       = False,
    )

    ocr_callback = OCRCallback(
        processor     = processor,
        val_jsonl     = cfg.val_jsonl,
        device        = device,
        ocr_threshold = args.ocr_threshold,
        ocr_wer_threshold = args.ocr_wer_threshold,
        ocr_cer_threshold = args.ocr_cer_threshold,
        stop_on_threshold = args.stop_on_ocr,
    )
    early_stop_callback = EarlyStoppingCallback(
        early_stopping_patience=cfg.patience,
        early_stopping_threshold=0.0,
    )
    wer_gate_callback = BaselineWERGateCallback(
        baseline_val_wer=baseline_val["wer"] if baseline_available else 1.0,
        max_delta=args.wer_delta_gate,
    )

    trainer = BalancedSeq2SeqTrainer(
        model            = model,
        args             = training_args,
        train_dataset    = train_ds,
        eval_dataset     = val_ds,
        data_collator    = collator,
        compute_metrics  = make_compute_metrics(processor.tokenizer, cfg),
        callbacks        = [ocr_callback, wer_gate_callback, early_stop_callback],
        processing_class = processor.feature_extractor,
        balanced_sampler = balanced_sampler,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    logger.info("\n── Training starts ──")
    trainer.train()

    history_rows = [
        row for row in trainer.state.log_history
        if any(key in row for key in ("eval_wer", "eval_cer", "eval_soft_cer", "ocr"))
    ]
    history_path = output_dir / "training_history.json"
    history_path.write_text(
        json.dumps(
            {
                "best_model_metric": args.best_model_metric,
                "optimizer": args.optimizer,
                "lr_scheduler_type": args.lr_scheduler_type,
                "weight_decay": cfg.weight_decay,
                "label_smoothing": cfg.label_smoothing,
                "train_jsonl": str(cfg.train_jsonl),
                "val_jsonl": str(cfg.val_jsonl),
                "test_jsonl": str(cfg.test_jsonl),
                "replay_jsonl": str(replay_jsonl) if replay_jsonl else None,
                "replay_ratio": args.replay_ratio,
                "speed_perturb": list(args.speed_perturb or []),
                "spec_augment": args.spec_augment,
                "freeze_encoder": args.freeze_encoder,
                "decoder_only_lora": args.decoder_only_lora,
                "history": history_rows,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    logger.info(f"Training history saved → {history_path}")

    # ── Test Evaluation ───────────────────────────────────────────────────────
    logger.info("\n── Test Evaluation (fine-tuned model, held-out test set) ──")
    test_results = trainer.evaluate(test_ds, metric_key_prefix="test")
    ft_wer      = test_results.get("test_wer",      float("nan"))
    ft_cer      = test_results.get("test_cer",      float("nan"))
    ft_soft_cer = test_results.get("test_soft_cer", float("nan"))

    logger.info(f"  Fine-tuned: WER={ft_wer:.3f}  CER={ft_cer:.3f}  Soft-CER={ft_soft_cer:.3f}")
    if baseline_available:
        dial_gap     = baseline_test["cer"] - ft_soft_cer
        dial_gap_pct = dial_gap / max(baseline_test["cer"], 1e-9) * 100
        wer_delta = baseline_test["wer"] - ft_wer
        cer_delta = baseline_test["cer"] - ft_cer
        wer_rel   = wer_delta / max(baseline_test["wer"], 1e-9) * 100
        cer_rel   = cer_delta / max(baseline_test["cer"], 1e-9) * 100
    else:
        dial_gap = float("nan")
        dial_gap_pct = float("nan")
        wer_delta = float("nan")
        cer_delta = float("nan")
        wer_rel = float("nan")
        cer_rel = float("nan")

    # ── Report ────────────────────────────────────────────────────────────────
    lora_meta: dict = {}
    if args.use_lora:
        trainable, total_p = model.get_nb_trainable_parameters()
        lora_meta = {"rank": args.lora_rank, "alpha": args.lora_alpha,
                     "dropout": args.lora_dropout, "targets": args.lora_targets,
                     "trainable_M": round(trainable / 1e6, 3),
                     "total_M": round(total_p / 1e6, 1)}

    comparison = {
        "model":           cfg.model_name,
        "lora":            lora_meta if args.use_lora else None,
        "corrected_only":  args.corrected_only,
        "mix_ratio":       args.mix_ratio,
        "optimizer":       args.optimizer,
        "lr_scheduler_type": args.lr_scheduler_type,
        "weight_decay":    cfg.weight_decay,
        "label_smoothing": cfg.label_smoothing,
        "train_jsonl":     str(cfg.train_jsonl),
        "val_jsonl":       str(cfg.val_jsonl),
        "test_jsonl":      str(cfg.test_jsonl),
        "replay_jsonl":    str(replay_jsonl) if replay_jsonl else None,
        "replay_ratio":    args.replay_ratio,
        "speed_perturb":   list(args.speed_perturb or []),
        "spec_augment":    args.spec_augment,
        "freeze_encoder":  args.freeze_encoder,
        "decoder_only_lora": args.decoder_only_lora,
        "best_model_metric": args.best_model_metric,
        "training_note": (
            "ABLATION — corrected-only causes over-correction regression. "
            "Main training uses full train split." if args.corrected_only
            else (
                "Mixed-domain replay training: Nabati remains primary while filtered SADA is sampled "
                "at a fixed replay ratio to reduce catastrophic forgetting."
                if replay_jsonl and args.replay_ratio > 0.0 else
                "Full train split (corrected + already-correct). "
                "Balanced sampler prevents over-correction."
            )
        ),
        "baseline": (
            {
                "source": "text_whisper stored in evaluation JSONL",
                "val_wer": round(baseline_val["wer"], 4),
                "val_cer": round(baseline_val["cer"], 4),
                "val_soft_cer": round(baseline_val["soft_cer"], 4),
                "test_wer": round(baseline_test["wer"], 4),
                "test_cer": round(baseline_test["cer"], 4),
                "test_soft_cer": round(baseline_test["soft_cer"], 4),
            }
            if baseline_available else None
        ),
        "fine_tuned": {
            "test_wer":      round(ft_wer,      4),
            "test_cer":      round(ft_cer,      4),
            "test_soft_cer": round(ft_soft_cer, 4),
        },
        "improvement": (
            {
                "wer_absolute": round(wer_delta, 4),
                "wer_relative_pct": round(wer_rel, 1),
                "cer_absolute": round(cer_delta, 4),
                "cer_relative_pct": round(cer_rel, 1),
            }
            if baseline_available else None
        ),
        "dialect_proximity_score": (
            {
                "value": round(dial_gap, 4),
                "pct_of_baseline_cer": round(dial_gap_pct, 1),
                "interpretation": (
                    f"{dial_gap_pct:.1f}% of apparent error is dialectal variation "
                    f"(CER-strict vs Soft-CER gap)"
                ),
            }
            if baseline_available else None
        ),
    }

    report_path = output_dir / "whisper_comparison.json"
    report_path.write_text(json.dumps(comparison, indent=2, ensure_ascii=False))

    print("\n" + "=" * 60)
    print("  WHISPER  BASELINE vs FINE-TUNED  (test set)")
    print("=" * 60)
    if baseline_available:
        print(f"  Baseline    WER={baseline_test['wer']:.3f}  CER={baseline_test['cer']:.3f}"
              f"  Soft-CER={baseline_test['soft_cer']:.3f}")
    else:
        print("  Baseline    unavailable for this dataset (no stored zero-shot transcript field)")
    print(f"  Fine-tuned  WER={ft_wer:.3f}  CER={ft_cer:.3f}"
          f"  Soft-CER={ft_soft_cer:.3f}"
          + ("  [LoRA]" if args.use_lora else "  [full FT]"))
    if baseline_available:
        print(f"  Δ WER {wer_delta:+.3f} ({wer_rel:+.1f}%)   Δ CER {cer_delta:+.3f} ({cer_rel:+.1f}%)")
        print(f"  Dialect Proximity Gap: {dial_gap:.4f} ({dial_gap_pct:.1f}% of CER)")
    print("=" * 60)
    print(f"  Report → {report_path}\n")
    logger.info(f"Report saved → {report_path}")

    # ── Save best adapter ─────────────────────────────────────────────────────
    best_dir = output_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    if args.use_lora:
        model.save_pretrained(str(best_dir))
        logger.info(f"LoRA adapter saved → {best_dir}")
        logger.info(f"  Reload: PeftModel.from_pretrained(base_model, '{best_dir}')")
    else:
        trainer.save_model(str(best_dir))
    processor.save_pretrained(str(best_dir))

    # ── Qualitative examples ──────────────────────────────────────────────────
    _show_examples(model, test_ds, processor, device, n=5)


# ══════════════════════════════════════════════════════════════════════════════
# 10. QUALITATIVE EXAMPLES
# ══════════════════════════════════════════════════════════════════════════════

def _show_examples(model, test_ds, processor, device, n=5) -> None:
    """Print REF vs HYP for n test clips (both already-correct and needs-correction)."""
    logger.info("\n── Qualitative Examples ──")
    print("\n── Qualitative Examples (test set) ──")
    model.eval()
    tokenizer = processor.tokenizer

    # Show mix: some already-correct, some needs-correction
    correct_idxs   = [i for i, r in enumerate(test_ds.records) if r["is_correct"]][:2]
    incorrect_idxs = [i for i, r in enumerate(test_ds.records) if not r["is_correct"]][:3]

    for tag, idx in [("already-correct", i) for i in correct_idxs] + \
                    [("needs-correction", i) for i in incorrect_idxs]:
        sample = test_ds[idx]
        ref    = test_ds.records[idx]["text_corrected"]
        feats  = torch.tensor(sample["input_features"]).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_ids = model.generate(input_features=feats, language="ar", task="transcribe")
        hyp = tokenizer.decode(pred_ids[0], skip_special_tokens=True).strip()
        print(f"\n  [{tag}]")
        print(f"   REF : {ref}")
        print(f"   HYP : {hyp}")
        logger.info(f"  [{tag}] REF: {ref}  |  HYP: {hyp}")


if __name__ == "__main__":
    main()

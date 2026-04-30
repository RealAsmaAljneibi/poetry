from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict


class PoetrySample(BaseModel):
    """
    Pydantic v2 schema for a single multimodal poetry clip.

    Field naming follows master_dataset.jsonl column names exactly so that
    json.loads() → model_validate() works without any key remapping.
    """

    model_config = ConfigDict(strict=True)

    # ── Audio ──────────────────────────────────────────────────────────────
    audio_filename: Path = Field(..., description="Absolute path to the .mp3 clip")
    source_poem: str = Field(..., description="Poem ID, e.g. poem0025_sa")
    poem_title: str | None = Field(
        default=None, description="Arabic poem title, e.g. يا مدوّر الهين"
    )
    start: int = Field(..., description="Clip start offset in ms")
    end: int = Field(..., description="Clip end offset in ms")

    # ── Text ───────────────────────────────────────────────────────────────
    text_whisper: str = Field(..., description="Original Whisper ASR output")
    text_corrected: str = Field(
        ..., description="Human-corrected ground-truth transcript (LS Pass-A)"
    )

    # ── Poet (critical — used to enforce poet-disjoint splits) ────────────
    poet_en: str = Field(..., description="Poet name in English")
    poet_ar: str = Field(default="", description="Poet name in Arabic")

    # ── Classification targets ─────────────────────────────────────────────
    genre_en: str = Field(..., description="Canonical genre label (AI silver)")
    genre_ar: str = Field(default="", description="Genre in Arabic")

    emotion_text: str = Field(
        ..., description="Emotion inferred from TEXT (AI silver label)"
    )
    emotion_text_ar: str = Field(default="", description="Emotion from text in Arabic")

    emotion_audio: str | None = Field(
        default=None,
        description="Emotion annotated from AUDIO only (human, LS Pass-A). None for 104 unannotated clips.",
    )

    # ── Supporting labels ──────────────────────────────────────────────────
    khaleeji_value: str | None = Field(
        default=None, description="Khaleeji cultural value (AI silver)"
    )
    khaleeji_value_ar: str | None = Field(default=None)
    audio_quality: str | None = Field(
        default=None,
        description="Audio quality flag from LS: clean | leading_noise | background_noise | unclear",
    )

    # ── Metadata ───────────────────────────────────────────────────────────
    translation_en: str | None = Field(default=None)
    imagery_tags_en: str | None = Field(default=None)
    poem_date: str | None = Field(default=None)


class SimilarPoem(BaseModel):
    """A single retrieval result returned by NabatiRetriever."""

    model_config = ConfigDict(strict=True)

    score: float = Field(
        ..., description="Cosine similarity score (+ imagery tag boost)"
    )
    poet_en: str = Field(default="")
    source_poem: str = Field(default="")
    genre_en: str = Field(default="")
    emotion_text: str = Field(default="")
    text_corrected: str = Field(default="")
    imagery_tags_en: str | None = Field(default=None)
    audio_filename: str = Field(default="")
    n_clips: int | None = Field(
        default=None, description="Total clips from this poem in the corpus"
    )


class RankedPrediction(BaseModel):
    """A label-confidence pair used for top-k outputs."""

    model_config = ConfigDict(strict=True)

    label: str = Field(...)
    prob: float = Field(..., ge=0.0, le=1.0)


class InferenceResult(BaseModel):
    """
    Validated output of scripts/demo.py for one audio clip.

    All downstream pipeline stages produce a field here so the result
    is a single structured artifact — inspectable, serialisable, and
    testable without re-running inference.
    """

    model_config = ConfigDict(strict=True)

    # ── Input ──────────────────────────────────────────────────────────────
    audio_file: str = Field(..., description="Path to the input .mp3 file")
    poem_id: str | None = Field(
        default=None, description="Canonical poem identifier if known"
    )

    # ── ASR ────────────────────────────────────────────────────────────────
    transcription: str = Field(
        ..., description="Whisper ASR output (fine-tuned if adapter available)"
    )
    asr_model: str = Field(
        default="openai/whisper-small", description="Model used for ASR"
    )

    # ── Genre ──────────────────────────────────────────────────────────────
    genre: str = Field(..., description="Predicted genre label")
    genre_confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Softmax probability of predicted genre"
    )
    genre_topk: list[RankedPrediction] = Field(
        default_factory=list, description="Top-k genre predictions"
    )
    manual_genre: str | None = Field(
        default=None,
        description="Manual corpus genre used for poem-level conditioning when available",
    )

    # ── Emotion (text) ─────────────────────────────────────────────────────
    emotion_text: str | None = Field(
        default=None,
        description="Emotion predicted from transcript (None if classifier not trained)",
    )
    emotion_text_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    emotion_clip_topk: list[RankedPrediction] = Field(
        default_factory=list, description="Clip-level text emotion top-k"
    )
    emotion_poem_raw_topk: list[RankedPrediction] = Field(
        default_factory=list,
        description="Poem-level raw emotion top-k after aggregation",
    )
    emotion_poem_raw_top1: str | None = Field(default=None)
    emotion_poem_raw_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    emotion_poem_secondary: list[str] = Field(default_factory=list)
    emotion_poem_final: str | None = Field(default=None)
    emotion_poem_final_reason: str | None = Field(default=None)
    poem_emotion_uncertainty: dict[str, float] = Field(
        default_factory=dict, description="Entropy, margin, and average clip confidence"
    )
    poem_emotion_clip_support: list[dict] = Field(
        default_factory=list,
        description="Fraction of poem clips supporting each top emotion",
    )

    # ── Emotion (audio) ────────────────────────────────────────────────────
    emotion_audio: str = Field(
        ..., description="Emotion predicted from audio mel-spectrogram"
    )
    emotion_audio_confidence: float = Field(..., ge=0.0, le=1.0)
    audio_emotion_poem_aux: str | None = Field(
        default=None,
        description="Poem-level auxiliary audio emotion after mapping to the core label space",
    )
    audio_emotion_poem_aux_confidence: float | None = Field(
        default=None, ge=0.0, le=1.0
    )
    audio_emotion_poem_aux_topk: list[RankedPrediction] = Field(default_factory=list)
    audio_emotion_used_in_decision: bool | None = Field(default=None)

    # ── Arousal (audio, from-scratch MLP on librosa features) ──────────────
    arousal: str | None = Field(
        default=None, description="Delivery arousal: Low / Medium / High"
    )
    arousal_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    arousal_poem: str | None = Field(
        default=None, description="Poem-level delivery arousal after clip aggregation"
    )
    arousal_poem_confidence: float | None = Field(default=None, ge=0.0, le=1.0)

    # ── Delivery Mismatch ──────────────────────────────────────────────────
    delivery_mismatch: bool | None = Field(
        default=None,
        description="True when text-implied arousal ≠ audio arousal (cultural ironic delivery)",
    )
    dms_poem: bool | None = Field(
        default=None, description="Poem-level delivery mismatch score"
    )
    delivery_nuance_tag: str | None = Field(
        default=None, description="Interpretive tag describing delivery style"
    )

    # ── Retrieval ──────────────────────────────────────────────────────────
    similar_poems: list[SimilarPoem] = Field(
        default_factory=list, description="Top-k most similar poems from corpus"
    )

    # ── Performance ────────────────────────────────────────────────────────
    inference_ms: float = Field(
        ..., ge=0.0, description="Total wall-clock inference time in milliseconds"
    )

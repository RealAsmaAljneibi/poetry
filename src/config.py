"""
src/config.py

Pydantic v2 training configuration with hard field-level constraints.

Why: If you accidentally type lr=2.0 instead of 2e-5, Pydantic raises
ValidationError *before training starts*, saving hours of wasted compute.

Usage:
    cfg = TextClassifierConfig(
        task="genre",
        train_jsonl="data/processed/train.jsonl",
        val_jsonl="data/processed/val.jsonl",
        test_jsonl="data/processed/test.jsonl",
        output_dir="outputs/models/arapoem_genre",
    )
    # Raises immediately if any field violates its constraint:
    # ValidationError: learning_rate must be <= 0.1

    # Serialise to JSON for reproducibility:
    cfg.model_dump_json(indent=2)
"""

from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict


# ── Base: shared by all training jobs ──────────────────────────────────────

class BaseTrainConfig(BaseModel):
    model_config = ConfigDict(frozen=True)   # immutable after creation → no accidental mutation

    # ── Optimisation ───────────────────────────────────────────────────────
    epochs:        int   = Field(default=10,  ge=1,   le=100,
                                 description="Max training epochs. [1, 100]")
    batch_size:    int   = Field(default=16,  ge=1,   le=256,
                                 description="Mini-batch size. [1, 256]")
    learning_rate: float = Field(default=2e-5, ge=1e-7, le=1e-1,
                                 description="Peak LR. [1e-7, 0.1] — values outside this diverge or freeze.")
    weight_decay:  float = Field(default=0.01, ge=0.0, le=1.0,
                                 description="AdamW weight decay. [0, 1]")
    warmup_ratio:  float = Field(default=0.1,  ge=0.0, le=0.5,
                                 description="Fraction of steps used for LR warmup. [0, 0.5]")
    dropout:       float = Field(default=0.1,  ge=0.0, lt=1.0,
                                 description="Dropout rate. [0, 1) — 1.0 would zero all outputs.")

    # ── Early stopping ──────────────────────────────────────────────────────
    patience:      int   = Field(default=5,   ge=1,   le=50,
                                 description="Epochs without val improvement before stopping. [1, 50]")

    # ── Paths ───────────────────────────────────────────────────────────────
    train_jsonl:   Path  = Field(..., description="Path to train split JSONL")
    val_jsonl:     Path  = Field(..., description="Path to val split JSONL")
    test_jsonl:    Path  = Field(..., description="Path to test split JSONL")
    output_dir:    Path  = Field(..., description="Where to save checkpoints and reports")

    # ── TensorBoard ─────────────────────────────────────────────────────────
    tensorboard_dir: Path = Field(
        default=Path("outputs/runs"),
        description="SummaryWriter log root. Each run gets a subfolder: <model>_<task>_<timestamp>"
    )
    log_every_n_steps: int = Field(default=10, ge=1, le=1000,
                                    description="How often to write scalar metrics to TensorBoard.")

    # ── Gradient accumulation (Week 4 SSL lab) ──────────────────────────────
    grad_accum_steps: int = Field(
        default=1, ge=1, le=32,
        description=(
            "Accumulate gradients over N micro-batches before stepping the optimizer. "
            "Effective batch size = batch_size × grad_accum_steps. "
            "Use 4 when batch_size=16 to simulate batch_size=64 on limited GPU memory. [1, 32]"
        )
    )

    # ── Label Smoothing (prevents overconfidence, improves calibration) ────────
    label_smoothing: float = Field(
        default=0.1, ge=0.0, lt=0.5,
        description=(
            "Cross-entropy label smoothing ε. Replaces hard targets with soft "
            "targets (ε/K for wrong classes, 1-ε+ε/K for the true class). "
            "ε=0.1 is standard; improves ECE by ~30-50%. 0.0 = disabled. [0, 0.5)"
        )
    )

    # ── LR Scheduler (Week 4 lab: OneCycleLR preferred for limited data) ────────
    scheduler: Literal["cosine", "one_cycle"] = Field(
        default="cosine",
        description=(
            "'cosine' = linear warmup + cosine decay (stable default). "
            "'one_cycle' = OneCycleLR: rises to max_lr then decays; "
            "preferred for small datasets — often converges faster."
        )
    )

    # ── Reproducibility ─────────────────────────────────────────────────────
    seed: int = Field(default=42, ge=0, le=2**31 - 1)


# ── Text Classifier (mBERT / AraPoemBERT) ──────────────────────────────────

class TextClassifierConfig(BaseTrainConfig):
    """
    Config for fine-tuning any HuggingFace encoder on genre or emotion classification.

    Supports three techniques from course lectures:
      - Discriminative LR: lower LR for earlier transformer layers (Week 4)
      - Gradual unfreezing: unfreeze 1-2 layers per epoch, not all at once (Week 4)
      - Focal loss: down-weights easy majority class (Ghazal) to improve Macro-F1 (Week 3)
    """

    # ── Task ────────────────────────────────────────────────────────────────
    task: Literal["genre", "emotion_text", "emotion_audio"] = Field(
        ..., description="Which label column to predict."
    )
    model_name: str = Field(
        default="CAMeL-Lab/bert-base-arabic-camelbert-msa",
        description="HuggingFace model ID. Use 'faisalq/bert-base-arapoembert' for domain-adapted training."
    )

    # ── Sequence ────────────────────────────────────────────────────────────
    max_seq_len: int = Field(
        default=128, ge=16, le=512,
        description="Max token length. BERT hard limit is 512. [16, 512]"
    )

    # ── Discriminative LR (Week 4) ──────────────────────────────────────────
    discriminative_lr_decay: float = Field(
        default=0.9, ge=0.1, le=1.0,
        description=(
            "Per-layer LR multiplier (bottom-up). "
            "Layer 0 LR = learning_rate * decay^11; top layer = learning_rate. "
            "1.0 = uniform LR (disabled). [0.1, 1.0]"
        )
    )

    # ── Gradual Unfreezing (Week 4) ─────────────────────────────────────────
    gradual_unfreeze_epochs: int = Field(
        default=2, ge=0, le=12,
        description=(
            "Unfreeze 1 transformer layer group every N epochs, starting from the top. "
            "BERT-base has 12 layers, so max meaningful value is 12. "
            "0 = train all layers from epoch 1. [0, 12]"
        )
    )

    # ── Focal Loss (Week 3) ──────────────────────────────────────────────────
    use_focal_loss: bool = Field(
        default=False,
        description="Replace cross-entropy with focal loss to handle class imbalance."
    )
    focal_gamma: float = Field(
        default=2.0, ge=0.0, le=5.0,
        description="Focal loss focusing parameter γ. 0 = standard CE. Typical range [1, 3]. [0, 5]"
    )

    # ── Cross-field validation ───────────────────────────────────────────────
    @model_validator(mode="after")
    def focal_gamma_requires_focal_loss(self) -> "TextClassifierConfig":
        if self.focal_gamma != 2.0 and not self.use_focal_loss:
            raise ValueError(
                "focal_gamma is set but use_focal_loss=False. "
                "Either set use_focal_loss=True or leave focal_gamma at default."
            )
        return self


# ── Audio CNN ───────────────────────────────────────────────────────────────

class AudioCNNConfig(BaseTrainConfig):
    """
    Config for training Emotion1DCNN from scratch on mel-spectrograms.
    """

    task: Literal["emotion_audio"] = Field(
        default="emotion_audio",
        description="Audio CNN always predicts emotion_audio (human-annotated)."
    )

    # ── Audio preprocessing ──────────────────────────────────────────────────
    max_audio_sec: int   = Field(default=8,   ge=1,  le=30,
                                  description="Clips longer than this are truncated. [1, 30]")
    sample_rate:   int   = Field(default=16000, ge=8000, le=48000,
                                  description="Resampling rate in Hz. [8000, 48000]")
    n_mels:        int   = Field(default=128,  ge=32, le=256,
                                  description="Number of mel filterbank channels. [32, 256]")

    # ── Augmentation ─────────────────────────────────────────────────────────
    noise_amplitude: float = Field(
        default=0.005, ge=0.0, le=0.1,
        description="Gaussian noise injection amplitude (fraction of signal max). [0, 0.1]"
    )

    @field_validator("sample_rate")
    @classmethod
    def sample_rate_must_be_standard(cls, v: int) -> int:
        standard = {8000, 16000, 22050, 44100, 48000}
        if v not in standard:
            raise ValueError(
                f"sample_rate={v} is non-standard. Use one of {sorted(standard)} "
                "to avoid librosa resampling artefacts."
            )
        return v


# ── ASR Fine-tuning ─────────────────────────────────────────────────────────

class ASRConfig(BaseTrainConfig):
    """
    Config for fine-tuning Whisper on Nabati Arabic.
    """

    model_name: str = Field(
        default="openai/whisper-small",
        description=(
            "Whisper model size. "
            "whisper-small=241.7M — approved exception (2026-03-02); "
            "required because baseline transcripts were generated by whisper-small and "
            "whisper-base (72.6M) cannot match its quality on Nabati dialect. "
            "Do NOT use whisper-medium (248M) or larger — no exception covers those."
        )
    )
    language:    str  = Field(default="ar",   description="Target language code for Whisper.")
    max_audio_sec: int = Field(default=30,  ge=1, le=30,
                                description="Whisper hard-coded limit is 30s. [1, 30]")

    # CER and WER thresholds — if eval exceeds these, training is likely diverging
    cer_warn_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Log a warning if val CER exceeds this. [0, 1]"
    )
    wer_warn_threshold: float = Field(
        default=0.8, ge=0.0, le=1.0,
        description="Log a warning if val WER exceeds this. [0, 1]"
    )


# ── Convenience defaults ─────────────────────────────────────────────────────

DATA_PATHS = dict(
    train_jsonl = Path("data/processed/train.jsonl"),
    val_jsonl   = Path("data/processed/val.jsonl"),
    test_jsonl  = Path("data/processed/test.jsonl"),
)

def mbert_genre_config() -> TextClassifierConfig:
    return TextClassifierConfig(
        model_name="bert-base-multilingual-cased",
        task="genre",
        output_dir=Path("outputs/models/mbert_genre"),
        **DATA_PATHS,
    )

def arapoem_genre_config() -> TextClassifierConfig:
    return TextClassifierConfig(
        model_name="faisalq/bert-base-arapoembert",
        task="genre",
        max_seq_len=32,          # AraPoemBERT hard limit: max_position_embeddings=32
        epochs=20,
        patience=8,
        learning_rate=3e-5,
        discriminative_lr_decay=0.9,
        gradual_unfreeze_epochs=1,   # 1 layer/epoch → all 10 BERT layers active by epoch 10
        grad_accum_steps=4,          # effective batch = 16×4 = 64 (SSL lab technique)
        use_focal_loss=False,        # Run 3: class weights alone — focal+weights double-penalise
        output_dir=Path("outputs/models/arapoem_genre"),
        **DATA_PATHS,
    )

def arapoem_emotion_config() -> TextClassifierConfig:
    return TextClassifierConfig(
        model_name="faisalq/bert-base-arapoembert",
        task="emotion_text",
        max_seq_len=32,          # AraPoemBERT hard limit: max_position_embeddings=32
        epochs=20,
        patience=8,
        learning_rate=3e-5,
        discriminative_lr_decay=0.9,
        gradual_unfreeze_epochs=1,
        grad_accum_steps=4,
        use_focal_loss=False,        # class weights alone
        output_dir=Path("outputs/models/arapoem_emotion"),
        **DATA_PATHS,
    )

def audio_cnn_config() -> AudioCNNConfig:
    return AudioCNNConfig(
        output_dir=Path("outputs/models/audio_cnn"),
        # CNN is trained from scratch — needs much higher LR than BERT fine-tuning
        learning_rate=1e-3,      # standard for CNN from scratch (BERT uses 2e-5)
        batch_size=32,
        epochs=30,
        patience=10,
        weight_decay=1e-4,
        warmup_ratio=0.05,
        **DATA_PATHS,
    )


# ── Arousal MLP ──────────────────────────────────────────────────────────────

class ArousalConfig(BaseTrainConfig):
    """
    Config for the Arousal MLP trained from scratch on librosa audio features.

    Why Arousal instead of 12-class audio emotion:
        Audio reliably encodes *delivery energy* (High/Medium/Low) but not
        fine-grained semantic emotion (which lives in the words).
        The 3-class problem is nearly balanced (33/31/36%) and the features
        are directly audible — making this a tractable from-scratch model.

    Feature vector (34 dimensions):
        13 MFCC means + 13 MFCC stds  (26)
        RMS energy mean + std          (2)
        Zero crossing rate mean        (1)
        Spectral centroid mean + std   (2)
        Spectral rolloff mean          (1)
        Tempo (BPM)                    (1)
        Pause ratio                    (1)
    """

    task: Literal["arousal"] = Field(
        default="arousal",
        description="Arousal MLP always predicts 3-class arousal (Low/Medium/High)."
    )

    # ── Audio preprocessing ──────────────────────────────────────────────────
    sample_rate:    int   = Field(default=16000, ge=8000, le=48000)
    max_audio_sec:  int   = Field(default=30,    ge=1,    le=30)
    n_mels:         int   = Field(default=128,   ge=32,   le=256)
    n_mfcc:         int   = Field(default=13,    ge=8,    le=40,
                                   description="Number of MFCC coefficients to extract. [8, 40]")

    # ── MLP architecture ─────────────────────────────────────────────────────
    hidden_dim:     int   = Field(default=128,   ge=32,   le=512,
                                   description="Hidden layer width. [32, 512]")
    n_layers:       int   = Field(default=2,     ge=1,    le=6,
                                   description="Number of hidden layers. [1, 6]")
    dropout:        float = Field(default=0.3,   ge=0.0,  lt=1.0)

    @field_validator("sample_rate")
    @classmethod
    def sample_rate_must_be_standard(cls, v: int) -> int:
        standard = {8000, 16000, 22050, 44100, 48000}
        if v not in standard:
            raise ValueError(f"sample_rate={v} is non-standard. Use one of {sorted(standard)}")
        return v


def arousal_config() -> ArousalConfig:
    """Default config for the Arousal MLP (from-scratch model)."""
    return ArousalConfig(
        output_dir=Path("outputs/models/arousal_mlp"),
        learning_rate=1e-3,
        batch_size=64,
        epochs=50,
        patience=10,
        weight_decay=1e-4,
        warmup_ratio=0.05,
        scheduler="cosine",
        label_smoothing=0.05,
        **DATA_PATHS,
    )

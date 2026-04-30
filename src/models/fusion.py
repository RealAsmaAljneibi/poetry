"""
src/models/fusion.py

NabatiMultimodalFusion — the component that makes Nabat-AI multimodal.

Architecture:
    Text: text_corrected ──► AraPoemBERT ──► text_embedding (768-dim)
                                                      ↘
                                               [Fusion Layer] ──► genre / emotion
                                                      ↗
    Audio:  .mp3  ──► mel-spec ──► Emotion1DCNN ──► audio_embedding (512-dim)

Three fusion strategies (used for ablation study):
    1. "concat"     — concatenate [text, audio] → MLP  (simplest, strong baseline)
    2. "gated"      — learnable gate weights how much to trust each modality
    3. "cross_attn" — audio tokens attend to text tokens and vice versa (strongest)

Ablation study modes:
    mode="text_only"   → only uses text_embedding, ignores audio
    mode="audio_only"  → only uses audio_embedding, ignores text
    mode="fusion"      → uses both (any of the three strategies above)

This lets you build the ablation table:
    | Model           | Genre F1 | Emotion F1 |
    |-----------------|----------|------------|
    | Text only       |  X       |  Y         |
    | Audio only      |  –       |  Z         |
    | Fusion (concat) |  X+      |  Y+        |
    | Fusion (gated)  |  X++     |  Y++       |
    | Fusion (attn)   |  X+++    |  Y+++      |
"""

from typing import Literal

from loguru import logger
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Fusion Strategies ───────────────────────────────────────────────────────


class ConcatFusion(nn.Module):
    """Concatenates text and audio embeddings → MLP → logits."""

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(text_dim + audio_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, text_emb: torch.Tensor, audio_emb: torch.Tensor) -> torch.Tensor:
        fused = torch.cat([text_emb, audio_emb], dim=-1)  # (B, text_dim + audio_dim)
        return self.mlp(fused)


class GatedFusion(nn.Module):
    """
    Gated fusion: the model learns how much to weight each modality.
    Key insight: when emotion_text == emotion_audio (15.5% of clips),
    both gates open equally. When they diverge (84.5% of clips, Nabati ironic
    delivery), the gate learns to trust the more informative modality for each class.
    """

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),  # 2 scalars: weight for text, weight for audio
            nn.Softmax(dim=-1),  # sum to 1 → interpretable
        )

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def _project_and_gate(
        self, text_emb: torch.Tensor, audio_emb: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        t = F.relu(self.text_proj(text_emb))
        a = F.relu(self.audio_proj(audio_emb))
        return t, a, self.gate(torch.cat([t, a], dim=-1))

    def forward(self, text_emb: torch.Tensor, audio_emb: torch.Tensor) -> torch.Tensor:
        t, a, gate_weights = self._project_and_gate(text_emb, audio_emb)
        w_text = gate_weights[:, 0:1]  # (B, 1)
        w_audio = gate_weights[:, 1:2]  # (B, 1)
        fused = w_text * t + w_audio * a  # (B, hidden) — weighted sum
        return self.classifier(fused)

    def get_gate_weights(
        self, text_emb: torch.Tensor, audio_emb: torch.Tensor
    ) -> torch.Tensor:
        """Returns (B, 2) gate weights [text_weight, audio_weight] for analysis."""
        _, _, gate_weights = self._project_and_gate(text_emb, audio_emb)
        return gate_weights


class CrossModalAttentionFusion(nn.Module):
    """
    Cross-modal attention fusion (strongest strategy).

    Audio embedding attends to text embedding (and vice versa):
    → the model learns WHICH parts of the text are most relevant to the audio emotion
    → captures the Nabati ironic delivery pattern: audio gate activates
       when voice contradicts the literal text meaning
    """

    def __init__(
        self,
        text_dim: int,
        audio_dim: int,
        hidden_dim: int,
        num_classes: int,
        dropout: float,
        num_heads: int = 4,
    ):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)

        # Multi-head cross-attention: audio queries attend to text keys/values
        self.audio_to_text_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        # And text queries attend to audio keys/values
        self.text_to_audio_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, text_emb: torch.Tensor, audio_emb: torch.Tensor) -> torch.Tensor:
        # Project to hidden_dim and add sequence dimension (L=1)
        t = self.text_proj(text_emb).unsqueeze(1)  # (B, 1, hidden)
        a = self.audio_proj(audio_emb).unsqueeze(1)  # (B, 1, hidden)

        # Audio attends to text: "given what the voice sounds like, what text is most relevant?"
        a_attended, _ = self.audio_to_text_attn(query=a, key=t, value=t)
        a_out = self.norm1(a + a_attended).squeeze(1)  # (B, hidden)

        # Text attends to audio: "given what the text says, how does the voice confirm/contradict it?"
        t_attended, _ = self.text_to_audio_attn(query=t, key=a, value=a)
        t_out = self.norm2(t + t_attended).squeeze(1)  # (B, hidden)

        fused = torch.cat([t_out, a_out], dim=-1)  # (B, hidden*2)
        return self.classifier(fused)


# ── Main Fusion Model ────────────────────────────────────────────────────────


class NabatiMultimodalFusion(nn.Module):
    """
    The top-level multimodal model for Nabat-AI.

    Wraps either AraPoemBERT (text encoder) + Emotion1DCNN (audio encoder)
    with a chosen fusion strategy.

    In ablation mode: can run text-only or audio-only by zeroing out the other modality.

    Usage:
        model = NabatiMultimodalFusion(
            text_encoder=arapoem,      # AraPoemBERT or mBERT, already fine-tuned
            audio_encoder=cnn,         # Emotion1DCNN, already fine-tuned
            fusion_strategy="gated",
            task="emotion_audio",
            num_classes=12,
        )
        logits = model(input_ids, attention_mask, mel_spec, mode="fusion")
    """

    def __init__(
        self,
        text_encoder: nn.Module,
        audio_encoder: nn.Module,
        fusion_strategy: Literal["concat", "gated", "cross_attn"] = "gated",
        task: Literal["genre", "emotion_text", "emotion_audio"] = "emotion_audio",
        num_classes: int = 12,
        text_dim: int = 768,  # AraPoemBERT hidden size
        audio_dim: int = 512,  # Emotion1DCNN embed() output dim
        hidden_dim: int = 512,
        dropout: float = 0.3,
        freeze_encoders: bool = True,  # Freeze pre-trained encoders during fusion training
    ):
        super().__init__()
        self.task = task

        self.text_encoder = text_encoder
        self.audio_encoder = audio_encoder

        if freeze_encoders:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            for param in self.audio_encoder.parameters():
                param.requires_grad = False
            logger.info(
                "Fusion: encoders frozen. Only the fusion head will be trained."
            )
        else:
            logger.info("Fusion: all parameters trainable (end-to-end fine-tuning).")

        self.fusion_strategy = fusion_strategy
        if fusion_strategy == "concat":
            self.fusion = ConcatFusion(
                text_dim, audio_dim, hidden_dim, num_classes, dropout
            )
        elif fusion_strategy == "gated":
            self.fusion = GatedFusion(
                text_dim, audio_dim, hidden_dim, num_classes, dropout
            )
        elif fusion_strategy == "cross_attn":
            self.fusion = CrossModalAttentionFusion(
                text_dim, audio_dim, hidden_dim, num_classes, dropout
            )
        else:
            raise ValueError(
                f"Unknown fusion_strategy='{fusion_strategy}'. "
                "Choose from: 'concat', 'gated', 'cross_attn'"
            )

        # These are simple linear classifiers on the raw embeddings.
        # Used ONLY in mode='text_only' or mode='audio_only'.
        self.text_only_head = nn.Linear(text_dim, num_classes)
        self.audio_only_head = nn.Linear(audio_dim, num_classes)

        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"NabatiMultimodalFusion({fusion_strategy}, task={task}, n_classes={num_classes}): "
            f"total={total:,} params, trainable={trainable:,}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mel_spec: torch.Tensor,
        mode: Literal["fusion", "text_only", "audio_only"] = "fusion",
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      (B, seq_len)     tokenised text input
            attention_mask: (B, seq_len)     padding mask
            mel_spec:       (B, 128, T)      mel-spectrogram
            mode:           ablation control — 'fusion' | 'text_only' | 'audio_only'
        Returns:
            logits: (B, num_classes)
        """
        if mode in ("fusion", "text_only"):
            text_output = self.text_encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )
            # HuggingFace: use [CLS] token (index 0) as the sentence embedding
            text_emb = text_output.last_hidden_state[:, 0, :]  # (B, 768)

        if mode in ("fusion", "audio_only"):
            audio_emb = self.audio_encoder.embed(mel_spec)  # (B, 512)

        if mode == "text_only":
            return self.text_only_head(text_emb)

        if mode == "audio_only":
            return self.audio_only_head(audio_emb)

        return self.fusion(text_emb, audio_emb)

    def get_gate_weights(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        mel_spec: torch.Tensor,
    ) -> torch.Tensor:
        """
        Only available for fusion_strategy='gated'.
        Returns (B, 2) showing [text_weight, audio_weight] per sample.
        Useful for analysis: on clips where text/audio emotion diverge,
        the model should learn to weight audio higher.
        """
        assert self.fusion_strategy == "gated", (
            "get_gate_weights() is only available for fusion_strategy='gated'"
        )

        text_emb = self.text_encoder(input_ids, attention_mask).last_hidden_state[
            :, 0, :
        ]
        audio_emb = self.audio_encoder.embed(mel_spec)
        return self.fusion.get_gate_weights(text_emb, audio_emb)


if __name__ == "__main__":
    from src.models.audio_cnn import Emotion1DCNN

    logger.add("logs/fusion.log", rotation="10 MB")
    device = torch.device("cpu")

    class StubTextEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(1, 768)

        def forward(self, input_ids, attention_mask):
            B = input_ids.shape[0]
            out = torch.randn(B, 10, 768)

            class Out:
                last_hidden_state = out

            return Out()

    text_enc = StubTextEncoder()
    audio_enc = Emotion1DCNN(num_classes=12)

    for strategy in ("concat", "gated", "cross_attn"):
        model = NabatiMultimodalFusion(
            text_encoder=text_enc,
            audio_encoder=audio_enc,
            fusion_strategy=strategy,
            num_classes=12,
        )
        B = 4
        dummy_ids = torch.zeros(B, 32, dtype=torch.long)
        dummy_mask = torch.ones(B, 32, dtype=torch.long)
        dummy_audio = torch.randn(B, 128, 251)

        for ablation_mode in ("fusion", "text_only", "audio_only"):
            logits = model(dummy_ids, dummy_mask, dummy_audio, mode=ablation_mode)
            logger.info(f"[{strategy}] mode={ablation_mode}: logits {logits.shape}  OK")

    logger.success("All fusion strategies and ablation modes verified.")

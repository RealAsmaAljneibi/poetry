"""
scripts/pretrain_audio_simclr.py

SimCLR self-supervised pre-training for the Emotion1DCNN encoder.

Key idea: Feed the same audio clip through two different random augmentations.
The encoder is trained to produce similar embeddings for the two views of the
same clip (positive pair) and dissimilar embeddings for different clips (negatives).

After pre-training:
  - The encoder weights are saved to outputs/models/simclr_encoder.pt
  - train_audio_cnn.py loads these weights and only fine-tunes the classifier head

Usage:
    uv run python scripts/pretrain_audio_simclr.py
"""

import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.audio_cnn import Emotion1DCNN
from src.training.trainer import set_seed, get_scheduler


# ── Hyper-parameters ──────────────────────────────────────────────────────────

SAMPLE_RATE = 16_000
MAX_AUDIO_SEC = 8
N_MELS = 128
BATCH_SIZE = 64
EPOCHS = 80
LR = 3e-4  # SimCLR uses lower LR than supervised
WEIGHT_DECAY = 1e-4
WARMUP_RATIO = 0.1
TEMPERATURE = 0.5  # NT-Xent temperature (0.5 is standard)
PROJ_DIM = 128  # projection head output dimension
SEED = 42


# ── NT-Xent Loss ────────────────────────────────────────────────────────────────


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross-Entropy Loss.

    For a batch of B pairs (z_i, z_j):
      - Positive pair: two augmented views of the same clip
      - All other 2B-2 clips in the batch are negatives
      - Loss pulls positives together and pushes negatives apart

    Lower temperature → sharper distribution → harder negatives get more weight.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        B = z_i.shape[0]
        z = torch.cat([z_i, z_j], dim=0)  # (2B, dim)

        # Cosine similarity matrix (already normalised, so just dot product)
        sim = torch.mm(z, z.T) / self.temperature  # (2B, 2B)

        # Remove self-similarity (diagonal = 1.0 after L2 norm)
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, -1e9)

        # Positive index: for i in [0,B) the positive is i+B, and vice versa
        labels = torch.cat(
            [
                torch.arange(B, 2 * B, device=z.device),  # view1_i → view2_i
                torch.arange(0, B, device=z.device),  # view2_i → view1_i
            ]
        )

        return F.cross_entropy(sim, labels)


# ── Projection Head ───────────────────────────────────────────────────────────


class ProjectionHead(nn.Module):
    """
    Two-layer MLP that maps encoder embeddings to the contrastive space.

    After pre-training, this head is discarded — only the encoder is kept.
    Using a projection head before contrastive loss (rather than directly on
    encoder output) is a key SimCLR finding: it preserves more information in
    the encoder embeddings for downstream tasks.
    """

    def __init__(
        self, input_dim: int = 512, hidden_dim: int = 256, output_dim: int = 128
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)  # L2-normalise for cosine similarity


# ── Audio Augmentation ────────────────────────────────────────────────────────


def random_augment(wav: np.ndarray, sr: int, max_len: int, n_mels: int) -> torch.Tensor:
    """
    Strong augmentation pipeline to create one view of an audio clip.
    Applied twice to the same raw wav to produce a positive pair.

    Augmentations:
      1. Random crop — different time window each call (temporal diversity)
      2. Volume scaling — randomly louder or quieter
      3. Gaussian noise — different random noise each call
      4. Mel spectrogram
      5. SpecAugment — different frequency/time masks each call
    """
    # 1. Random crop: pick a random start within the clip
    crop_len = min(len(wav), max_len)
    if len(wav) > crop_len:
        start = np.random.randint(0, len(wav) - crop_len)
        wav = wav[start : start + crop_len]
    else:
        wav = np.pad(wav, (0, max_len - len(wav)))

    # 2. Volume scaling (0.7 – 1.3×)
    scale = 0.7 + 0.6 * np.random.random()
    wav = wav * scale

    # 3. Gaussian noise
    noise_amp = 0.005 * np.random.uniform() * max(np.amax(np.abs(wav)), 1e-6)
    wav = wav + noise_amp * np.random.normal(size=wav.shape).astype(np.float32)

    # 4. Mel spectrogram
    mel = librosa.feature.melspectrogram(y=wav, sr=sr, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)
    mel_t = torch.tensor(mel_db)  # (n_mels, T)

    # 5. SpecAugment — frequency and time masking
    mel_t = _spec_augment(
        mel_t, freq_mask_param=30, time_mask_param=50, num_freq=2, num_time=2
    )
    return mel_t


def _spec_augment(
    mel: torch.Tensor,
    freq_mask_param: int = 30,
    time_mask_param: int = 50,
    num_freq: int = 2,
    num_time: int = 2,
) -> torch.Tensor:
    mel = mel.clone()
    n_mels, T = mel.shape
    mean_val = mel.mean()

    for _ in range(num_freq):
        f = np.random.randint(0, freq_mask_param + 1)
        f0 = np.random.randint(0, max(1, n_mels - f))
        mel[f0 : f0 + f, :] = mean_val

    for _ in range(num_time):
        t = np.random.randint(0, min(time_mask_param + 1, T))
        t0 = np.random.randint(0, max(1, T - t))
        mel[:, t0 : t0 + t] = mean_val

    return mel


# ── SSL Dataset ───────────────────────────────────────────────────────────────


class AudioSSLDataset(Dataset):
    """
    Returns TWO differently-augmented views of the same audio clip (positive pair).
    Labels are NOT used — this is the SSL pre-training dataset.

    Loads all audio files found in any of the three JSONL splits.
    """

    def __init__(
        self,
        jsonl_paths: list[Path],
        max_audio_sec: int = MAX_AUDIO_SEC,
        sample_rate: int = SAMPLE_RATE,
        n_mels: int = N_MELS,
    ):
        self.sr = sample_rate
        self.max_len = max_audio_sec * sample_rate
        self.n_mels = n_mels
        self.paths: list[Path] = []

        seen = set()
        for jsonl_path in jsonl_paths:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    p = Path(rec.get("audio_filename", ""))
                    if str(p) not in seen and p.exists():
                        self.paths.append(p)
                        seen.add(str(p))

        logger.info(f"AudioSSLDataset: {len(self.paths)} unique clips (no labels used)")

    def __len__(self) -> int:
        return len(self.paths)

    def _load_wav(self, path: Path) -> np.ndarray:
        try:
            wav, _ = librosa.load(str(path), sr=self.sr, mono=True)
        except Exception as e:
            logger.warning(f"Failed to load {path}: {e} — using silence")
            wav = np.zeros(self.max_len, dtype=np.float32)
        return wav

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        wav = self._load_wav(self.paths[idx])
        view1 = random_augment(wav, self.sr, self.max_len, self.n_mels)
        view2 = random_augment(wav, self.sr, self.max_len, self.n_mels)
        return view1, view2


# ── Collate (pad to batch max time) ──────────────────────────────────────────


def collate_ssl(batch: list[tuple]) -> tuple[torch.Tensor, torch.Tensor]:
    v1s, v2s = zip(*batch)
    max_T = max(v.shape[-1] for v in v1s + v2s)

    def pad(v):
        return F.pad(v, (0, max_T - v.shape[-1]))

    return (
        torch.stack([pad(v) for v in v1s]),
        torch.stack([pad(v) for v in v2s]),
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    logger.add("logs/pretrain_simclr.log", rotation="10 MB")
    set_seed(SEED)

    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    logger.info(
        f"SimCLR pre-training | device={device} | epochs={EPOCHS} | "
        f"batch={BATCH_SIZE} | lr={LR:.1e} | τ={TEMPERATURE}"
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    data_dir = Path("data/processed")
    ssl_ds = AudioSSLDataset(
        [
            data_dir / "train.jsonl",
            data_dir / "val.jsonl",
            data_dir / "test.jsonl",
        ]
    )
    if len(ssl_ds) == 0:
        logger.error("No audio files found. Check data/processed/*.jsonl paths.")
        return

    loader = DataLoader(
        ssl_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_ssl,
        drop_last=True,
    )
    logger.info(
        f"Loader: {len(loader)} batches/epoch "
        f"(effective negatives per sample: {2 * BATCH_SIZE - 2})"
    )

    # ── Model: encoder + projection head ─────────────────────────────────────
    encoder = Emotion1DCNN(num_classes=12).to(device)  # num_classes unused in SSL
    proj = ProjectionHead(input_dim=512, hidden_dim=256, output_dim=PROJ_DIM).to(device)

    enc_params = sum(p.numel() for p in encoder.parameters())
    proj_params = sum(p.numel() for p in proj.parameters())
    logger.info(
        f"Encoder: {enc_params:,} params | ProjectionHead: {proj_params:,} params"
    )

    # ── Loss ──────────────────────────────────────────────────────────────────
    criterion = NTXentLoss(temperature=TEMPERATURE)

    # ── Optimizer + Scheduler ─────────────────────────────────────────────────
    params = list(encoder.parameters()) + list(proj.parameters())
    optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = len(loader) * EPOCHS
    scheduler = get_scheduler(optimizer, total_steps, WARMUP_RATIO)

    # ── Pre-training loop ─────────────────────────────────────────────────────
    output_dir = Path("outputs/models")
    output_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    logger.info("=" * 60)
    logger.info(
        f"Starting SimCLR pre-training: {EPOCHS} epochs, {len(loader)} steps/epoch"
    )
    logger.info("=" * 60)

    for epoch in range(EPOCHS):
        encoder.train()
        proj.train()
        total_loss = 0.0

        for v1, v2 in loader:
            v1, v2 = v1.to(device), v2.to(device)

            # Encoder embed → projection head → L2-normalised
            z1 = proj(encoder.embed(v1))  # (B, PROJ_DIM)
            z2 = proj(encoder.embed(v2))

            loss = criterion(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        current_lr = scheduler.get_last_lr()[0]

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch + 1:3d}/{EPOCHS} | "
                f"NT-Xent loss={avg_loss:.4f} | lr={current_lr:.2e}"
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            # Save ONLY the encoder weights (projection head is discarded)
            encoder_ckpt = output_dir / "simclr_encoder.pt"
            torch.save(encoder.state_dict(), encoder_ckpt)

    logger.success(
        f"Pre-training done. Best NT-Xent loss={best_loss:.4f} "
        f"(< 2.5 indicates good clustering)"
    )
    logger.info(f"Encoder weights saved → {output_dir / 'simclr_encoder.pt'}")
    logger.info(
        "Next step: run train_audio_cnn.py --pretrained-encoder "
        "outputs/models/simclr_encoder.pt"
    )


if __name__ == "__main__":
    main()

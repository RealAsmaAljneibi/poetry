"""
scripts/train_vae_augment.py

Conditional Variational Autoencoder (cVAE) for mel-spectrogram synthesis.

Why this matters for Nabat-AI:
  - 12 emotion classes, highly imbalanced (Longing: 4 clips, Defiance: 81 clips)
  - Zero-precision on Longing, Compassion, Humor in every trained model
  - Conditional VAE: condition decoder on class label → generate mel spectrograms
    specifically for underrepresented emotion classes
  - Mix synthetic + real data at 1:1 ratio during training (real data only for eval)

Architecture (cVAE):
  Encoder: flatten(mel) + one_hot(label) → 512-d → μ, log σ²
  Reparameterise: z = μ + ε·σ  (ε ~ N(0, I))
  Decoder: z + one_hot(label) → 512-d → reconstruct mel
  Loss: MSE reconstruction + β·KL divergence

Usage:
  # Train and generate
  uv run python scripts/train_vae_augment.py
  uv run python scripts/train_vae_augment.py --epochs 200 --beta 1.0
  uv run python scripts/train_vae_augment.py --generate-only   # skip training

After running:
  outputs/augmentation/vae_mel_synthetic.jsonl  ← metadata for synthetic clips
  outputs/augmentation/mel_*.npy                 ← generated mel spectrograms
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

import librosa
from loguru import logger
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.labels import EMOTION_CLASSES, encode_emotion, ID2EMOTION
from src.training.trainer import set_seed

# ── Hyperparameters ───────────────────────────────────────────────────────────

SAMPLE_RATE = 16_000
MAX_AUDIO_SEC = 8
N_MELS = 64  # reduced from 128 for feasible flat VAE
HOP_LENGTH = 256
N_TIME = int(MAX_AUDIO_SEC * SAMPLE_RATE / HOP_LENGTH)  # ≈ 500 frames
LATENT_DIM = 64
HIDDEN_DIM = 512
BATCH_SIZE = 32
EPOCHS = 150
LR = 1e-3
BETA = 1.0  # KL weight (β-VAE; 1.0 = standard VAE)
NUM_CLASSES = len(EMOTION_CLASSES)  # 12
SEED = 42

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("outputs/augmentation")


# ── Dataset ───────────────────────────────────────────────────────────────────


class MelDataset(Dataset):
    """
    Load mel spectrograms from audio clips + emotion labels.
    Pads/trims to fixed size (N_MELS × N_TIME) for the flat VAE.
    """

    def __init__(self, jsonl_path: Path, augment: bool = False):
        self.samples: list[dict] = []
        self.augment = augment

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                label_str = rec.get("emotion_audio", "")
                if not label_str:
                    continue
                label_id = encode_emotion(label_str)
                if label_id < 0:
                    continue
                audio_path = Path(rec.get("audio_filename", ""))
                if not audio_path.exists():
                    continue
                self.samples.append(
                    {"path": audio_path, "label": label_id, "emotion": label_str}
                )

        counts = Counter(s["label"] for s in self.samples)
        logger.info(
            f"MelDataset: {len(self.samples)} clips — "
            f"min class={min(counts.values())}, max class={max(counts.values())}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_mel(self, path: Path) -> np.ndarray:
        try:
            wav, _ = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
        except Exception as e:
            logger.warning(f"  Failed to load {path}: {e}")
            return np.zeros((N_MELS, N_TIME), dtype=np.float32)

        mel = librosa.feature.melspectrogram(
            y=wav, sr=SAMPLE_RATE, n_mels=N_MELS, hop_length=HOP_LENGTH
        )
        mel_db = librosa.power_to_db(mel, ref=np.max).astype(np.float32)

        # Pad or trim to fixed N_TIME columns
        T = mel_db.shape[1]
        if T < N_TIME:
            mel_db = np.pad(mel_db, ((0, 0), (0, N_TIME - T)))
        else:
            mel_db = mel_db[:, :N_TIME]

        # Normalise to [-1, 1]  (dB range is typically -80 to 0)
        mel_db = (mel_db + 40.0) / 40.0  # maps [-80, 0] → [-1, 1]
        return mel_db

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        s = self.samples[idx]
        mel = self._load_mel(s["path"])  # (N_MELS, N_TIME)

        if self.augment:
            # Light SpecAugment: 1 frequency mask + 1 time mask
            mel = mel.copy()
            f = np.random.randint(0, 10)
            f0 = np.random.randint(0, max(1, N_MELS - f))
            mel[f0 : f0 + f, :] = 0.0

            t = np.random.randint(0, 30)
            t0 = np.random.randint(0, max(1, N_TIME - t))
            mel[:, t0 : t0 + t] = 0.0

        return torch.tensor(mel, dtype=torch.float32), s["label"]


# ── Model ─────────────────────────────────────────────────────────────────────

INPUT_DIM = N_MELS * N_TIME  # flat vector size


class ConditionalVAE(nn.Module):
    """
    Conditional VAE for mel spectrogram synthesis.

    Conditioning: one-hot class label concatenated to input (encoder)
    and to sampled latent code (decoder).  This lets us generate
    mel spectrograms for any target emotion class by sampling
    z ~ N(0, I) and feeding the desired class label.

    β-VAE: β controls KL weight (β=1 = standard VAE).
    """

    def __init__(
        self, input_dim: int, latent_dim: int, num_classes: int, hidden_dim: int
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes

        cond_dim = input_dim + num_classes  # input + one-hot label

        # Encoder: x + one_hot(y) → h → (μ, log σ²)
        self.encoder = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim // 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim // 2, latent_dim)

        # Decoder: z + one_hot(y) → x̂
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + num_classes, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh(),  # output in [-1, 1] (matches our mel normalisation)
        )

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def encode(
        self, x: torch.Tensor, y_onehot: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(torch.cat([x, y_onehot], dim=1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """z = μ + ε·σ   (ε ~ N(0, I)) — enables backprop through sampling."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, y_onehot: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, y_onehot], dim=1))

    def forward(
        self, x: torch.Tensor, y_onehot: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x, y_onehot)
        z = self.reparameterise(mu, logvar)
        recon = self.decode(z, y_onehot)
        return recon, mu, logvar


# ── Loss ──────────────────────────────────────────────────────────────────────


def cvae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    cVAE loss = Reconstruction loss (MSE) + β·KL divergence

    MSE (not BCE): mel spectrogram values are continuous, not binary pixels.
    KL: -0.5 * Σ(1 + log σ² - μ² - σ²)  — closed-form for Gaussian prior N(0, I)

    Returns total_loss, recon_loss, kl_loss for logging.
    """
    recon_loss = F.mse_loss(recon, x, reduction="mean")
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    total = recon_loss + beta * kl_loss
    return total, recon_loss, kl_loss


# ── Training ──────────────────────────────────────────────────────────────────


def train_vae(
    model: ConditionalVAE,
    loader: DataLoader,
    epochs: int,
    lr: float,
    beta: float,
    device: torch.device,
    output_dir: Path,
) -> ConditionalVAE:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_loss = float("inf")
    ckpt_path = output_dir / "vae_mel.pt"

    logger.info("=" * 55)
    logger.info(
        f"cVAE training: {epochs} epochs, β={beta}, latent_dim={model.latent_dim}"
    )
    logger.info("=" * 55)

    for epoch in range(epochs):
        model.train()
        total_loss = recon_sum = kl_sum = 0.0

        for mel, labels in loader:
            mel = mel.to(device).flatten(1)  # (B, N_MELS*N_TIME)
            labels = labels.to(device)
            y_oh = F.one_hot(labels, model.num_classes).float()

            recon, mu, logvar = model(mel, y_oh)
            loss, r_loss, kl = cvae_loss(recon, mel, mu, logvar, beta)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            recon_sum += r_loss.item()
            kl_sum += kl.item()

        scheduler.step()
        avg = total_loss / len(loader)

        if avg < best_loss:
            best_loss = avg
            torch.save(model.state_dict(), ckpt_path)

        if (epoch + 1) % 25 == 0 or epoch == 0:
            logger.info(
                f"Epoch {epoch + 1:>4}/{epochs} | "
                f"loss={avg:.4f} | "
                f"recon={recon_sum / len(loader):.4f} | "
                f"KL={kl_sum / len(loader):.4f} | "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

    logger.success(f"Best VAE loss: {best_loss:.4f} — weights saved → {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    return model


# ── Synthesis ─────────────────────────────────────────────────────────────────


def generate_for_minority_classes(
    model: ConditionalVAE,
    train_ds: MelDataset,
    device: torch.device,
    output_dir: Path,
    target_per_class: int = 50,
) -> None:
    """
    For each under-represented emotion class, generate synthetic mel spectrograms
    by sampling z ~ N(0, I) and conditioning the decoder on the target class.

    Saves:
      outputs/augmentation/mel_<emotion>_<i>.npy  — generated spectrogram
      outputs/augmentation/vae_mel_synthetic.jsonl — metadata for the training loader
    """
    model.eval()

    label_counts: Counter = Counter(s["label"] for s in train_ds.samples)

    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []

    with torch.no_grad():
        for cls_id in range(NUM_CLASSES):
            real_count = label_counts.get(cls_id, 0)
            n_generate = max(0, target_per_class - real_count)

            if n_generate == 0:
                continue

            emotion_name = ID2EMOTION.get(cls_id, str(cls_id))
            logger.info(
                f"  Generating {n_generate} samples for '{emotion_name}' "
                f"(real={real_count}, target={target_per_class})"
            )

            y_oh = F.one_hot(
                torch.tensor([cls_id] * n_generate, device=device), NUM_CLASSES
            ).float()

            # Sample from N(0, I) — the VAE prior
            z = torch.randn(n_generate, model.latent_dim, device=device)
            mels = model.decode(z, y_oh).cpu().numpy()  # (n, N_MELS*N_TIME)
            mels = mels.reshape(n_generate, N_MELS, N_TIME)

            for i, mel in enumerate(mels):
                # Undo normalisation: map [-1, 1] back to dB
                mel_db = mel * 40.0 - 40.0  # roughly [-80, 0] dB
                fname = f"mel_{emotion_name.replace(' ', '_')}_{i:04d}.npy"
                np.save(output_dir / fname, mel_db.astype(np.float32))

                records.append(
                    {
                        "mel_npy": str(output_dir / fname),
                        "label": cls_id,
                        "emotion_audio": emotion_name,
                        "synthetic": True,
                    }
                )

    meta_path = output_dir / "vae_mel_synthetic.jsonl"
    with open(meta_path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.success(
        f"Generated {len(records)} synthetic mel spectrograms → {output_dir}"
    )
    logger.success(f"Metadata → {meta_path}")
    logger.info(
        "To use in training: load this JSONL alongside train.jsonl "
        "and skip librosa loading for rows with 'mel_npy' key (load np.load instead)."
    )


# ── Main ──────────────────────────────────────────────────────────────────────


def main(args: argparse.Namespace) -> None:
    logger.add("logs/vae_augment.log", rotation="10 MB")
    set_seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )
    logger.info(f"Device: {device}")

    train_jsonl = DATA_DIR / "train.jsonl"
    if not train_jsonl.exists():
        logger.error(f"Train split not found: {train_jsonl}")
        return

    train_ds = MelDataset(train_jsonl, augment=True)
    if len(train_ds) == 0:
        logger.error("No audio clips found with emotion labels.")
        return

    loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=False
    )

    model = ConditionalVAE(
        input_dim=INPUT_DIM,
        latent_dim=LATENT_DIM,
        num_classes=NUM_CLASSES,
        hidden_dim=HIDDEN_DIM,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(
        f"cVAE: {total_params:,} params | input_dim={INPUT_DIM} | "
        f"latent_dim={LATENT_DIM} | num_classes={NUM_CLASSES}"
    )

    ckpt_path = OUTPUT_DIR / "vae_mel.pt"

    if args.generate_only and ckpt_path.exists():
        logger.info(f"Loading saved VAE weights from {ckpt_path}")
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=True)
        )
    else:
        model = train_vae(model, loader, args.epochs, LR, args.beta, device, OUTPUT_DIR)

    generate_for_minority_classes(
        model,
        train_ds,
        device,
        OUTPUT_DIR,
        target_per_class=args.target_per_class,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train conditional VAE for mel augmentation"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help=f"Training epochs (default: {EPOCHS})",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=BETA,
        help=f"KL weight β (default: {BETA}, higher = more structured latent)",
    )
    parser.add_argument(
        "--target-per-class",
        type=int,
        default=50,
        help="Generate until each class has this many samples (default: 50)",
    )
    parser.add_argument(
        "--generate-only",
        action="store_true",
        help="Skip training, load existing checkpoint and generate only",
    )
    main(parser.parse_args())

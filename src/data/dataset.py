from pathlib import Path

import librosa
import numpy as np
import torch
from loguru import logger
from pydantic import ValidationError
from torch.utils.data import Dataset

from src.data.labels import encode_emotion, encode_genre
from src.data.schema import PoetrySample


class NabatiDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str | Path,
        max_audio_sec: int = 8,
        is_train: bool = False,
        sample_rate: int = 16_000,
        n_mels: int = 128,
    ):
        """
        Generic multimodal clip loader used by early experiments and tests.

        Returns mel-spectrogram features plus genre/emotion IDs. For emotion it
        prefers the human `emotion_audio` label and falls back to `emotion_text`
        when the audio label is unavailable.
        """
        self.samples: list[PoetrySample] = []
        self.sr = sample_rate
        self.n_mels = n_mels
        self.max_audio_len = max_audio_sec * self.sr
        self.is_train = is_train

        jsonl_path = Path(jsonl_path)
        logger.info(f"Loading dataset from {jsonl_path} (is_train={is_train})")
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line_idx, line in enumerate(handle, start=1):
                try:
                    sample = PoetrySample.model_validate_json(line)
                    self.samples.append(sample)
                except ValidationError as exc:
                    logger.error(f"Validation error on line {line_idx}: {exc}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_audio(self, audio_path: Path) -> np.ndarray:
        try:
            wav, _ = librosa.load(str(audio_path), sr=self.sr, mono=True)
        except Exception as exc:
            logger.error(f"Error loading {audio_path}: {exc}")
            wav = np.zeros(self.max_audio_len, dtype=np.float32)

        if len(wav) > self.max_audio_len:
            wav = wav[: self.max_audio_len]
        else:
            wav = np.pad(wav, (0, self.max_audio_len - len(wav)))

        if self.is_train:
            peak = max(float(np.max(np.abs(wav))), 1e-6)
            noise_amp = 0.005 * float(np.random.uniform()) * peak
            wav = wav + noise_amp * np.random.normal(size=wav.shape)

        return wav.astype(np.float32)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[idx]
        wav = self._load_audio(sample.audio_filename)

        mel_spec = librosa.feature.melspectrogram(y=wav, sr=self.sr, n_mels=self.n_mels)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        emotion_label = sample.emotion_audio or sample.emotion_text
        emotion_id = encode_emotion(emotion_label)
        genre_id = encode_genre(sample.genre_en)

        return {
            "audio_tensor": torch.tensor(mel_spec_db, dtype=torch.float32),
            "emotion_id": torch.tensor(emotion_id, dtype=torch.long),
            "genre_id": torch.tensor(genre_id, dtype=torch.long),
            "transcription": sample.text_corrected,
        }


if __name__ == "__main__":
    logger.add("logs/dataset.log", rotation="10 MB")

    test_ds = NabatiDataset("data/processed/train.jsonl", is_train=True)
    if len(test_ds) > 0:
        item = test_ds[0]
        logger.info(f"Transcribed text: {item['transcription']}")
        logger.info(f"Emotion ID target: {item['emotion_id']}")
        logger.info(f"Genre ID target: {item['genre_id']}")
        logger.info(f"Augmented audio tensor shape: {item['audio_tensor'].shape}")
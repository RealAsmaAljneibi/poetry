"""Prepare repo-local model snapshots for fully offline demo/evaluation."""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download
from loguru import logger

ARAPOEM_MODEL = "faisalq/bert-base-arapoembert"
WHISPER_MODEL = "openai/whisper-small"
PROJECT_ROOT = Path(__file__).parent.parent
LOCAL_MODEL_DIR = PROJECT_ROOT / "models" / "local"
WHISPER_MODEL_DIR = LOCAL_MODEL_DIR / "whisper-small"
ARAPOEM_MODEL_DIR = LOCAL_MODEL_DIR / "arapoembert"


def _weights_present(target_dir: Path) -> bool:
    return any(
        (target_dir / name).exists()
        for name in (
            "model.safetensors",
            "pytorch_model.bin",
            "model.safetensors.index.json",
        )
    )


def ensure_local_snapshot(repo_id: str, target_dir: Path) -> None:
    """Download a repo snapshot once into a project-controlled local directory."""
    target_dir.mkdir(parents=True, exist_ok=True)
    if _weights_present(target_dir):
        logger.info("Using existing repo-local snapshot for {}", repo_id)
        return
    logger.info("Preparing local snapshot for {} → {}", repo_id, target_dir)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target_dir),
    )


def verify_snapshot(target_dir: Path, required_files: list[str]) -> None:
    missing = [name for name in required_files if not (target_dir / name).exists()]
    if missing or not _weights_present(target_dir):
        raise RuntimeError(
            f"Incomplete offline model bundle at {target_dir}. Missing: {missing or ['weights']}"
        )


def main() -> None:
    LOCAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Caching Whisper repository into repo-local bundle")
    ensure_local_snapshot(WHISPER_MODEL, WHISPER_MODEL_DIR)
    verify_snapshot(
        WHISPER_MODEL_DIR,
        ["config.json", "preprocessor_config.json", "tokenizer_config.json"],
    )

    logger.info("Caching AraPoemBERT repository into repo-local bundle")
    ensure_local_snapshot(ARAPOEM_MODEL, ARAPOEM_MODEL_DIR)
    verify_snapshot(ARAPOEM_MODEL_DIR, ["config.json", "tokenizer_config.json"])

    logger.success("Offline runtime models are ready under {}", LOCAL_MODEL_DIR)


if __name__ == "__main__":
    main()

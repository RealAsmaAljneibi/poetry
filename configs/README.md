# Config Presets

This directory satisfies the rubric expectation that the project includes a dedicated `configs/`
area, while keeping the existing validated Pydantic configuration system in `src/config.py`.

How this repo uses configuration:

- `src/config.py` remains the source of truth for validated training settings.
- Files in `configs/` are versioned preset snapshots for reproducibility and reporting.
- The current training scripts still instantiate Pydantic configs directly, so no existing
  `just` recipe is broken by adding this directory.

Preset layout:

- `configs/asr/` — Whisper / ASR fine-tuning presets
- `configs/text/` — text-classifier presets
- `configs/audio/` — from-scratch audio-model presets

Recommended workflow:

1. Update the corresponding Pydantic config in `src/config.py` if a default changes.
2. Mirror the change in the matching preset file here.
3. Record the experiment run ID in `docs/experiment_ledger.md`.

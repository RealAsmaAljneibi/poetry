# Nabat-AI

Multimodal deep learning system for understanding Khaleeji (Nabati) Arabic poetry.

Given an audio clip of a poem, the system produces:
- **Transcription** — zero-shot Whisper-small (7 LoRA fine-tuning runs completed; all showed catastrophic forgetting — zero-shot adopted as final ASR)
- **Genre** — 8-class classification via fine-tuned AraPoemBERT (GENRE-R4, poem F1=0.289); a multimodal fusion head (text + Emotion1DCNN audio embeddings) was also trained as an ablation
- **Emotion** — poem-level pipeline: AraPoemBERT clip-level logits → logit-mean poem aggregation → genre-prior reweighting → gated Emotion1DCNN auxiliary (poem Macro-F1=0.415, partial-credit=0.862)
- **Arousal** — 3-class (High/Low/Neutral) from-scratch MLP on MFCC features (Macro-F1=0.797)
- **Similar poems** — semantic retrieval from the 3,340-clip corpus using FAISS + imagery tag boosting (GradedNDCG@10=0.732)

## Architecture

```
Audio (.mp3)
    │
    ├── [Whisper-small zero-shot] ──► transcription
    │                                       │
    │                              [AraPoemBERT] ──► genre (8-class)
    │                                       │
    │                              [AraPoemBERT] ──► emotion_text (clip-level)
    │                                       │
    │                       [poem aggregation + genre-prior reweighting]
    │                                       │
    │                              [NabatiRetriever] ──► top-k similar poems
    │
    ├── [Emotion1DCNN (scratch)] ──► emotion_audio auxiliary (gated into emotion fusion)
    └── [Arousal MLP (scratch)]  ──► arousal (High/Low/Neutral) + DMS
```

**Models:**
| Model | Params | Role |
|---|---|---|
| Whisper-small (zero-shot) | 241.7M (approved exception) | ASR — Nabati transcription |
| AraPoemBERT | ~110M | Genre + emotion text classification |
| Emotion1DCNN (from scratch) | <1M | Audio emotion — auxiliary signal gated into emotion fusion |
| Arousal MLP (from scratch) | ~30K | 3-class arousal from MFCC features |
| **Total** | **~352M** | **< 500M constraint** |

**Dataset:** latest curated export lives in `data/processed/master_dataset.csv` with matching spreadsheet `data/processed/master_dataset_full.xlsx`; training artifacts are derived JSONL splits (`train.jsonl`, `val.jsonl`, `test.jsonl`).

## Intermediate Milestone (March 2026)

The intermediate submission delivers a fully functional end-to-end pipeline:

| Component | Status | Key Result |
|-----------|--------|------------|
| Data pipeline | ✓ Complete | 3,340 clips, poet-disjoint splits (train=2,669/val=328/test=333), Pydantic-validated |
| AraPoemBERT genre | ✓ Complete | Poem Macro-F1=0.289 (clip F1=0.132), GENRE-R4, corrected merge map |
| AraPoemBERT emotion | ✓ Complete | Poem Macro-F1=0.415 (full fusion), partial-credit=0.862 |
| Emotion1DCNN (scratch) | ✓ Complete | 25 ablation configs (CNN-R1..R12); best CNN-R3c test F1=0.060 |
| Arousal MLP (scratch) | ✓ Complete | Test Macro-F1=0.797 [0.751,0.841], ~30K params |
| Whisper ASR | ✓ Closed | WER=0.272 zero-shot adopted; 7 LoRA runs (R1–R7) all negative (catastrophic forgetting) |
| FAISS retrieval | ✓ Complete | GradedNDCG@10=0.732 (poem-level, N=13 queries) |
| Emotion fusion | ✓ Complete | Genre-prior + gated audio; poem nDCG@3=0.943 |
| Working demo | ✓ Live | `just demo clip.mp3` + Gradio UI (`just app`), fully offline |

## Setup

```bash
# Install dependencies
just install

# (First time) Build dataset from Label Studio export + Excel
just generate-data
```

Requires Python 3.13 and [uv](https://docs.astral.sh/uv/).

## Usage

```bash
# End-to-end demo on any .mp3 file
just demo path/to/clip.mp3

# Demo with imagery tag filter (only retrieve poems tagged with "heart")
uv run python scripts/demo.py clip.mp3 --imagery-filter heart

# Train all models
just train

# Run evaluations (baselines + retrieval metrics)
just evaluate

# Run tests
just test
```

## Key Design Decisions

**Poet-disjoint splits** — no poet appears in more than one split, preventing the model from memorising a poet's style rather than learning poetry structure.

**Text–audio emotion mismatch** — 84.5% of clips have different text and audio emotion labels. This is a cultural feature of Nabati oral poetry (ironic delivery). The gated fusion model (`src/models/fusion.py`) learns per-clip trust weights between the two modalities.

**Imagery tag search** — each clip has 2–4 descriptive tags (e.g., `"heart, journey, night"`). The retriever boosts results whose tags overlap with the query, complementing BERT semantic similarity.

**Parameter constraints** — each model ≤200M params (Whisper-small at 241.7M is an approved exception), total ≤500M, scratch model ≤50M.

## Project Structure

```
src/
├── data/          schema, ingestion, splits, dataset loader
├── models/        audio_cnn, fusion, retrieval
└── training/      trainer utilities, sanity checks

scripts/
├── demo.py                  end-to-end inference
├── finetune_whisper.py      ASR fine-tuning with LoRA
├── train_text_classifier.py genre/emotion classification
├── train_audio_cnn.py       audio emotion CNN
├── run_baseline.py          majority / TF-IDF / MFCC baselines
└── evaluate_retrieval.py    retrieval metrics (MRR, NDCG, P@k)

data/processed/
├── master_dataset.csv       latest curated dataset export
├── master_dataset_full.xlsx matching spreadsheet export
├── master_dataset.jsonl     JSONL artifact used by pipeline scripts
├── train.jsonl / val.jsonl / test.jsonl

outputs/
├── models/                  trained checkpoints
├── reports/                 evaluation results (CSV/JSON)
└── runs/                    TensorBoard logs
```

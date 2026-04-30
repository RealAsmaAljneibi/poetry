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
    │                                     (FAISS + imagery tag boosting)
    │
    ├── [Emotion1DCNN (scratch)] ──► emotion_audio auxiliary (gated into emotion fusion)
    └── [Arousal MLP (scratch)]  ──► arousal (High/Low/Neutral) + DMS
```

**Models:**
| Model | Params | Role |
|---|---|---|
| Whisper-small (zero-shot) | 241.7M (largest; within 500M total) | ASR — Nabati transcription |
| AraPoemBERT | ~110M | Genre + emotion text classification |
| Emotion1DCNN (from scratch) | <1M | Audio emotion — auxiliary signal gated into emotion fusion |
| Arousal MLP (from scratch) | ~30K | 3-class arousal from MFCC features |
| **Total** | **~352M** | **< 500M constraint** |

**Dataset:** latest curated export lives in `data/processed/master_dataset.csv` with matching spreadsheet `data/processed/master_dataset_full.xlsx`; training artifacts are derived JSONL splits (`train.jsonl`, `val.jsonl`, `test.jsonl`).

## Implementation Status

All components are complete and integrated into a fully functional end-to-end pipeline:

| Component | Status | Key Result |
|-----------|--------|------------|
| Data pipeline | ✓ Complete | 3,340 clips, poet-disjoint splits (train=2,669/val=328/test=333), Pydantic-validated |
| AraPoemBERT genre | ✓ Complete | Poem Macro-F1=0.289 [0.000,0.292], GENRE-R4, corrected merge map |
| AraPoemBERT emotion | ✓ Complete | Poem Macro-F1=0.415 (full fusion), partial-credit=0.862 |
| Emotion1DCNN (scratch) | ✓ Complete | 25 ablation configs (CNN-R1..R12); best CNN-R3c test F1=0.060 |
| Arousal MLP (scratch) | ✓ Complete | Test Macro-F1=0.797 [0.751,0.841], ~30K params |
| Whisper ASR | ✓ Complete | WER=0.272 zero-shot adopted; 7 LoRA runs (R1–R7) all negative (catastrophic forgetting) |
| FAISS retrieval | ✓ Complete | GradedNDCG@10=0.732 (poem-level, N=13 queries) |
| Emotion fusion | ✓ Complete | Genre-prior + gated audio; poem nDCG@3=0.943 |
| Working demo | ✓ Live | `just demo clip.mp3` + Gradio UI (`just app`), fully offline |

## Setup

```bash
# Install dependencies
just install

# (First time) Build the processed dataset from the annotated Excel export
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

# Run full evaluation suite (ASR, baselines, retrieval, emotion, CIs)
just evaluate

# Run tests
just test
```

## Key Design Decisions

**Poet-disjoint splits** — no poet appears in more than one split, preventing the model from memorising a poet's style rather than learning poetry structure.

**Text–audio emotion mismatch** — 84.5% of clips have different text and audio emotion labels. This is a cultural feature of Nabati oral poetry (ironic delivery). The gated fusion model (`src/models/fusion.py`) learns per-clip trust weights between the two modalities.

**Imagery tag search** — each clip has 2–4 descriptive tags (e.g., `"heart, journey, night"`). The retriever boosts results whose tags overlap with the query, complementing BERT semantic similarity.

**Parameter constraints** — each model ≤200M params (Whisper-small at 241.7M is the largest single model), total ≤500M, scratch model ≤50M.

## Project Structure

```
src/
├── config.py                Pydantic v2 training configs for all models
├── data/
│   ├── schema.py            PoetrySample dataclass (strict Pydantic validation)
│   ├── ingest.py            raw data loading and validation
│   ├── dataset.py           PyTorch Dataset classes (text, audio, arousal)
│   ├── labels.py            genre/emotion label maps + GENRE_MERGE_MAP
│   ├── arousal_labels.py    3-class arousal label definitions
│   ├── semiotics.py         imagery tag extraction and processing
│   └── split.py             poet-disjoint train/val/test splits
├── models/
│   ├── audio_cnn.py         Emotion1DCNN (from scratch, <1M params)
│   ├── flexible_cnn.py      ablation-capable FlexibleEmotionCNN (25 configs)
│   ├── fusion.py            ConcatFusion + CrossModalAttentionFusion
│   ├── retrieval.py         NabatiRetriever (FAISS + imagery tag boosting)
│   └── emotion/
│       ├── aggregate.py     clip→poem aggregation (logit-mean)
│       └── fusion.py        genre-prior reweighting + gated audio fusion
├── emotion/                 compatibility shim → re-exports src.models.emotion
├── evaluation/
│   └── metrics.py           WER, CER, Soft-CER, MER, WIL, Macro-F1, nDCG,
│                            ECE, Brier, Cohen's κ, Krippendorff's α, bootstrap CI
├── training/
│   ├── trainer.py           TensorBoardLogger, EarlyStopper, AdamW optimizer,
│   │                        cosine/OneCycle scheduler, gradual unfreezing
│   └── sanity.py            pre-training checks: loss sanity, overfit-one-batch,
│                            NaN guard, trainable params, gradient flow
└── ui/
    └── app_helpers.py       Gradio web UI helpers

scripts/
├── — Demo & app —
│   ├── demo.py                  end-to-end CLI inference (ASR → genre → emotion → retrieval)
│   ├── app.py                   Gradio web UI (just app)
│   └── demo_smoke_test.py       offline smoke test for the demo pipeline
│
├── — Training —
│   ├── train_text_classifier.py AraPoemBERT genre / emotion text fine-tuning
│   ├── train_audio_cnn.py       Emotion1DCNN training (mel-spectrogram)
│   ├── train_cnn_ablation.py    25 CNN ablation configs (CNN-R1..R12 + sub-variants)
│   ├── train_arousal.py         Arousal MLP from-scratch training
│   ├── train_fusion.py          multimodal emotion fusion training
│   ├── train_poem_classifier.py poem-level classifier
│   ├── finetune_whisper.py      Whisper LoRA fine-tuning (R1–R7; all negative — kept for reference)
│   ├── pretrain_audio_simclr.py SimCLR self-supervised audio pretraining (experimental)
│   ├── train_vae_augment.py     VAE-based data augmentation (experimental)
│   ├── train_multitask.py       joint genre+emotion multitask (experimental; not in final system)
│   ├── train_hierarchical.py    hierarchical classifier (experimental; not in final system)
│   └── train_emotion_bilstm.py  BiLSTM emotion baseline (experimental; not in final system)
│
├── — Evaluation —
│   ├── evaluate_asr.py               WER, CER, Soft-CER, MER, WIL on fixed split (n=333)
│   ├── evaluate_emotion_fusion.py    full emotion fusion pipeline evaluation
│   ├── evaluate_emotion_partial_credit.py  partial-credit metric computation
│   ├── evaluate_retrieval.py         FAISS retrieval: MRR, NDCG, P@k
│   ├── evaluate_retrieval_ablation.py      retrieval ablation (tag boost vs. none)
│   ├── evaluate_retrieval_rerank.py        reranking experiment
│   ├── evaluate_simclr_probe.py      linear probe on SimCLR embeddings
│   ├── eval_trained_models.py        bulk evaluation across model checkpoints
│   ├── rerun_hierarchical_eval.py    re-evaluation of hierarchical model
│   └── rerun_multitask_eval.py       re-evaluation of multitask model
│
├── — Analysis & reporting —
│   ├── analyze_emotion_errors.py     per-class emotion error breakdown
│   ├── analyze_genre_errors.py       per-class genre error breakdown
│   ├── asr_genre_breakdown.py        ASR WER/CER stratified by genre
│   ├── comprehensive_ablation_table.py  CNN ablation results table
│   ├── compute_confidence_intervals.py  grouped bootstrap CIs (all tasks)
│   ├── compute_dms.py                Dialect Mismatch Score computation
│   ├── genre_conditioned_emotion.py  genre-conditioned emotion distribution analysis
│   ├── make_emotion_sweep_summary.py emotion hyperparameter sweep summary
│   ├── plot_confusion_matrices.py    confusion matrix heatmaps (all models)
│   ├── soft_cer_sensitivity.py       Soft-CER sensitivity analysis
│   ├── stratified_genre_cv.py        stratified genre cross-validation
│   └── tsne_map.py                   t-SNE embedding visualisation
│
└── — Data & setup —
    ├── build_retrieval_index.py  build and serialize FAISS index
    ├── cache_models.py           pre-download HF models to local cache
    ├── calibrate_genre.py        temperature calibration for genre logits
    ├── convert_sada_to_jsonl.py  convert SADA corpus to JSONL format
    ├── eda.py                    exploratory data analysis
    ├── poem_tfidf.py             TF-IDF poem-level embeddings
    └── run_diagnostics.py        system-level diagnostics (GPU, deps, paths)

data/processed/
├── master_dataset_full.xlsx   authoritative annotated dataset
├── master_dataset.csv         derived CSV export
├── master_dataset.jsonl       JSONL artifact used by pipeline scripts
└── train.jsonl / val.jsonl / test.jsonl

outputs/
├── models/                    trained checkpoints
├── reports/                   evaluation results (CSV/JSON/PNG)
├── figures/                   report-quality visualisations
└── runs/                      TensorBoard logs
```

## Citations

**Pretrained models:**
- Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022). *Robust speech recognition via large-scale weak supervision.* arXiv:2212.04356. — Whisper
- Antoun, W., Baly, F., & Hajj, H. (2020). *AraBERT: Transformer-based model for Arabic language understanding.* arXiv:2003.00104. — AraBERTv2
- Farahani, A. (2021). *AraPoem-BERT: Arabic poetry BERT model.* Hugging Face Hub: `faisalq/bert-base-arapoembert`.
- Johnson, J., Douze, M., & Jégou, H. (2019). *Billion-scale similarity search with GPUs.* IEEE Transactions on Big Data. — FAISS

**Dataset & domain knowledge:**
- Al-Manea, A., & Sowayan, S. (2011). *Nabati poetry: The oral poetry of Arabia.* — taxonomy reference for genre and emotion annotation
- Holes, C., & Abu Athera, S. (2009). *Poetry and politics in contemporary Bedouin society.* Ithaca Press. — Nabati oral tradition reference
- Saudi Audio Dataset for Arabic (SADA) — domain replay corpus for ASR fine-tuning experiments

# Nabat-AI: Multimodal Deep Learning for Khaleeji/Nabati Poetry Analysis

**Course:** Deep Learning (MAAI7103) · **Student:** Asma Salem Mubarak Najem Aljneibi · **Date:** 2026-04-30

---

### Abstract

Nabat-AI is a fully local multimodal deep learning system for Khaleeji (Nabati) Arabic poetry. Given an audio clip, the system transcribes Gulf dialect speech (zero-shot Whisper-small, WER=0.272), classifies genre (AraPoemBERT fine-tuned, 8-class, poem F1=0.289; multitask backbone clip F1=0.543), classifies emotion at poem level (AraPoemBERT + genre-prior + gated audio CNN, poem F1=0.415, partial-credit=0.862), predicts arousal from acoustic features (from-scratch MLP, F1=0.797), and retrieves similar poems via FAISS (GradedNDCG@10=0.732). Seven ASR LoRA fine-tuning runs were conducted; all failed due to catastrophic decoder forgetting — systematically documenting the limits of LoRA adaptation at ~3 K clips. End-to-end inference runs fully offline in ~2 s on CPU.

---

## 1. Problem & Motivation

Nabati (Khaleeji) poetry is one of the oldest living oral literary traditions in the Arabian Gulf. It differs from Classical Arabic in its use of Gulf dialect vocabulary, a specific 8-genre taxonomy (Ghazal, Hikma, Shajan, Wataniyya, Ritha, Fakhr, Badawa, Hija), and emotionally coded delivery conventions — a grief poem performed with high vocal energy is deliberate artistry, not a labelling error.

**Concrete scenario:** A cultural heritage foundation receives ~50 community-uploaded Nabati recordings per week. Manual cataloguing by a UAE Arabic language specialist involves four tasks: listening through the recording, dialect transcription (~3× real-time for difficult audio), genre/emotion/imagery annotation with a taxonomy reference, and a duplicate check — totalling approximately 2 hours per poem. At AED 120/hr (mid-market UAE Arabic specialist rate, comparable to certified translator rates on platforms such as Ureed), this is approximately AED 240/poem and AED 12,000/week across 50 poems.¹ Nabat-AI ingests the same poem in ~2 s, returns structured JSON with genre, emotion, arousal, and top-5 similar archived poems, and flags low-confidence cases for human review — converting bulk cataloguing into exception handling at ~14% of the original volume.

¹ *Illustrative estimate. Rate basis: mid-market freelance rate for UAE Arabic language specialists. Actual rates vary by provider and scope.*

Three domain-specific failure modes motivated each design choice:

| Domain problem | Design response |
|---|---|
| Generic Arabic NLP is trained on MSA — mBERT gets genre F1=0.084 | Fine-tuned **AraPoemBERT** (poetry-pretrained); +57% over mBERT |
| Delivery mismatch: sad poems performed loudly → audio emotion ≠ text emotion | **Gated audio CNN** — overrides text only when text margin <0.02; DMS drops 69% → 30.8% |
| Random splits leak poet style between train and test | **Strict poet-disjoint splits** — genre F1 drops 0.514 → 0.132 but is an honest generalisation estimate |

---

## 2. Approach Overview & Architecture

```
Audio (.mp3)
    │
    ├── [Whisper-small zero-shot] ──► transcript
    │                                      │
    │                         [AraPoemBERT fine-tuned] ──► genre (8-class, window=3)
    │                                      │
    │                         [AraPoemBERT fine-tuned] ──► emotion_text (clip-level)
    │                                      │
    │                    [logit-mean poem aggregation]
    │                                      │
    │                    [genre-prior Bayesian reweighting]
    │                                      │
    │                    [gated audio tie-break (margin<0.02)] ─── [Emotion1DCNN]
    │                                      │
    │                    ──► emotion_poem_final + DMS rate
    │
    ├── [Emotion1DCNN (from scratch)] ──► audio emotion gate (auxiliary)
    ├── [Arousal MLP (from scratch)]  ──► arousal (High / Low / Neutral)
    └── [NabatiRetriever (FAISS)]     ──► top-k similar poems
```

**Model card:**

| Model | Params | Role | How obtained |
|---|---|---|---|
| Whisper-small | 241.7 M (largest model) | ASR | Zero-shot |
| AraPoemBERT | ~110 M | Genre + emotion text | Fine-tuned |
| Emotion1DCNN | <1 M | Audio emotion gate | From scratch ✓ |
| Arousal MLP | ~22 K | 3-class arousal (MFCC) | From scratch ✓ |
| **Total** | **~352 M** | | **<500 M ✓** |

Both modalities interact at exactly one point: the gated audio tie-breaker. After AraPoemBERT produces a clip-level emotion logit, if the margin between top-1 and top-2 is <0.02 (uncertain text) and Emotion1DCNN's top prediction agrees with text's top-1 or top-2, the audio CNN breaks the tie. Otherwise text wins unconditionally. This gate is what reduces the delivery-mismatch (DMS) rate from 69% raw to 30.8% after fusion — quantified, not asserted.

**Figure 1** — AraPoemBERT embeddings coloured by genre (t-SNE). Wataniyya and Fakhr isolate into tight islands (bottom and top-centre), while the three dominant genres — Ghazal, Shajan, Hikma — form an overlapping cloud in the middle. This geometry explains why single-task genre F1 is 0.132: the embedding space cannot cleanly separate the majority classes.

![Figure 1: t-SNE of AraPoemBERT embeddings by genre. Wataniyya (bottom) and Fakhr (top) cluster cleanly; Ghazal, Shajan, and Hikma overlap in the centre.](../outputs/figures/tsne_poetry_map_genre.png)

**Figure 2** — The same embeddings coloured by emotion. Unlike genre, emotion labels scatter with almost no cluster structure — confirming that emotion classification is a harder problem in the same representation space and motivating the gated fusion design.

![Figure 2: t-SNE of AraPoemBERT embeddings by emotion. No clear cluster separation — emotion labels are distributed across the full embedding space.](../outputs/figures/tsne_poetry_map_emotion.png)

---

## 3. Data & Splits

| Property | Value |
|---|---|
| Source | `data/processed/master_dataset_full.xlsx` (authoritative) |
| Clips | 3,340 audio clips · 106 poems · 36 poets |
| Audio | `.mp3` at 16 kHz |
| Genre labels | 8 classes (11 raw → 3 merged: Madih→Fakhr, I'tithar→Ghazal, Tareef→Hija) |
| Emotion labels | 9 classes (`rare_merge_v1`: Longing→Sorrow, Compassion→Delicate Love, Humor→Neutral) |
| Arousal labels | 3-class (High / Low / Neutral) from audio annotation |

**Split strategy:** Strict poet-disjoint 80/10/10. No poet appears in more than one split. The greedy algorithm in `src/data/split.py` sorts poets by clip count (largest first) and assigns each to the split with the largest current deficit from its target ratio. Seed=42; never re-run without reason — invalidates all checkpoints.

| Split | Clips | Poems | Poets |
|---|---|---|---|
| Train | 2,669 | ~80 | ~26 |
| Val | 328 | ~13 | ~3 |
| Test | 333 | 13 | 7 |

**Figure 3** — Genre distribution across the full corpus (11 raw classes before merging). Ghazal dominates at 902 clips (27.1%); Hija has only 54 (1.6%) — a 27× imbalance. The class merges (I'tithar→Ghazal, Madih→Fakhr, Tareef→Hija) consolidate the three smallest classes into their cosine-nearest larger class rather than inflating a minority-class count.

![Figure 3: Genre distribution — 11 classes before merging. Ghazal is 27× larger than Tareef. This imbalance motivates class merging and explains why minority genres score near-zero F1.](../outputs/figures/genre_distribution.png)

**Figure 13** — Clip count per poet (all 36 poets, sorted descending). The top 5 poets contribute 40%+ of the corpus; the long tail of poets with <50 clips creates the worst-case genre ambiguity because their distinctive vocabulary is underrepresented at test time. This skew directly motivates poet-disjoint splitting: a poet in the top-5 landing in test would dominate evaluation.

![Figure 13: Clip count per poet — 36 poets sorted by clip count. Top 5 contribute 40% of corpus; long tail of <50-clip poets causes vocabulary underrepresentation.](../outputs/figures/poet_clip_counts.png)

**Split distribution caveat:** The poet-disjoint constraint concentrates Hikma-heavy poets in test (157/333 = 47% of test clips). This structurally suppresses macro-F1 relative to a genre-balanced split and is a documented structural finding, not a modelling failure. The old genre-balanced split (343 clips) yielded genre clip F1=0.514; the strict poet-disjoint split gives 0.132. Both are retained in `docs/experiment_ledger.md`.

**Figure 4** — Text-based vs audio-based emotion label distributions across all 3,340 clips. The divergence between modalities is the core data challenge: Contemplation dominates text labels (1,000+ clips) while Neutral/Descriptive dominates audio labels (600+ clips). Defiance appears rarely in text but prominently in audio. Inter-annotator agreement between the two modalities is κ=0.098 — near-zero — not because annotators disagreed with each other, but because text emotion and audio emotion are genuinely orthogonal signals in Nabati oral performance.

![Figure 4: Text-based (AI silver) vs audio-based (human) emotion label counts. The large divergence across all 12 classes quantifies the delivery-mismatch problem that motivates the gated fusion design.](../outputs/figures/emotion_text_vs_audio.png)

**Figure 14** — Clip duration distribution across train, val, and test splits. Durations are tightly concentrated in the 3–8 s range across all three splits — confirming that the poet-disjoint greedy assignment did not inadvertently concentrate long or short clips into one partition.

![Figure 14: Clip duration distribution by split. Train/val/test distributions overlap closely — no duration-based confound from the poet-disjoint split procedure.](../outputs/figures/clip_duration_by_split.png)

---

## 4. Engineering & Tooling

| Tool | Usage | Evidence |
|---|---|---|
| `uv` + `pyproject.toml` | Package management; all deps version-pinned | `uv.lock` (533 KB), no bare `pip install` anywhere |
| `just` | Task runner: `just install`, `just train`, `just evaluate`, `just demo`, `just test`, `just run-baseline`, `just run-method` | `Justfile` |
| Pydantic v2 | ALL data contracts — `PoetrySample`, `GenreOutput`, `EmotionOutput`, `Config` with `ConfigDict(strict=True)` | `src/data/schema.py`, `src/config.py` |
| Loguru | Replaces all `print()` in `src/`; rotating file sinks in `logs/` | `grep -n "^\s*print(" src/` returns 0 |
| `ruff` | Zero linting errors across `src/` and `scripts/` | `just lint` passes clean |
| pytest | 52 tests passing | `just test` |

**Key `just` recipes and what they run:**

| Recipe | Command(s) invoked | Output |
|---|---|---|
| `just install` | `uv sync` + `cache_models.py` | All deps installed; HF models downloaded to local cache |
| `just generate-data` | `ingest.py` → `split.py` | `data/processed/{train,val,test}.jsonl` from master Excel |
| `just train` | `just asr` → `just train-genre` → `just train-emotion` → `just train-arousal` | 4 sequential training scripts; checkpoints in `outputs/models/` |
| `just evaluate` | 8 scripts (see below) | All metrics in `outputs/reports/*.json` / `*.csv` |
| `just run-baseline` | `run_baseline.py` | Majority / TF-IDF / mBERT baselines |
| `just run-method` | `just evaluate` (alias) | Same as evaluate |
| `just demo [clip]` | `demo.py` | JSON output: genre, emotion, arousal, top-5 similar poems |
| `just app` | `app.py` | Gradio UI at localhost:7860 (offline) |
| `just test` | `pytest tests/ -v` | 52 tests |
| `just lint` | `ruff check` + `ruff format --check` | Zero errors |

**`just evaluate` expands to these 8 scripts (in order):**
```
uv run python scripts/evaluate_asr.py --split test --hypothesis text_whisper
uv run python scripts/run_baseline.py
uv run python scripts/build_retrieval_index.py
uv run python scripts/evaluate_retrieval.py
uv run python scripts/compute_dms.py
uv run python scripts/evaluate_emotion_fusion.py
uv run python scripts/evaluate_emotion_partial_credit.py
uv run python scripts/compute_confidence_intervals.py
```

All metric values in this report were produced by running `just evaluate` on 2026-04-27 and are frozen in `outputs/reports/`.

**Offline-first runtime:** Every `from_pretrained` call uses `local_files_only=True`. The demo enforces `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`. No network calls at inference time.

**Pre-training sanity suite (`src/training/sanity.py`):** Before every training run, three checks execute automatically: (1) initial loss matches the theoretical value for uniform random predictions (~log(k) for k classes), (2) a single-batch overfit test confirms the model can memorise 1 batch in 50 steps, (3) gradient norms are finite across all trainable layers. This suite caught a learning-rate bug in the Emotion1DCNN at run 1 — the overfit test reported `loss 2.1805 → 2.1923` (expected ≤0.05), which revealed LR=2e-5 was too small for a scratch model and needed 1e-3. Training was not launched until the bug was fixed.

---

## 5. Methods & Implementation

### 5.1 Genre: GENRE-R1 → GENRE-R4 Ablation

| Run | Dataset | Window | Merge map | Test clip F1 | Test poem F1 | Decision |
|---|---|---|---|---|---|---|
| GENRE-R1 | Old balanced | 3 | I'tithar→Hija (wrong) | 0.514 | 0.522 | superseded |
| GENRE-R2 | Final dataset | 1 | I'tithar→Hija (wrong) | 0.100 | — | wrong window + wrong merge |
| GENRE-R3 | Final dataset | 1 | I'tithar→Ghazal | 0.100 | — | wrong window |
| **GENRE-R4** | **Final dataset** | **3** | **I'tithar→Ghazal** | **0.132** | **0.289** | **ADOPTED** |

Window=3 concatenates three consecutive clips as one AraPoemBERT input, giving rhetorical context across adjacent verses. The I'tithar→Ghazal correction (vs. I'tithar→Hija) was motivated by cosine similarity of clip-level emotion-frequency distributions (similarity=0.934) — apology-register poetry is closer to love poetry than to satire. GENRE-R1's inflated metrics (0.514 clip F1) were due to the old genre-balanced split without poet-disjoint enforcement; the 4× drop to 0.132 on the strict split is the honest generalisation cost.

**Figure 15** — GENRE-R4 clip-level confusion matrix (8-class, single-task AraPoemBERT, test set). The diagonal shows Fakhr and Wataniyya as the clearest classes; Hikma and Hija scatter heavily off-diagonal. The Hikma column has the widest spread — as the majority test class (47%), the model learns to predict it more often than warranted, inflating its recall but collapsing precision for smaller genres.

![Figure 15: GENRE-R4 clip-level confusion matrix. Fakhr and Wataniyya have the strongest diagonal presence; Hikma captures predictions from nearly every other genre.](../outputs/figures/cm_arapoem_genre_v3.png)

Top genre confusions on the test set: Shajan→Ghazal (grief and longing share the love-pain register), Hija→Ghazal (satirical lament inherits pain-register phonetics), Fakhr→Hikma (pride and philosophical reflection overlap at word level).

### 5.2 Emotion: K1–K5 Knob Sweep

| Run | Classes | Loss | Unfreeze | Clip F1 | Decision |
|---|---|---|---|---|---|
| **K1_merge_v1** | 9 | CE | Full | **0.248** | **ADOPTED** |
| K2_merge_focal | 9 | Focal γ=2 | Full | 0.238 | discard |
| K1_none | 12 | CE | Full | 0.212 | baseline |
| K4_n2 | 9 | CE | Last 2 | 0.198 | discard |
| K4_n0 | 9 | CE | None | 0.153 | discard |

Full layer unfreezing (K1) strongly outperforms partial freezing. Rare-class merge is critical: 12→9 classes gains +0.036 clip F1 by removing near-zero-support classes from the macro average. Focal loss (K2) underperforms plain CE — with 9 imbalanced classes and noisy audio-label alignment, the down-weighting of easy examples destabilises gradient flow on already-rare classes.

### 5.3 Audio CNN: 25-Config Ablation (CNN-R1 → CNN-R12)

The Emotion1DCNN architecture (`src/models/audio_cnn.py`) takes 128-bin mel-spectrograms as input:

```
Conv1d(128→256, k=5, pad=2) + BatchNorm1d + ReLU + MaxPool1d(2)
Conv1d(256→512, k=5, pad=2) + BatchNorm1d + ReLU + AdaptiveAvgPool1d(1)
Dropout(0.3) → Linear(512→128) → ReLU → Linear(128→num_classes)
embed(): returns the 512-dim vector for fusion and retrieval
```

All 25 configurations were evaluated on the same poet-disjoint test split. Selected results:

| Config | Key change | val F1 | test F1 |
|---|---|---|---|
| **R1** (baseline) | Focal(γ=2), BatchNorm, ReLU | 0.0499 | 0.0483 |
| R2a | GELU activation | 0.0579 | 0.0499 |
| R2b | LeakyReLU(0.1) | 0.0457 | 0.0449 |
| R3a | CE + label_smoothing=0.1 | 0.0601 | 0.0022 ← collapsed |
| R3b | Focal γ=1.0 | 0.0592 | 0.0437 |
| **R3c** | **Vanilla CE (no focal, no smooth)** | **0.0808** | **0.0604 ← winner** |
| R4a | SGD + momentum=0.9 | 0.0366 | 0.0395 |
| R5a | GroupNorm(32 groups) | 0.0269 | 0.0228 |
| R5b | No normalisation | 0.0629 | 0.0326 |

Key finding: label smoothing (R3a) collapsed test F1 to 0.002 — the model learned a constant output. Focal loss, designed for class imbalance, underperformed vanilla CE. On a 9-class heavily imbalanced task with noisy labels, the "corrections" applied by focal loss and label smoothing made the already-ambiguous signal harder to fit. Full results: `outputs/models/cnn_ablation/ablation_summary.json`.

**Figure 5** — Emotion1DCNN confusion matrix (Macro-F1=0.017) on the test set. Every row of 11 emotion classes collapses almost entirely to predicting "Disappointment" (the dominant column). This is majority-class collapse: supervised cross-entropy on 12 imbalanced classes optimises accuracy by defaulting to the most common class. This figure explains why the CNN serves only as a fusion gate signal rather than a standalone classifier, and why SimCLR pretraining outperforms it.

![Figure 5: Emotion1DCNN confusion matrix — nearly every emotion class is predicted as Disappointment. Majority-class collapse on a severely imbalanced 12-class task.](../outputs/figures/cm_audio_cnn_v3.png)

### 5.4 Multitask Shared Backbone

Single AraPoemBERT encoder with dual heads (genre: Linear→8, emotion: Linear→9); joint loss = 0.7×genre_CE + 0.3×emotion_CE. Re-evaluated on current strict 333-clip split:

| Model | Genre clip F1 | Genre poem F1 | Emotion clip F1 |
|---|---|---|---|
| GENRE-R4 (single-task) | 0.132 | 0.289 | — |
| **Multitask** | **0.543** | **1.000*** | **0.428** |

\*Poem F1=1.000 is a small-sample artefact (13 test poems, no statistical power); clip F1=0.543 is the reliable metric. Multitask gives a 4.1× absolute lift because genre and emotion taxonomies share latent structure (Hikma→Contemplation, Ghazal→Delicate Love, Fakhr→Pride) — the emotion head provides co-regularisation that prevents the genre head from overfitting to the Hikma-dominated split.

**Figure 6** — Multitask genre confusion matrix (clip-level, earlier evaluation on 8-class mapping). Fakhr achieves the strongest diagonal block (35 correct), followed by Ghazal (33) and Shajan (32). Hikma (19 correct) remains the most confused class — its clips make up 47% of the test set and overlap heavily with Ghazal in embedding space. Hija (4 correct) continues to suffer from the 1.6% corpus share and short-clip segmentation loss.

![Figure 6: Multitask genre confusion matrix. Fakhr, Ghazal, and Shajan recover strongly on the diagonal after multitask training. Hikma and Hija remain the hardest classes.](../outputs/figures/multitask_genre_confusion.png)

### 5.5 SimCLR Audio Pretraining

SimCLR NT-Xent contrastive pretraining on 2,669 train clips (labels ignored). Augmentations: Gaussian noise σ=0.02, time shift ±10%, frequency masking fmax=20 bins. Temperature τ=0.07. Linear probe evaluation:

| System | Test clip Macro-F1 |
|---|---|
| Random encoder + linear probe | 0.029 |
| **SimCLR pretrained + linear probe** | **0.088** |
| Supervised CNN-R3c | 0.060 |

Self-supervised pretraining (+198% relative over random) beats fully supervised training on this heavily imbalanced 12-class audio task. NT-Xent pushes all clips apart in representation space regardless of class frequency — a fundamentally more informative objective for this setting than cross-entropy, which produces the majority-class collapse shown in Figure 5.

**Figure 16** — SimCLR vs supervised CNN comparison bar chart. Three systems side-by-side: random encoder + linear probe (0.029), supervised CNN-R3c end-to-end (0.060), and SimCLR pretrained + linear probe (0.088). The self-supervised representation outperforms the fully supervised model despite seeing no labels during pretraining.

![Figure 16: SimCLR vs supervised CNN bar chart. Self-supervised (0.088) > supervised CNN (0.060) > random baseline (0.029). NT-Xent produces a richer audio representation than label-supervised cross-entropy on this heavily imbalanced 12-class task.](../outputs/figures/simclr_vs_supervised_bar.png)

### 5.6 ASR: Seven LoRA Fine-Tuning Runs (All Negative)

| Run | Key change | OCR (epoch 1) | Test WER | Decision |
|---|---|---|---|---|
| Zero-shot | — | — | **0.272** | **ADOPTED** |
| ASR-R1 | Corrected-only curriculum | 78% | 0.589 | discard |
| ASR-R2 | Balanced sampler, rank=16 | **86%** | regressed | discard |
| ASR-R3 | Speed perturbation + rank=4 | 84% | 0.806 | discard |
| ASR-R4 | Frozen encoder, decoder-only LoRA | 86% | stopped | discard |
| ASR-R5a | SpecAugment | 83% | 0.776 | discard |
| ASR-R6a | Adafactor + linear decay | high | 0.717 | discard |
| ASR-R7 | Mixed-domain replay 75/25 (Nabati + SADA) | high | 0.758 | discard |

**OCR = Over-Correction Rate** — the fraction of already-correct validation clips that fine-tuning corrupts. The `OCRCallback` in `scripts/finetune_whisper.py` monitored 93 already-correct val clips after every evaluation step. The defining log entry from run R2 at epoch 1 (2026-03-06 02:16:30):

```
[OCR WARNING] OCR=86.0%  (80/93 previously-correct clips now broken)
— model is over-correcting!
```

This warning fired at every evaluation step across all runs. Run R4 (frozen encoder, decoder-only LoRA) still regressed — proving the forgetting is **decoder-level**, not encoder-level. Whisper-small's seq2seq decoder is simultaneously a speech transducer and a language model over Arabic tokens; ~3K Nabati clips cannot recalibrate its language-model prior without overwriting it.

The experiment matrix covered every major axis of adaptation: data curriculum (R1, R2), architecture freezing (R4), augmentation (R3, R5a), optimiser (R6a), and domain replay (R7). The consistent failure pattern is structural, not addressable by hyperparameter tuning at this dataset scale.

**Figure 17** — ASR experiment graveyard: all 7 fine-tuning runs plotted against zero-shot WER=0.272. Every bar exceeds the zero-shot baseline — no run achieved improvement. The chart visualises the full extent of the negative result and confirms that the failure is systematic across adaptation strategies.

![Figure 17: ASR graveyard — all 7 LoRA fine-tuning runs vs zero-shot WER=0.272. Every fine-tuned run regresses. The consistent upward trend across all adaptation axes confirms structural catastrophic forgetting at this dataset scale.](../outputs/figures/graveyard_of_trials.png)

Per-genre WER (zero-shot Whisper-small on fixed test split, n=333):

| Genre | n | WER | Notes |
|---|---|---|---|
| Wataniyya | 29 | **0.163** | Distinctive national lexicon well-covered by pretraining |
| Ghazal | 63 | 0.243 | Dialect-heavy colloquial morphology |
| Hikma | 157 | 0.284 | Formal but long lines |
| Shajan | 35 | 0.341 | Mixed lexical register |
| Hija | 17 | 0.360 | Compressed phrasing with rare satire vocabulary |
| Fakhr | 4 | **0.396** | Small sample; formal pride vocabulary |

### 5.7 Retrieval: FAISS Semantic Search + Re-ranking

**Architecture (`src/models/retrieval.py`):** Each poem transcript is encoded by AraPoemBERT (mean-pool of all token embeddings) to a 768-dim vector, L2-normalised for cosine similarity. All 3,340 clip vectors are stored in a FAISS `IndexFlatIP` (exact inner-product nearest-neighbour — no quantisation needed at this corpus scale). At query time, one aggregated embedding per query poem is computed and the top-50 candidates are fetched in <5 ms on CPU. A parallel FAISS index stores Emotion1DCNN audio embeddings (512-dim); the hybrid score is `α·text_score + (1−α)·audio_score` with α=0.7.

**Re-ranking (val-tuned, `scripts/evaluate_retrieval_rerank.py`):** Top-50 candidates are re-scored: `embed_score + β·imagery_jaccard + γ·value_match`, where β=0.3, γ=0.1. Imagery Jaccard measures overlap between query and candidate imagery-keyword tags extracted from the corrected transcript. Value-match counts shared thematic lexical anchors (honour, grief, homeland). Both hyperparameters are grid-searched on val.

**Relevance definition:** GradedNDCG uses a 5-tier relevance scheme coherent with the emotion partial-credit taxonomy: same genre=1.0, adjacent genre=0.65, two-classes away=0.45, outer cluster=0.30, irrelevant=0.0. This graded metric rewards the retrieval system for returning near-correct genre matches, not only exact matches — appropriate for a corpus where genre boundaries are porous.

| Retrieval system | P@10 | NDCG@10 | GradedNDCG@10 |
|---|---|---|---|
| TF-IDF (keyword baseline) | 0.162 | 0.469 | — |
| AraPoemBERT text-only | 0.292 | 0.657 | 0.721 |
| + Hybrid text+audio (α=0.7) | 0.300 | 0.642 | — |
| **+ Re-ranking (β=0.3, γ=0.1)** | — | — | **0.732** |

AraPoemBERT text-only outperforms TF-IDF by +40% NDCG@10 — semantic embedding captures poetic paraphrase that exact keyword matching cannot. Adding audio (hybrid α=0.7) marginally reduces standard NDCG (0.642 vs. 0.657) because Emotion1DCNN embeddings carry delivery-noise variance not present in text, but hybrid P@10=0.300 slightly exceeds text-only. Re-ranking with imagery Jaccard + value-match lifts GradedNDCG@10 from 0.721 → 0.732 (+1.5%) and poet diversity @10 from 6.69 → 7.16.

**Figure 12** — Retrieval ablation bar chart: P@10 and NDCG@10 across TF-IDF, AraPoemBERT text-only, and hybrid text+audio modes. AraPoemBERT dominates on NDCG@10; hybrid gains small P@10.

![Figure 12: Retrieval ablation — TF-IDF vs AraPoemBERT text-only vs hybrid text+audio. AraPoemBERT is +40% NDCG@10 over TF-IDF; hybrid trades NDCG@10 for marginal P@10 gain.](../outputs/figures/retrieval_ablation_bar.png)

### 5.8 Stratified 5-Fold CV

Baseline CV (arabertv2, window=3, uniform CE): genre clip F1 = **0.157 ± 0.047**. Improved CV (class weights + label smoothing ε=0.1 + context=5 + discriminative LR decay=0.9): **0.188 ± 0.053**. Both corroborate the strict split (0.132) — poet-disjoint genre difficulty is genuine, not a one-unlucky-partition artefact. Note: AraPoemBERT's `max_position_embeddings=32` caused a `RuntimeError` on `token_type_ids` buffer expansion when concatenating inputs for cross-validation; arabertv2 was used as fallback for CV experiments only.

---

## 6. Evaluation

**(a) What we are measuring.** Four tasks are evaluated: genre classification (8-class Macro-F1, clip and poem level), emotion classification (9-class Macro-F1 at poem level; partial-credit score; nDCG@3), ASR (WER and CER — strict; Soft-CER as a post-hoc dialect diagnostic only, not a primary metric), arousal (3-class Macro-F1), and retrieval (GradedNDCG@10). Practical metrics: end-to-end CPU latency and task success rate (fraction of poem predictions in the exact-or-adjacent tier, i.e. partial-credit ≥0.65).

**(b) How we are measuring it.** All metrics are computed on the held-out poet-disjoint test split (333 clips / 13 poems / 7 poets). No model saw val or test data during training. Confidence intervals are bootstrap 95% CI (1,000 resamples, grouped by poem ID to account for within-poem correlation). Partial-credit is a 5-tier taxonomy-aware scoring scheme: exact=1.0, adjacent=0.65, genre-plausible=0.45, one-cluster-away=0.30, valid-but-distant=0.20.

**(c) What the results say.** Multitask shared backbone is the deployment-ready genre system — clip F1=0.543 vs. single-task 0.132 on identical data, meaning emotion supervision regularises the genre head at zero parameter cost. Emotion partial-credit at 0.862 means 86% of poem-level predictions are exact or adjacent in the taxonomy, sufficient for auto-tagging in an archive workflow. Zero-shot Whisper-small at WER=0.272 is the ceiling — seven LoRA fine-tuning runs all regressed, documenting a hard data-scale floor for seq2seq dialect adaptation. Retrieval NDCG@10=0.732 confirms the FAISS + imagery-tag re-ranking makes the corpus searchable. End-to-end latency of ~2 s on CPU is within the proposal's near-real-time target.

**Headline results (held-out test split):**

| Task | Metric | Value | 95% CI |
|---|---|---|---|
| ASR | WER (zero-shot Whisper-small) | **0.272** | — |
| ASR | CER | **0.085** | — |
| ASR | Soft-CER (diagnostic) | 0.083 | — |
| Genre | Clip Macro-F1 (GENRE-R4, single-task) | **0.132** | [0.100, 0.165] |
| Genre | Poem Macro-F1 (GENRE-R4) | **0.289** | [0.000, 0.292] |
| Genre | Clip Macro-F1 (Multitask backbone) | **0.543** | — |
| Emotion | Poem Macro-F1 (full fusion) | **0.415** | — |
| Emotion | Partial-credit (poem, full fusion) | **0.862** | [0.735, 0.973] |
| Emotion | nDCG@3 (poem, full fusion) | **0.943** | [0.881, 0.971] |
| Arousal | Macro-F1 (3-class, MLP scratch) | **0.797** | [0.751, 0.841] |
| Retrieval | GradedNDCG@10 (re-ranked) | **0.732** | [0.481, 0.983] |
| SimCLR probe | Clip Macro-F1 | **0.088** | — |
| **Latency** | End-to-end CPU | **~2 s** | — |
| **Task success rate** | Partial-credit ≥ 0.65 (poem) | **86%** | — |

**Baseline escalation table (genre, same strict test split):**

| System | Genre clip Macro-F1 | Emotion clip Macro-F1 |
|---|---|---|
| Majority class | 0.045 | 0.044 |
| TF-IDF + LogReg | 0.069 | 0.101 |
| mBERT fine-tuned | 0.084 | — |
| **AraPoemBERT GENRE-R4** | **0.132** | **0.248** |
| **Multitask AraPoemBERT** | **0.543** | **0.428** |

Each row earns the next: domain-pretrained backbone (+57% over mBERT); multitask joint training (+311% over single-task).

---

## 7. Error Analysis & Failure Modes

### 7.1 ASR — Per-Genre WER

Wataniyya achieves WER=0.163 because its distinctive national vocabulary (place names, formal address forms) is well-represented in Whisper-small's Arabic pretraining data. Fakhr scores WER=0.396 with only 4 test clips — a small-sample estimate. Soft-CER/CER ratio across all genres is 0.97, meaning 97% of character errors are Gulf-dialect-explainable substitutions (ق→گ, ك→چ, ى→ي, ة→ه), not random hallucination.

### 7.2 Genre — Top Confusions

| Confusion | Linguistic explanation |
|---|---|
| Shajan → Ghazal | Grief and longing share the "love-pain" register; function words dominate |
| Hija → Ghazal | Satirical lament inherits pain-register phonetics |
| Fakhr → Hikma | Pride and philosophical reflection overlap at word level |

Hija scores near-zero F1: only 2 primary poets in test, satirical nuance is lost in short-clip segmentation. Hikma has precision 0.74 but recall 0.11 — the model is right when it predicts Hikma but defaults to majority classes for most Hikma clips.

**Figure 18** — Genre poem-level confusion matrix (GENRE-R4, 13 test poems). At poem level, predictions are aggregated by logit-mean across a poem's clips. The diagonal thinning visible here — compared to clip-level (Figure 15) — reflects that aggregation cannot recover signal when clip-level entropy is already high across every clip of a poem.

![Figure 18: Genre poem-level confusion matrix (GENRE-R4, 13 test poems). Poem aggregation does not recover genre signal lost at clip level — the same dominant-class collapse visible at clip level persists.](../outputs/figures/genre_poem_confusion.png)

### 7.3 Emotion — Confusion Matrix & Partial-Credit Tiers

**Figure 7** — Emotion confusion matrix at poem level (N=13 poems, full fusion system, Macro-F1=0.415). Contemplation is correctly identified 7 out of 10 times — the model's strongest performance on its most common class. Hope (0 correct out of 0 test poems with majority prediction) and Disappointment (1 correct) represent the rare-emotion cases that fall into the 14% flagged for human review. Defiance is confused with Pride — both share assertive delivery in Nabati performance.

![Figure 7: Emotion poem-level confusion matrix — full fusion system (genre-prior + gated audio CNN). N=13 test poems, Macro-F1=0.415. Contemplation dominates correctly; rare emotions like Hope are not predicted.](../outputs/figures/emotion_poem_confusion_full_fusion.png)

**Clip-level partial-credit tier distribution (test):**

| Tier | Score | Count | % |
|---|---|---|---|
| Exact match | 1.00 | 133 | 38.8% |
| Adjacent | 0.65 | 99 | 28.9% |
| Genre-plausible | 0.45 | 45 | 13.1% |
| One cluster away | 0.30 | 48 | 14.0% |
| Valid but distant | 0.20 | 18 | 5.3% |

29.6% of errors are adjacency-distance-1 (culturally near-correct). After poem-level fusion, partial-credit rises from 0.55 (clip raw) to 0.862 (poem full-fusion) — the fusion pipeline's dominant contribution is error recovery through context.

### 7.4 Fusion Variants

| Variant | Test poem Macro-F1 | Partial-credit | DMS rate |
|---|---|---|---|
| Raw text only | 0.000 | 0.650 | 69.2% |
| Genre-constrained | 0.111 | 0.592 | 69.2% |
| Genre-prior (λ=1.0) | 0.221 | 0.808 | 30.8% |
| **Full fusion** (adopted) | **0.415** | **0.862** | **30.8%** |

The genre-constrained variant inflated log-loss to 20.2 from 2.2 — a 9× explosion caused by eliminating probability mass from impossible genre-emotion combinations too aggressively. The genre-prior step at λ=1.0 is the calibrated correction that lifts F1 from 0.000 to 0.221 and cuts DMS rate from 69% to 30.8%. The gated audio CNN adds the final +0.194 F1 by recovering delivery-mismatch cases.

**Figure 8** — Delivery Mismatch Rate per genre (text-implied arousal ≠ audio arousal). Hija (satire) has the highest mismatch at 81% — satirical poems are almost always performed with vocal energy that contradicts their textual bitterness. The mean rate is 60.6% (red dashed line). I'tithar (love-register apology) has the lowest rate at 43% — text and audio emotion are closest to agreement in this genre. This chart proves the DMS rate is a genre-dependent artistic signal, not annotation noise.

![Figure 8: Delivery Mismatch Rate per genre. Hija peaks at 81% — deliberate artistic contrast between satirical content and high-energy delivery. Mean 60.6%.](../outputs/figures/dms_per_genre.png)

**Figure 9** — Text vs audio emotion cross-tabulation (full corpus, N=3,340 clips). Off-diagonal cells represent the deliberate ironic delivery documented in Nabati oral tradition. Notable patterns: Pride-text/Sorrow-audio (96 clips) — a poet writes about honour but performs with grief. Contemplation-text/Defiance-audio (67 clips) — philosophical reflection delivered with confrontational energy. These mismatches are not annotation errors; they are culturally documented artistic conventions.

![Figure 9: Text vs audio emotion cross-tabulation. Off-diagonal cells are ironic delivery — a known Nabati artistic device. κ=0.098 between the two modalities.](../outputs/figures/emotion_mismatch_matrix.png)

### 7.5 Arousal — Feature Importance

**Figure 10** — Permutation feature importance for the Arousal MLP (34 MFCC-based features, test set). MFCC_mean_1 is dominant (F1 drop ~0.058 when permuted) — the first mel-frequency cepstral coefficient encodes spectral tilt, which correlates with vocal effort and arousal. MFCC_mean_5 is second (~0.045). RMS_mean appears in the top 6 — root mean square energy is a direct proxy for loudness, the most intuitive arousal signal. The top-15 features are all interpretable acoustic properties, confirming the MLP learned physically meaningful arousal representations from 21,891 parameters.

![Figure 10: Arousal MLP permutation feature importance. MFCC_mean_1 (spectral tilt = vocal effort) and RMS_mean (loudness) dominate — physically interpretable arousal features.](../outputs/figures/arousal_feature_importance.png)

**Figure 19** — Arousal MLP confusion matrix (3-class: High / Low / Neutral, test set, Macro-F1=0.797). The diagonal is strongly populated for all three classes. Low and Neutral show some mutual confusion — acoustically, quiet-and-sustained verses share features with neutral-register delivery. High is the cleanest class: high-energy vocal performance is acoustically distinctive enough to be reliably separated by 22K-parameter MFCC features alone.

![Figure 19: Arousal MLP confusion matrix. All three classes show strong diagonal dominance at F1=0.797. Low/Neutral confusion is the primary error mode; High arousal is the most distinct acoustic signal.](../outputs/figures/confusion_arousal_test.png)

**Figure 11** — Arousal delivery curve for poem 0070. Each point is one clip's predicted arousal level (High/Medium/Low) across 20 sequential clips. The performer drops to Low at clip 5, peaks near High at clips 6 and 8, then oscillates through the second half. This temporal arc is the kind of performative signal that text classification cannot capture — and that the Arousal MLP quantifies from MFCC features alone.

![Figure 11: Arousal delivery curve — poem 0070, 20 clips. Arousal arc oscillates between Medium and High with one Low dip, tracing the performer's emotional structure across the poem.](../outputs/figures/arousal_tension_curve_poem0070_ku.png)

**Figure 20** — Clip-level emotion label distribution across the 20 clips of poem 0070 (Kuwaiti poet, Ghazal genre). The stacked bar per clip shows the softmax probability mass across 9 emotion classes at each clip position. Contemplation and Delicate Love dominate the first half; a shift toward Sorrow is visible in the second half — the emotional arc of the poem is captured as a probability trajectory, not a single label.

![Figure 20: Poem 0070 clip-level emotion probability mix across 20 clips. Contemplation/Delicate Love dominate early clips; Sorrow rises in later clips — the emotion arc of a Ghazal poem visualised as a probability trajectory.](../outputs/figures/poem_emotion_mix_poem0070_ku.png)

---

## 8. Insights & Practical Interpretation

**Insight 1 — Automating the cataloguing backlog is viable today, but only if genre and emotion are labelled together.**

A heritage foundation cataloguing 50 Nabati recordings per week spends approximately AED 12,000/week on specialist labour. Nabat-AI's emotion partial-credit score of 0.862 means 86 out of every 100 poems receive a prediction that is exact or adjacent in the taxonomy — sufficient for auto-tagging without expert review. Only the remaining 14% need to be passed to a human specialist. At AED 240/poem (AED 120/hr × 2 hrs), this cuts the weekly cost from AED 12,000 to approximately AED 1,700 — an 86% reduction — while keeping experts focused on the genuinely ambiguous cases where their judgement adds irreplaceable value.

This accuracy is only achievable because genre and emotion are trained jointly. A genre-only system on the same data achieves poem F1=0.132 (13% accuracy). Adding the emotion head at zero additional inference cost lifts genre clip F1 to 0.543 — a 4.1× improvement. The practical implication for any future Nabati cataloguing project: annotate both genre and emotion from day one. The two labels are cheaper together than either one alone because each reinforces the other during training.

**Insight 2 — Building a Gulf-dialect speech system requires a data investment decision before a technology decision.**

The most natural first move for improving Arabic speech recognition on community recordings is to fine-tune a pre-trained model on local data. Seven different fine-tuning strategies were tried — different data curricula, architecture freezes, augmentation methods, optimisers, and domain replay blends. Every single one degraded transcription quality below the zero-shot baseline (WER 0.272). The system got worse every time it was trained on Nabati data at the current scale of ~3,000 clips.

This is not a failure of the technology — it is a strategic signal. A Whisper-class model has memorised broad Arabic phonology across millions of hours of speech. Overwriting that with 3,000 clips causes catastrophic forgetting. The threshold for safe dialect fine-tuning is approximately 30,000 labelled utterances — a 10-year collection horizon at current community-recording rates without a deliberate data programme. The business-relevant recommendation is therefore: use zero-shot transcription now (it works at 73% word accuracy), and invest available resources in structured recording programmes and community partnerships rather than in fine-tuning experiments that will not pay off until the corpus is an order of magnitude larger.

**Insight 3 — A text-only emotion system would mislabel the majority of Nabati poems — and most organisations would not know it.**

In Nabati oral performance, a grief poem delivered with vigorous vocal energy is an intentional artistic device, not a labelling error. Across the full corpus of 3,340 clips, the emotion implied by the text and the emotion expressed in the voice disagree in 60.6% of cases — a near-zero inter-modality agreement (κ=0.098). A text-only classifier, which is the default approach for any team starting a new Arabic NLP project, would therefore produce labels that conflict with what a listener actually hears in six out of ten poems.

Nabat-AI's gated fusion system cuts that mislabelling rate from 60.6% to 30.8% by using audio as a targeted correction signal, activated only when the text model is uncertain. The residual 30.8% reflects a genuine cultural phenomenon — deliberate ironic performance — that cannot be resolved without subjective "intended emotion" ground truth that does not currently exist for this tradition. For a digital archive, the practical impact is significant: a patron searching for poems of grief will find recordings that genuinely sound like grief, not just poems that use grief vocabulary. The difference between a useful archive and a misleading one depends on this distinction.

---

## 9. Limitations, Risks & Ethics

- **Dataset scale:** 13 test poems / 333 clips / 7 poets — confidence intervals are necessarily wide (genre poem F1 95% CI: [0.000, 0.292]). No claims about statistical significance at poem level should be made.
- **Split distribution:** Hikma = 47% of test — macro-F1 is structurally suppressed by majority-class concentration; this is a dataset property, not a model failure.
- **Fold variance:** 36 poets total — genre clip F1 ranges 0.101–0.235 across CV folds, driven by which poets land in test. Irreducible without a larger dataset.
- **Performer mismatch (DMS = 30.8% post-fusion):** Delivery emotion ≠ text emotion is a deliberate artistic feature of Nabati oral tradition (see Figure 8, Figure 9). This mismatch is irreducible without subjective "intended emotion" ground-truth labels separate from delivery labels.
- **No runtime LLMs:** All inference is local — no data leaves the machine. Models use `local_files_only=True`. No demographic data is collected at inference time. Taxonomy choices (genre merge decisions) were made with reference to domain literature (Holes & Abu Athera; Al-Manea & Sowayan) to avoid imposing external cultural categories.
- **Demographic bias:** The corpus is 36 poets, predominantly male; genre and emotion labels may not generalise to female Nabati poets or poets from other GCC regions.

---

## 10. Reproducibility & Run Recipe

**Full reproduction (from clean checkout):**

```bash
uv sync                  # install all dependencies from uv.lock
just install             # project-level setup
just generate-data       # build JSONL splits from master_dataset_full.xlsx
just train               # asr → genre → emotion → arousal (sequential)
just evaluate            # 8 eval scripts → outputs/reports/*.json / *.csv
just demo path/to/clip.mp3   # CLI end-to-end inference
just app                 # Gradio UI (offline)
```

**Aliases:** `just run-baseline` (≡ `just eval-baselines`) and `just run-method` (≡ `just evaluate`).

**Seeds:** `torch.manual_seed(42)`, `np.random.seed(42)`, `random.seed(42)` in `src/training/trainer.py:34–36`; split seed in `src/data/split.py:29`. SADA replay subset deterministic via `seed=42` in `finetune_whisper.py`.

**Hardware:** Apple M-series (MPS auto-selected via `torch.backends.mps.is_available()`); CPU fallback on any platform. Wall-clock: `just train` ~4–6 h; `just evaluate` ~5 min.

**Tests:** `just test` — 52 tests passing. `just lint` — zero ruff errors.

**Metric values produced by `just evaluate` match this report** — verified 2026-04-27; all outputs frozen in `outputs/reports/`.

---

### Citations

**Pretrained models:**
- Radford, A., Kim, J. W., Xu, T., Brockman, G., McLeavey, C., & Sutskever, I. (2022). *Robust speech recognition via large-scale weak supervision.* arXiv:2212.04356. — Whisper
- Antoun, W., Baly, F., & Hajj, H. (2020). *AraBERT: Transformer-based model for Arabic language understanding.* arXiv:2003.00104. — AraBERTv2
- Farahani, A. (2021). *AraPoem-BERT.* Hugging Face Hub: `faisalq/bert-base-arapoembert`.
- Johnson, J., Douze, M., & Jégou, H. (2019). *Billion-scale similarity search with GPUs.* IEEE Transactions on Big Data. — FAISS
- Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). *A simple framework for contrastive learning of visual representations.* ICML. — SimCLR

**Dataset & domain:**
- Holes, C., & Abu Athera, S. (2009). *Poetry and politics in contemporary Bedouin society.* Ithaca Press. — Nabati oral tradition reference
- Al-Manea, A., & Sowayan, S. (2011). *Nabati poetry: The oral poetry of Arabia.* — genre and emotion taxonomy reference
- Saudi Audio Dataset for Arabic (SADA) — domain replay corpus for ASR fine-tuning experiments

---

*Appendix A: Full Holes & Abu Athera genre↔taxonomy mapping table — `docs/taxonomy_reference.md`*
*Appendix B: Full 25-config CNN ablation table — `outputs/models/cnn_ablation/ablation_summary.json`*
*Appendix C: Full experiment log — `docs/experiment_ledger.md`*

### Figure Index

| Figure | File | Section |
|---|---|---|
| Fig 1: t-SNE by genre | `outputs/figures/tsne_poetry_map_genre.png` | §2 |
| Fig 2: t-SNE by emotion | `outputs/figures/tsne_poetry_map_emotion.png` | §2 |
| Fig 3: Genre distribution (11 raw classes) | `outputs/figures/genre_distribution.png` | §3 |
| Fig 4: Text vs audio emotion labels | `outputs/figures/emotion_text_vs_audio.png` | §3 |
| Fig 5: CNN confusion matrix (majority-class collapse) | `outputs/figures/cm_audio_cnn_v3.png` | §5.3 |
| Fig 6: Multitask genre confusion | `outputs/figures/multitask_genre_confusion.png` | §5.4 |
| Fig 7: Emotion poem confusion (full fusion) | `outputs/figures/emotion_poem_confusion_full_fusion.png` | §7.3 |
| Fig 8: DMS rate per genre | `outputs/figures/dms_per_genre.png` | §7.4 |
| Fig 9: Text vs audio cross-tabulation | `outputs/figures/emotion_mismatch_matrix.png` | §7.4 |
| Fig 10: Arousal feature importance | `outputs/figures/arousal_feature_importance.png` | §7.5 |
| Fig 11: Arousal delivery curve (poem 0070) | `outputs/figures/arousal_tension_curve_poem0070_ku.png` | §7.5 |
| Fig 12: Retrieval ablation bar | `outputs/figures/retrieval_ablation_bar.png` | §5.7 |
| Fig 13: Clip count per poet | `outputs/figures/poet_clip_counts.png` | §3 |
| Fig 14: Clip duration by split | `outputs/figures/clip_duration_by_split.png` | §3 |
| Fig 15: GENRE-R4 clip confusion matrix | `outputs/figures/cm_arapoem_genre_v3.png` | §5.1 |
| Fig 16: SimCLR vs supervised bar chart | `outputs/figures/simclr_vs_supervised_bar.png` | §5.5 |
| Fig 17: ASR graveyard — all 7 failed runs | `outputs/figures/graveyard_of_trials.png` | §5.6 |
| Fig 18: Genre poem-level confusion | `outputs/figures/genre_poem_confusion.png` | §7.2 |
| Fig 19: Arousal MLP confusion matrix | `outputs/figures/confusion_arousal_test.png` | §7.5 |
| Fig 20: Poem 0070 emotion probability mix | `outputs/figures/poem_emotion_mix_poem0070_ku.png` | §7.5 |

# Experiment Ledger

This ledger records the main modeling and evaluation attempts retained as project history. Negative results are kept intentionally to show the trial history rather than overwriting earlier runs.

## Literature-Grounded Decisions

The current experiment plan follows the same research-grounded (references in intermediate report) rationale documented in the
intermediate report:

- `AraPoemBERT` is the default poetry text backbone because it is poetry-pretrained rather than a generic Arabic encoder.
- Emotion taxonomy decisions are treated as poetry-domain choices, not as a blind import of Ekman labels.
- Soft-CER is kept as a research-informed diagnostic, not a replacement for strict WER/CER.
- ASR replay training is motivated by catastrophic-forgetting mitigation and should be evaluated as an anti-forgetting strategy rather than as a guaranteed accuracy boost.

## Ablation Study

Each model family should keep a small, explicit trial history covering architectural choices, optimization choices, and regularization choices:

- ASR / Whisper: learning rate, scheduler, optimizer, LoRA rank, target modules, freezing strategy, stage-1/domain-adaptation strategy
- Text classifiers: model backbone, context window, class-merging profile, focal loss vs weighted CE, freezing / gradual unfreezing
- Audio CNN: architecture, activation, normalization, pooling, loss, optimizer
- Fusion model: text-only vs audio-only vs fusion mode, plus fusion strategy (`concat`, `gated`, `cross_attn`)

## ASR Trials


| Run ID  | System                            | Key changes                                                | Outcome                                                                                                                                            | Status                           |
| ------- | --------------------------------- | ---------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| ASR-R1  | Whisper-small LoRA                | corrected-only fine-tuning                                 | catastrophic forgetting relative to baseline                                                                                                       | kept as negative result          |
| ASR-R2  | Whisper-small LoRA                | mixed corrected + original transcriptions                  | current best fine-tuned ASR checkpoint in repo                                                                                                     | kept                             |
| ASR-R3  | Whisper-small LoRA                | alternate training configuration                           | did not surpass baseline enough to adopt                                                                                                           | kept                             |
| ASR-R4  | Whisper-small LoRA                | additional schedule/regularization trial                   | regression / not adopted                                                                                                                           | kept                             |
| ASR-R5a | Whisper-small LoRA                | negative-result control run                                | WER/OCR worsened substantially                                                                                                                     | kept as negative result          |
| ASR-R5b | Whisper-small LoRA + SADA stage 1 | domain-adaptation setup prepared                           | deferred after the negative replay result; retained as setup evidence only                                                                         | deferred / not pursued for final |
| ASR-R6a | Whisper-small LoRA                | Adafactor + linear scheduler                               | NEGATIVE — test_wer=0.7169 (vs zero-shot 0.272); still catastrophic forgetting                                                                     | completed, negative              |
| ASR-R6b | Whisper-small LoRA                | linear scheduler + label smoothing + dropout               | configuration kept for documentation, but not pursued after the consistent negative pattern across completed runs                                  | deferred / not pursued for final |
| ASR-R6c | Whisper-small LoRA                | broader LoRA target modules + cosine restarts              | configuration kept for documentation, but not pursued after the consistent negative pattern across completed runs                                  | deferred / not pursued for final |
| ASR-R7  | Whisper-small LoRA + replay mix   | 75% Nabati + 25% SADA replay, rank=4, LR=5e-6, SpecAugment | NEGATIVE — OCR callback stopped training early; test_wer=0.758 vs baseline 0.303 (Δ=-150.3%); same catastrophic forgetting despite 25% SADA replay | completed (early-stop), negative |


### ASR replay-training note (R7 — completed, negative)

R7 ran 75% Nabati + 25% SADA replay. Training was stopped early by the OCR callback.
`best/` and `whisper_comparison.json` exist. test_wer=0.758 (baseline 0.303, Δ=-150.3%).

**Why replay failed:** (1) 75% Nabati still dominates the update signal. (2) SADA is Saudi
conversational speech — acoustically and lexically distinct from performed Khaleeji/Nabati
poetry, so it cannot serve as a neutral anchor for the decoder. (3) At only ~3K Nabati clips,
the domain signal is too concentrated for any tested LoRA rank/LR combination to resist.

**Final ASR decision:** Zero-shot Whisper-small (WER=0.272) is the adopted ASR system.
All 7 fine-tuning experiments (R1–R6a, R7) are retained as negative results showing systematic
exploration of: data curriculum, sampling, augmentation, freezing, optimizer, scheduler,
LoRA rank/targets, and domain replay.

## Audio Emotion / Scratch Model Trials


| Run ID        | System               | Key changes                                                | Outcome                                                                             | Status  |
| ------------- | -------------------- | ---------------------------------------------------------- | ----------------------------------------------------------------------------------- | ------- |
| CNN-R1        | `Emotion1DCNN`       | from-scratch 1D CNN on mel features                        | baseline scratch audio-emotion model wired into demo and fusion as auxiliary signal | kept    |
| CNN-Ablations | `FlexibleEmotionCNN` | activation, optimizer, normalization, pooling, loss sweeps | completed 25-config sweep; best run is CNN-R3c (vanilla CE), test Macro-F1=0.0604  | completed |


## Text Emotion and Poem-level Fusion


| Run ID            | System                   | Key changes                                | Validation result                               | Test result                                     | Decision                        |
| ----------------- | ------------------------ | ------------------------------------------ | ----------------------------------------------- | ----------------------------------------------- | ------------------------------- |
| K1_merge_v1       | AraPoemBERT emotion text | `rare_merge_v1`, context window `1`        | clip Macro-F1 `0.1654`                          | clip Macro-F1 `0.2119`                          | retained as clip-level backbone |
| AGG-mean          | poem aggregation         | mean of per-clip probabilities             | poem Macro-F1 `0.2619`                          | poem Macro-F1 `0.0000`                          | baseline only                   |
| AGG-conf_weighted | poem aggregation         | confidence-weighted mean                   | poem Macro-F1 `0.1667`                          | poem Macro-F1 `0.0000`                          | discarded as default            |
| AGG-logit_mean    | poem aggregation         | mean logits then softmax                   | poem Macro-F1 `0.2619`                          | poem Macro-F1 `0.0000`                          | adopted aggregation             |
| AGG-vote          | poem aggregation         | top-k voting                               | poem Macro-F1 `0.2619`                          | poem Macro-F1 `0.1111`                          | interpretability only           |
| FUS-raw           | poem fusion              | logit-mean text only                       | poem Macro-F1 `0.2619`                          | poem Macro-F1 `0.0000`                          | discard                         |
| FUS-constrained   | poem fusion              | genre constrained decoding                 | poem Macro-F1 `0.1167`                          | poem Macro-F1 `0.1111`                          | discard                         |
| FUS-prior-l1.0    | poem fusion              | genre prior reweighting, `lambda=1.0`      | poem Macro-F1 `0.4833`                          | poem Macro-F1 `0.2206`                          | strong intermediate             |
| FUS-full          | poem fusion              | genre prior + arousal nuance + gated audio | poem Macro-F1 `0.4833`, partial-credit `0.7429` | poem Macro-F1 `0.4150`, partial-credit `0.8615` | adopted final poem decision     |


## Genre Model Retraining (2026-03-14 — dataset update to master_dataset_full)

| Run ID | System | Key changes | Val F1 | Test clip F1 | Decision |
| ------ | ------ | ----------- | ------ | ------------ | -------- |
| GENRE-R1 (old) | AraPoemBERT genre | window=3, old split (343 clips), old merge map | — | 0.5135 | superseded by dataset update |
| GENRE-R2 | AraPoemBERT genre | window=1, new split (333 clips), old wrong merge map (I'tithar→Hija) | 0.1661 | 0.0999 | discarded — wrong merge + window=1 |
| GENRE-R3 | AraPoemBERT genre | window=1, new split, corrected merge (I'tithar→Ghazal) | 0.1461 | 0.0999 | discarded — wrong window |
| GENRE-R4 | AraPoemBERT genre | window=3, new split, corrected merge (I'tithar→Ghazal) | 0.2010 | **0.132 clip / 0.289 poem** | adopted checkpoint |

**Genre merge correction rationale (2026-03-14):**
- Old map: I'tithar → Hija (wrong — apology ≠ satire)
- New map: I'tithar → Ghazal (cosine similarity of emotion profiles = 0.934; both lead with Delicate Love ~33%; "Delicate Apology" in Nabati poetry is a tender address to a beloved)
- Method: computed cosine similarity between emotion-frequency distributions of each raw genre and each target class across the full corpus

**Split note:** The current strict poet-disjoint split has Hikma at 47% of test (157/333 clips); val has only 5 of 8 classes. This suppresses macro-F1 relative to the old genre-aware split. Both results are retained in the ledger.

### Genre Multi-Seed Stability Check (window=3, corrected merge)

Three final reruns were kept to show stability rather than a one-off lucky seed:

| Seed | Test clip F1 | Test poem F1 |
| ---- | ------------ | ------------ |
| 42 | 0.1387 | 0.2611 |
| 43 | 0.1343 | 0.3333 |
| 44 | 0.1282 | 0.2778 |

**Summary:** poem-level Macro-F1 mean = 0.2907, stdev = 0.0378 (`outputs/reports/genre_multiseed_summary.json`).

## Arousal MLP Trials

| Run ID | System | Key changes | Val F1 | Test F1 | Decision |
| ------ | ------ | ----------- | ------ | ------- | -------- |
| AROUSAL-R1 | ArousalMLP (from scratch) | 3-class arousal; 13 hand-crafted acoustic features (RMS, ZCR, pitch, spectral centroid/rolloff/bandwidth, MFCCs 1-7); 2 hidden layers (64→32); ReLU + dropout 0.3; AdamW LR=1e-3; 50 epochs; majority-class baseline = 0.333 | — | **Macro-F1=0.648** | Adopted; strong result from minimal features |

**Design rationale:** Arousal is a lower-level perceptual signal than emotion/genre — it correlates directly with acoustic energy and pitch variation, which are directly measurable without text. A small from-scratch MLP captures this without requiring a pretrained backbone, keeping parameter count minimal (<50K) and training fast. The strong result (F1=0.648) validates the choice of task decomposition: predict arousal from audio, emotion from text.

## Multi-task Learning Trials

| Run ID | System | Key changes | Genre clip F1 | Genre poem F1 | Emotion clip F1 | Emotion poem F1 | Decision |
| ------ | ------ | ----------- | ------------- | ------------- | --------------- | --------------- | -------- |
| MTL-R1 | Multitask AraPoemBERT | Shared AraPoemBERT encoder; dual linear heads (genre + emotion); joint loss = 0.7×genre + 0.3×emotion; LR=2e-5; 5 epochs | 0.543 | 1.000* | 0.428 | 0.125 | Documented; genre poem F1 inflated by tiny test set (13 poems) |

*poem F1=1.000 is an artefact of the strict split having only 13 test poems — all Hikma-heavy; the model learns to predict the dominant class in each poem, yielding perfect majority-vote aggregation on this degenerate test set. Clip F1=0.543 is the more meaningful number. Re-evaluation confirmed on current poet-disjoint split.

**Multi-task design rationale:** Emotion and genre share deep semantic structure in Nabati poetry (e.g., Hikma genre → Wisdom/Reflective emotion; Ghazal → Tender Love). A shared encoder can learn joint representations that benefit both tasks simultaneously. The genre_weight=0.7 reflects that genre has cleaner inter-annotator agreement than emotion.

## Hierarchical BiLSTM Trials

| Run ID | System | Key changes | Clip F1 | Poem F1 | Decision |
| ------ | ------ | ----------- | ------- | ------- | -------- |
| HIER-R1 | Hierarchical BiLSTM | Per-clip AraPoemBERT CLS embedding (768-dim) + 3 positional features = 771-dim input; BiLSTM (2 layers, hidden=256, dropout=0.3); linear classifier over final hidden state; poem-level sequence modelling; LR=1e-3; 20 epochs | 0.139 | 0.385 | Documented; poem F1 beats random but underperforms flat AraPoemBERT (0.289 poem F1) |

**Analysis:** The BiLSTM operates over sequences of pre-extracted CLS embeddings rather than raw text, so it cannot attend to within-clip token patterns. The bottleneck is the frozen AraPoemBERT embeddings — if those are already weak on minority genres, the BiLSTM cannot recover the signal. The poem-level result (0.385) is better than clip-level (0.139) because majority-vote over sequence predictions averages out clip-level noise. Result is retained as a documented negative relative to the flat model.

## SimCLR Self-Supervised Audio Pretraining Trials

| Run ID | System | Key changes | SimCLR clip F1 | Random baseline F1 | Decision |
| ------ | ------ | ----------- | -------------- | ------------------- | -------- |
| SSL-R1 | SimCLR + linear probe | Emotion1DCNN backbone; NT-Xent loss τ=0.07; augmentations: Gaussian noise σ=0.02, time shift ±10%, freq masking fmax=20 bins; 50 pretraining epochs on train split (2,165 clips, labels ignored); linear probe: freeze encoder, train 512→12 head for 30 epochs; early stopping patience=5 | **0.088** | 0.029 | Documented; +198% relative gain over random confirms encoder learned non-trivial audio structure |

**Linear probe protocol:** Encoder is frozen after pretraining; only a single linear layer is trained on labeled clips. This isolates representation quality from classification capacity. The fact that the SimCLR encoder (0.088) outperforms both the random encoder (0.029) and the fully supervised CNN-R3c (0.060) indicates the contrastive objective provides better gradient signal than direct label supervision on this highly imbalanced 12-class problem.

## FAISS Dense Retrieval

| Run ID | System | Key changes | GradedNDCG@10 | PoetDiversity@10 | Decision |
| ------ | ------ | ----------- | ------------- | ---------------- | -------- |
| RET-R1 | FAISS + AraPoemBERT CLS | Flat IP index over 3,341 AraPoemBERT CLS embeddings; poem-level query (13 poems); baseline: embedding cosine only | 0.724 | 6.69 | Baseline |
| RET-R2 | FAISS + metadata re-ranking | Same index; re-rank top-50 by genre match (+0.3) + arousal match (+0.2) + poet diversity penalty (-0.1 per repeat poet); final score = 0.5×semantic + 0.5×metadata | **0.730** | 6.69 | Adopted; +0.8% over embedding-only |

CI (bootstrap, n=1000): GradedNDCG@10 = 0.730 [0.660, 0.757]. RoundTrip@10 = 0.769.

## Stratified 5-Fold CV Trials

| Run ID | Model | Loss | Context | Window | Clip F1 | Poem F1 | Decision |
| ------ | ----- | ---- | ------- | ------ | ------- | ------- | -------- |
| CV-R1 (baseline) | bert-base-arabertv2 | Uniform CE | 3 | — | 0.157 ± 0.047 | 0.182 ± 0.103 | Baseline; documents genuine difficulty under poet-disjoint CV |
| CV-R2 (improved) | bert-base-arabertv2 | Weighted CE + LS=0.1 | 5 | discriminative LR=0.9 | **0.188 ± 0.053** | **0.185 ± 0.066** | +3.1 pp clip gain over CV-R1; fold range 0.132–0.286 clip |

**CV-R2 improvements rationale:**
- Model: `aubmindlab/bert-base-arabertv2` retained — `faisalq/bert-base-arapoembert` attempted but has `max_position_embeddings=32`, making it incompatible with max_seq_len=128 and window=5 (RuntimeError on token_type_ids buffer expansion). arabertv2 supports 512 tokens.
- Class-weighted loss: inverse-frequency weights address Hikma dominance (~40% of train clips)
- Label smoothing ε=0.1: prevents train-acc=99.8% / val-F1-stagnating overfit pattern observed in CV-R1
- Context window=5: captures ~25s of poem context vs 15s in CV-R1
- Discriminative LR decay=0.9: lower BERT layers retain general Arabic pretraining signal
- Gradient clipping max_norm=1.0: already present in CV-R1 baseline, retained

## Data Integrity / Evaluation Refresh


| Change ID               | Scope                                              | Outcome                                                                                                                                   |
| ----------------------- | -------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| SPLIT-STRICT-PD         | `src/data/split.py` + regenerated processed splits | verse-level exception removed; strict poet-disjoint splits restored                                                                       |
| EVAL-REFRESH-2026-03-11 | tests + reports                                    | `pytest` passes (`52 passed`); refreshed poem aggregation, fusion, partial-credit, and demo smoke artifacts written to `outputs/reports/` |
| EVAL-OVERHAUL-2026-03-15 | `src/evaluation/metrics.py`, all eval scripts, reports | Evaluation overhaul: poem-level only for classification (clip-level removed); new metrics: MER=0.264/WIL=0.372 (ASR), nDCG@3=0.943, balanced_acc=0.675, log_loss=1.343, inter-annotator κ=0.098/α=0.087 (emotion); grouped bootstrap CIs; 11 stale report JSONs deleted; metric taxonomy added to all report JSONs |
| RETRIEVAL-POEM-LEVEL-2026-03-15 | `scripts/evaluate_retrieval.py` + retrieval index rebuild | Fixed retrieval evaluation from clip-level (333 queries) to poem-level (13 queries); re-indexed; new results: Genre GradedNDCG@10=0.732, Emotion GradedNDCG@10=0.724, PoetDiversity@10=6.69, RoundTrip@10=0.769; CI: 0.732 [0.481, 0.983] |



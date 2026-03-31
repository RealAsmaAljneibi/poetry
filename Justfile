# Nabat-AI Justfile  |  requires: uv (https://docs.astral.sh/uv/)

# ── Setup ─────────────────────────────────────────────────────────────────────

install:
    uv sync
    uv run python scripts/cache_models.py

cache-models:
    uv run python scripts/cache_models.py

# ── Data ──────────────────────────────────────────────────────────────────────

data:
    uv run python src/data/ingest.py
    uv run python src/data/split.py

# alias used in RTM runbook
generate-data:
    just data

eda:
    uv run python scripts/eda.py

tsne:
    uv run python scripts/tsne_map.py

# ── Training ──────────────────────────────────────────────────────────────────

# Train all main models (ASR → genre → emotion → arousal)
train:
    just asr
    uv run python scripts/train_text_classifier.py --model arapoem --task genre --context-window 3
    uv run python scripts/train_text_classifier.py --model arapoem --task emotion_text --context-window 1 --emotion-merge-profile rare_merge_v1 --run-id K1_merge_v1
    uv run python scripts/train_arousal.py

train-genre:
    uv run python scripts/train_text_classifier.py --model arapoem --task genre --context-window 3

train-emotion:
    uv run python scripts/train_text_classifier.py --model arapoem --task emotion_text --context-window 1 --emotion-merge-profile rare_merge_v1 --run-id K1_merge_v1

train-arousal:
    uv run python scripts/train_arousal.py

train-cnn:
    uv run python scripts/train_audio_cnn.py

train-fusion STRATEGY="gated" MODE="fusion":
    uv run python scripts/train_fusion.py --fusion-strategy {{STRATEGY}} --mode {{MODE}}

train-bilstm:
    uv run python scripts/train_emotion_bilstm.py

train-multitask:
    uv run python scripts/train_multitask.py --genre-weight 0.7 --emotion-weight 0.3

train-hierarchical:
    uv run python scripts/train_hierarchical.py --precompute --pos-features

pretrain-simclr:
    uv run python scripts/pretrain_audio_simclr.py

train-vae:
    uv run python scripts/train_vae_augment.py

# ── ASR fine-tuning runs (all resulted in catastrophic forgetting) ─────────────

# R1: corrected-only data
asr-r1:
    uv run python scripts/finetune_whisper.py --model openai/whisper-small --use-lora --lora-rank 16 --corrected-only --epochs 5 --batch-size 4 --grad-accum 8 --learning-rate 1e-5 --output-dir outputs/models/whisper_nabati_run1_corrected

# R2: balanced sampler (adopted for `just asr`)
asr:
    uv run python scripts/finetune_whisper.py --model openai/whisper-small --use-lora --lora-rank 16 --epochs 5 --batch-size 4 --grad-accum 8 --learning-rate 1e-5 --mix-ratio 0.5 --output-dir outputs/models/whisper_nabati_run2

# R3: speed perturbation
asr-r3:
    uv run python scripts/finetune_whisper.py --model openai/whisper-small --use-lora --lora-rank 4 --epochs 3 --batch-size 4 --grad-accum 8 --learning-rate 5e-6 --mix-ratio 0.5 --speed-perturb 0.9 1.0 1.1 --ocr-threshold 0.30 --stop-on-ocr --output-dir outputs/models/whisper_nabati_run3_speed

# R4: frozen encoder, decoder-only LoRA
asr-r4:
    uv run python scripts/finetune_whisper.py --model openai/whisper-small --use-lora --lora-rank 4 --epochs 5 --batch-size 4 --grad-accum 8 --learning-rate 1e-5 --mix-ratio 0.5 --freeze-encoder --decoder-only-lora --best-model-metric wer --output-dir outputs/models/whisper_nabati_run4_decoder

# R5a: SpecAugment
asr-r5-spec:
    uv run python scripts/finetune_whisper.py --model openai/whisper-small --use-lora --lora-rank 8 --epochs 5 --batch-size 4 --grad-accum 8 --learning-rate 5e-6 --mix-ratio 0.5 --spec-augment --spec-time-masks 2 --spec-time-max 80 --spec-freq-masks 2 --spec-freq-max 20 --best-model-metric wer --output-dir outputs/models/whisper_nabati_run5_spec

# R5b: SADA domain anchor (requires: just convert-sada /path/to/sada first)
asr-r5-sada-stage1:
    uv run python scripts/finetune_whisper.py --model openai/whisper-small --train-jsonl data/processed/sada_train.jsonl --val-jsonl data/processed/sada_val.jsonl --test-jsonl data/processed/sada_val.jsonl --use-lora --lora-rank 16 --epochs 3 --batch-size 4 --grad-accum 8 --learning-rate 1e-5 --mix-ratio 0.0 --spec-augment --best-model-metric wer --output-dir outputs/models/whisper_sada_stage1

asr-r5-sada-stage2:
    uv run python scripts/finetune_whisper.py --model outputs/models/whisper_sada_stage1/best --use-lora --lora-rank 4 --epochs 5 --batch-size 4 --grad-accum 8 --learning-rate 3e-6 --mix-ratio 0.5 --spec-augment --best-model-metric wer --output-dir outputs/models/whisper_nabati_run5_sada

# R6a: Adafactor optimizer
asr-r6-adafactor:
    uv run python scripts/finetune_whisper.py --model openai/whisper-small --use-lora --lora-rank 8 --epochs 5 --batch-size 4 --grad-accum 8 --learning-rate 1e-5 --mix-ratio 0.5 --optimizer adafactor --lr-scheduler-type linear --weight-decay 0.0 --best-model-metric wer --output-dir outputs/models/whisper_nabati_run6a_adafactor

# R6b: linear decay + label smoothing
asr-r6-smooth:
    uv run python scripts/finetune_whisper.py --model openai/whisper-small --use-lora --lora-rank 8 --lora-dropout 0.10 --epochs 5 --batch-size 4 --grad-accum 8 --learning-rate 5e-6 --mix-ratio 0.5 --label-smoothing 0.05 --lr-scheduler-type linear --best-model-metric wer --output-dir outputs/models/whisper_nabati_run6b_linear_smooth

# R6c: broader LoRA targets + cosine restarts
asr-r6-broad:
    uv run python scripts/finetune_whisper.py --model openai/whisper-small --use-lora --lora-rank 4 --lora-targets q_proj k_proj v_proj out_proj --epochs 5 --batch-size 4 --grad-accum 8 --learning-rate 3e-6 --mix-ratio 0.5 --lr-scheduler-type cosine_with_restarts --best-model-metric wer --output-dir outputs/models/whisper_nabati_run6c_broad_lora

# R7: mixed-domain replay (requires sada_train.jsonl)
asr-r7-replay:
    uv run python scripts/finetune_whisper.py --model openai/whisper-small --use-lora --lora-rank 4 --epochs 5 --batch-size 4 --grad-accum 8 --learning-rate 5e-6 --mix-ratio 0.5 --replay-jsonl data/processed/sada_train.jsonl --replay-ratio 0.25 --spec-augment --best-model-metric wer --output-dir outputs/models/whisper_nabati_run7_replay

convert-sada SADA_DIR="":
    #!/usr/bin/env bash
    if [ -z "{{SADA_DIR}}" ]; then echo "Usage: just convert-sada /path/to/sada"; exit 1; fi
    uv run python scripts/convert_sada_to_jsonl.py --sada-dir "{{SADA_DIR}}"

# ── CNN Ablation ───────────────────────────────────────────────────────────────

cnn-list:
    uv run python scripts/train_cnn_ablation.py --list

cnn-run RUN="CNN-R1":
    uv run python scripts/train_cnn_ablation.py --run {{RUN}}

cnn-all:
    uv run python scripts/train_cnn_ablation.py --run all

cnn-summary:
    uv run python scripts/train_cnn_ablation.py --summary

cnn-smoke:
    uv run python scripts/train_cnn_ablation.py --run all --epochs 3

# aliases used in RTM runbook
cnn-ablation-list:
    just cnn-list

cnn-ablation-run RUN="CNN-R1":
    just cnn-run {{RUN}}

cnn-ablation-all:
    just cnn-all

cnn-ablation-summary:
    just cnn-summary

cnn-ablation-smoke:
    just cnn-smoke

cnn-ablation-baseline:
    just cnn-run CNN-R1

# ── Fusion ablation ────────────────────────────────────────────────────────────

fusion-ablation:
    uv run python scripts/train_fusion.py --fusion-strategy gated --mode text_only
    uv run python scripts/train_fusion.py --fusion-strategy gated --mode audio_only
    uv run python scripts/train_fusion.py --fusion-strategy concat --mode fusion
    uv run python scripts/train_fusion.py --fusion-strategy gated --mode fusion
    uv run python scripts/train_fusion.py --fusion-strategy cross_attn --mode fusion

calibrate-genre:
    uv run python scripts/calibrate_genre.py

# ── Re-evaluation ──────────────────────────────────────────────────────────────

rerun-multitask:
    uv run python scripts/rerun_multitask_eval.py

rerun-hierarchical:
    uv run python scripts/rerun_hierarchical_eval.py

eval-simclr:
    uv run python scripts/evaluate_simclr_probe.py

genre-cv:
    uv run python scripts/stratified_genre_cv.py

techniques-table:
    uv run python scripts/comprehensive_ablation_table.py

# ── Evaluation ─────────────────────────────────────────────────────────────────

evaluate:
    uv run python scripts/evaluate_asr.py --split test --hypothesis text_whisper
    uv run python scripts/run_baseline.py
    uv run python scripts/build_retrieval_index.py
    uv run python scripts/evaluate_retrieval.py
    uv run python scripts/compute_dms.py
    uv run python scripts/evaluate_emotion_fusion.py
    uv run python scripts/evaluate_emotion_partial_credit.py
    uv run python scripts/compute_confidence_intervals.py

eval-baselines:
    uv run python scripts/run_baseline.py

# alias used in RTM runbook
evaluate-baselines:
    just eval-baselines

eval-retrieval:
    uv run python scripts/build_retrieval_index.py
    uv run python scripts/evaluate_retrieval.py

eval-retrieval-rerank:
    uv run python scripts/evaluate_retrieval_rerank.py

eval-retrieval-ablation:
    uv run python scripts/evaluate_retrieval_ablation.py

# ── Demo ───────────────────────────────────────────────────────────────────────

demo AUDIO="":
    #!/usr/bin/env bash
    if [ -z "{{AUDIO}}" ]; then
        CLIP=$(uv run python -c "import json; r=json.loads(open('data/processed/test.jsonl').readline()); print(r['audio_filename'])")
        echo "No audio specified — using first test clip: $CLIP"
        uv run python scripts/demo.py "$CLIP" --top-k 5
    else
        uv run python scripts/demo.py "{{AUDIO}}" --top-k 5
    fi

demo-smoke:
    uv run python scripts/demo_smoke_test.py

app:
    uv run python scripts/app.py

app-share:
    uv run python scripts/app.py --share

# ── Tests & quality ────────────────────────────────────────────────────────────

test:
    uv run pytest tests/ -v

lint:
    uv run ruff check src/ scripts/
    uv run ruff format src/ scripts/ --check

lint-fix:
    uv run ruff check src/ scripts/ --fix
    uv run ruff format src/ scripts/

# ── Clean ──────────────────────────────────────────────────────────────────────

clean:
    rm -rf .ruff_cache .pytest_cache outputs/reports outputs/figures outputs/retrieval outputs/runs outputs/demo_result.json
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

clean-all:
    rm -rf .ruff_cache .pytest_cache outputs/ logs/
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

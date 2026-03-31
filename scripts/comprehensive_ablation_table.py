#!/usr/bin/env python3
"""
Generate the definitive "Techniques Attempted" summary table
by reading all existing result JSON files.

This script scans outputs/reports/ for eval JSON files and extracts
key metrics to produce a comprehensive markdown table and JSON summary.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stderr),
        logging.FileHandler("logs/comprehensive_ablation.log", mode="a"),
    ],
)
logger = logging.getLogger(__name__)


def load_json_file(file_path: Path) -> Optional[Dict[str, Any]]:
    """Load a JSON file safely."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Failed to load {file_path}: {e}")
        return None


def extract_genre_baselines(baseline_csv: Path) -> Dict[str, Dict[str, float]]:
    """Extract genre baseline results from CSV."""
    results = {}
    try:
        with open(baseline_csv, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[1:]:  # Skip header
                parts = line.strip().split(",")
                if len(parts) >= 2:
                    model = parts[0]
                    macro_f1 = float(parts[1]) if parts[1] else 0.0
                    weighted_f1 = float(parts[2]) if parts[2] else 0.0
                    results[model] = {"macro_f1": macro_f1, "weighted_f1": weighted_f1}
    except Exception as e:
        logger.error(f"Error reading baseline CSV: {e}")
    return results


def compile_results(reports_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compile all results into categories.
    Returns a dict mapping category -> list of technique dicts.
    """
    techniques = {
        "genre_classification": [],
        "emotion_classification": [],
        "audio_classification": [],
        "asr": [],
        "other": [],
    }

    # Load baseline CSV
    baseline_csv = reports_dir / "baseline_results.csv"
    baselines = extract_genre_baselines(baseline_csv)
    logger.info(f"Loaded {len(baselines)} baseline entries")

    # GENRE CLASSIFICATION
    # 1. Majority-class baseline
    if "Majority-class | Genre" in baselines:
        techniques["genre_classification"].append({
            "technique": "Majority-class baseline",
            "dl_concept": "Frequency prior (zero-shot)",
            "clip_f1": None,
            "poem_f1": baselines["Majority-class | Genre"]["macro_f1"],
            "status": "baseline",
            "notes": "test macro F1",
        })

    # 2. TF-IDF + LogReg baseline
    if "TF-IDF + LogReg | Genre" in baselines:
        techniques["genre_classification"].append({
            "technique": "TF-IDF + LogReg",
            "dl_concept": "Bag-of-words baseline",
            "clip_f1": None,
            "poem_f1": baselines["TF-IDF + LogReg | Genre"]["macro_f1"],
            "status": "baseline",
            "notes": "test macro F1",
        })

    # 3. mBERT genre
    if "mBERT fine-tuned | Genre" in baselines:
        techniques["genre_classification"].append({
            "technique": "mBERT fine-tuned genre",
            "dl_concept": "Transformer (frozen pooler)",
            "clip_f1": None,
            "poem_f1": baselines["mBERT fine-tuned | Genre"]["macro_f1"],
            "status": "baseline",
            "notes": "test macro F1",
        })

    # 4. AraPoemBERT genre (GENRE-R4, window=3) — adopted
    arapoem_genre_file = reports_dir / "arapoem_genre_report.txt"
    if arapoem_genre_file.exists():
        try:
            with open(arapoem_genre_file, "r", encoding="utf-8") as f:
                content = f.read()
                # Extract weighted avg f1-score line (last row)
                lines = content.strip().split("\n")
                for line in reversed(lines):
                    if "weighted avg" in line:
                        parts = line.split()
                        if len(parts) >= 4:
                            weighted_f1 = float(parts[-2])
                            techniques["genre_classification"].append({
                                "technique": "AraPoemBERT genre (GENRE-R4, window=3)",
                                "dl_concept": "Transformer fine-tuning + context window",
                                "clip_f1": None,
                                "poem_f1": weighted_f1,
                                "status": "adopted",
                                "notes": "test weighted F1; on 6 genre classes",
                            })
                            break
        except Exception as e:
            logger.warning(f"Could not parse arapoem_genre_report.txt: {e}")

    # 5. Multitask (genre 0.7 + emotion 0.3)
    multitask_file = reports_dir / "multitask_eval.json"
    if multitask_file.exists():
        data = load_json_file(multitask_file)
        if data:
            techniques["genre_classification"].append({
                "technique": "Multitask (genre 0.7 + emotion 0.3)",
                "dl_concept": "Shared encoder, task-specific heads",
                "clip_f1": data.get("test_genre_clip_f1"),
                "poem_f1": data.get("test_genre_poem_f1"),
                "status": "negative",
                "notes": "lower than adopted baseline",
            })

    # 6. Hierarchical BiLSTM (from hier_pos_genre_report.txt)
    hier_genre_file = reports_dir / "hier_pos_genre_report.txt"
    if hier_genre_file.exists():
        try:
            with open(hier_genre_file, "r", encoding="utf-8") as f:
                content = f.read()
                lines = content.strip().split("\n")
                for line in reversed(lines):
                    if "weighted avg" in line:
                        parts = line.split()
                        if len(parts) >= 4:
                            weighted_f1 = float(parts[-2])
                            techniques["genre_classification"].append({
                                "technique": "Hierarchical BiLSTM",
                                "dl_concept": "RNN with poetic structure",
                                "clip_f1": None,
                                "poem_f1": weighted_f1,
                                "status": "documented",
                                "notes": "test weighted F1; 8 genre classes",
                            })
                            break
        except Exception as e:
            logger.warning(f"Could not parse hier_pos_genre_report.txt: {e}")

    # EMOTION CLASSIFICATION
    # 1. Majority-class baseline
    if "Majority-class | Emotion (text)" in baselines:
        techniques["emotion_classification"].append({
            "technique": "Majority-class baseline",
            "dl_concept": "Frequency prior (zero-shot)",
            "clip_f1": None,
            "poem_f1": baselines["Majority-class | Emotion (text)"]["macro_f1"],
            "status": "baseline",
            "notes": "test macro F1",
        })

    # 2. TF-IDF + LogReg emotion
    if "TF-IDF + LogReg | Emotion (text)" in baselines:
        techniques["emotion_classification"].append({
            "technique": "TF-IDF + LogReg emotion",
            "dl_concept": "Bag-of-words + linear classifier",
            "clip_f1": None,
            "poem_f1": baselines["TF-IDF + LogReg | Emotion (text)"]["macro_f1"],
            "status": "baseline",
            "notes": "test macro F1",
        })

    # 3. MFCC + SVM emotion
    if "MFCC + SVM | Emotion (audio)" in baselines:
        techniques["emotion_classification"].append({
            "technique": "MFCC + SVM emotion",
            "dl_concept": "Hand-crafted features + SVM",
            "clip_f1": None,
            "poem_f1": baselines["MFCC + SVM | Emotion (audio)"]["macro_f1"],
            "status": "baseline",
            "notes": "test macro F1 (audio-based)",
        })

    # 4. AraPoemBERT emotion (K1_merge_v1) — clip level
    emotion_exp_file = reports_dir / "emotion_experiments_summary.json"
    if emotion_exp_file.exists():
        data = load_json_file(emotion_exp_file)
        if data and "experiments" in data:
            for exp in data["experiments"]:
                if exp.get("run_id") == "K1_merge_v1":
                    techniques["emotion_classification"].append({
                        "technique": "AraPoemBERT emotion (K1_merge_v1)",
                        "dl_concept": "Transformer + class merging",
                        "clip_f1": exp.get("test_hard_macro_f1_clip"),
                        "poem_f1": exp.get("test_hard_macro_f1_poem"),
                        "status": "adopted",
                        "notes": "clip-level; 9 emotion classes",
                    })
                    break

    # 5. Emotion BiLSTM
    emotion_bilstm_file = reports_dir / "emotion_bilstm_eval.json"
    if emotion_bilstm_file.exists():
        data = load_json_file(emotion_bilstm_file)
        if data:
            techniques["emotion_classification"].append({
                "technique": "Emotion BiLSTM",
                "dl_concept": "RNN sequence modeling",
                "clip_f1": data.get("test_clip_macro_f1"),
                "poem_f1": data.get("test_poem_macro_f1"),
                "status": "negative",
                "notes": "underperformed fusion",
            })

    # 6. Poem-level fusion (FUS-full) — adopted
    emotion_fusion_file = reports_dir / "emotion_fusion_eval.json"
    if emotion_fusion_file.exists():
        data = load_json_file(emotion_fusion_file)
        if data and "test" in data:
            test_data = data["test"]
            if "systems" in test_data and "full_fusion" in test_data["systems"]:
                fusion_metrics = test_data["systems"]["full_fusion"]["poem_metrics"]
                techniques["emotion_classification"].append({
                    "technique": "Poem-level fusion (FUS-full)",
                    "dl_concept": "Multi-modal ensemble with text+audio logit fusion",
                    "clip_f1": None,
                    "poem_f1": fusion_metrics.get("hard_macro_f1"),
                    "status": "adopted",
                    "notes": "Best: λ=1.0, logit_mean aggregation",
                })

    # AUDIO (FROM SCRATCH)
    cnn_exp_file = reports_dir / "cnn_experiments_summary.json"
    if cnn_exp_file.exists():
        data = load_json_file(cnn_exp_file)
        if isinstance(data, list):
            # Add random baseline
            techniques["audio_classification"].append({
                "technique": "Random baseline",
                "dl_concept": "Uniform random guess (12-class)",
                "clip_f1": 1.0 / 12,  # 0.0833
                "poem_f1": None,
                "status": "baseline",
                "notes": "theoretical lower bound",
            })

            # Find CNN-R3c (best ablation)
            for run in data:
                if run.get("run_id") == "CNN-R3c":
                    techniques["audio_classification"].append({
                        "technique": "CNN-R3c (best ablation)",
                        "dl_concept": "1D Conv on mel-spectrograms, vanilla CE loss",
                        "clip_f1": run.get("test_f1"),
                        "poem_f1": None,
                        "status": "adopted",
                        "notes": "Conv1d 2-block, 888K params",
                    })
                    break

    # ASR
    # 1. Zero-shot Whisper-small (adopted)
    asr_file = reports_dir / "asr_eval_test_text_whisper.json"
    if asr_file.exists():
        data = load_json_file(asr_file)
        if isinstance(data, list) and len(data) > 0:
            asr_data = data[0]
            techniques["asr"].append({
                "technique": "Whisper-small zero-shot",
                "dl_concept": "Pre-trained ASR (no fine-tuning)",
                "clip_f1": None,
                "poem_f1": None,
                "status": "adopted",
                "notes": f"WER=0.245, CER=0.081, soft_CER=0.075 (baseline)",
            })

    # 2. Fine-tuning runs (all negative)
    asr_lora_file = reports_dir / "trained_models_eval.json"
    if asr_lora_file.exists():
        data = load_json_file(asr_lora_file)
        if isinstance(data, list) and len(data) > 0:
            asr_lora = data[0]
            if "LoRA" in asr_lora.get("model", ""):
                techniques["asr"].append({
                    "technique": "Whisper-small LoRA (7 fine-tuning runs)",
                    "dl_concept": "LoRA adapter on pre-trained Whisper",
                    "clip_f1": None,
                    "poem_f1": None,
                    "status": "negative",
                    "notes": f"All runs worse: WER=0.589, CER=0.224, soft_CER=0.209",
                })

    # OTHER
    # 1. Arousal MLP
    arousal_file = reports_dir / "arousal_eval.json"
    if arousal_file.exists():
        data = load_json_file(arousal_file)
        if data:
            techniques["other"].append({
                "technique": "Arousal MLP (from scratch)",
                "dl_concept": "Learned feature MLP on hand-engineered features",
                "clip_f1": data.get("test_macro_f1"),
                "poem_f1": None,
                "status": "documented",
                "notes": f"3-class arousal; {data.get('params')} params",
            })

    # 2. Retrieval (GradedNDCG@10)
    retrieval_file = reports_dir / "retrieval_eval.json"
    if retrieval_file.exists():
        data = load_json_file(retrieval_file)
        if data and "Genre Retrieval" in data:
            graded_ndcg = data["Genre Retrieval"].get("GradedNDCG@10")
            techniques["other"].append({
                "technique": "Retrieval (FAISS hybrid text+audio)",
                "dl_concept": "Dense embeddings + semantic search",
                "clip_f1": None,
                "poem_f1": graded_ndcg,
                "status": "documented",
                "notes": "GradedNDCG@10 for genre query task",
            })

    logger.info(f"Compiled {sum(len(v) for v in techniques.values())} total techniques")
    return techniques


def generate_markdown_table(techniques: Dict[str, List[Dict[str, Any]]]) -> str:
    """Generate comprehensive markdown table."""
    md = "# Comprehensive Techniques Attempted\n\n"
    md += "**Last updated:** 2026-03-28\n\n"

    categories = {
        "genre_classification": "Genre Classification",
        "emotion_classification": "Emotion Classification",
        "audio_classification": "Audio Classification (From Scratch)",
        "asr": "Automatic Speech Recognition (ASR)",
        "other": "Other Tasks",
    }

    for key, title in categories.items():
        if not techniques[key]:
            continue

        md += f"## {title}\n\n"
        md += "| Technique | DL Concept | Clip F1 | Poem F1 | Status | Notes |\n"
        md += "|-----------|-----------|---------|---------|--------|-------|\n"

        for tech in techniques[key]:
            clip_f1_str = (
                f"{tech['clip_f1']:.4f}" if tech["clip_f1"] is not None else "—"
            )
            poem_f1_str = (
                f"{tech['poem_f1']:.4f}" if tech["poem_f1"] is not None else "—"
            )
            status = tech["status"].upper()
            notes = tech["notes"]

            md += f"| {tech['technique']} | {tech['dl_concept']} | {clip_f1_str} | {poem_f1_str} | {status} | {notes} |\n"

        md += "\n"

    return md


def generate_json_summary(
    techniques: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    """Generate JSON summary."""
    summary = {
        "generated_at": "2026-03-28",
        "project": "Nabat-AI: Multimodal Khaleeji Poetry Analysis",
        "purpose": "Definitive techniques attempted and their outcomes",
        "metric_tiers": {
            "primary": ["poem_macro_f1", "ndcg_at_3", "top3_accuracy"],
            "secondary": ["log_loss", "balanced_accuracy"],
            "constraint": ["partial_credit", "genre_plausibility_rate", "dms_rate", "ece"],
        },
        "techniques_by_category": {},
    }

    category_labels = {
        "genre_classification": "Genre Classification",
        "emotion_classification": "Emotion Classification",
        "audio_classification": "Audio Classification",
        "asr": "Automatic Speech Recognition",
        "other": "Other Tasks",
    }

    for key, label in category_labels.items():
        if techniques[key]:
            summary["techniques_by_category"][label] = techniques[key]

    summary["summary_stats"] = {
        "total_techniques": sum(len(v) for v in techniques.values()),
        "total_adopted": sum(
            1
            for v in techniques.values()
            for t in v
            if t["status"] == "adopted"
        ),
        "total_negative": sum(
            1
            for v in techniques.values()
            for t in v
            if t["status"] == "negative"
        ),
        "total_baseline": sum(
            1
            for v in techniques.values()
            for t in v
            if t["status"] == "baseline"
        ),
        "total_documented": sum(
            1
            for v in techniques.values()
            for t in v
            if t["status"] == "documented"
        ),
    }

    return summary


def main() -> None:
    """Main entry point."""
    reports_dir = Path(__file__).parent.parent / "outputs" / "reports"
    output_dir = reports_dir

    if not reports_dir.exists():
        logger.error(f"Reports directory not found: {reports_dir}")
        sys.exit(1)

    logger.info(f"Scanning {reports_dir} for result files...")

    # Compile all results
    techniques = compile_results(reports_dir)

    # Generate markdown table
    md_table = generate_markdown_table(techniques)
    md_output = output_dir / "comprehensive_techniques_table.md"
    with open(md_output, "w", encoding="utf-8") as f:
        f.write(md_table)
    logger.info(f"Wrote markdown table: {md_output}")

    # Generate JSON summary
    json_summary = generate_json_summary(techniques)
    json_output = output_dir / "comprehensive_techniques_table.json"
    with open(json_output, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote JSON summary: {json_output}")

    # Print summary to console
    logger.info(f"=== SUMMARY ===")
    logger.info(f"Total techniques: {json_summary['summary_stats']['total_techniques']}")
    logger.info(f"Adopted: {json_summary['summary_stats']['total_adopted']}")
    logger.info(f"Negative: {json_summary['summary_stats']['total_negative']}")
    logger.info(f"Baseline: {json_summary['summary_stats']['total_baseline']}")
    logger.info(f"Documented: {json_summary['summary_stats']['total_documented']}")


if __name__ == "__main__":
    main()

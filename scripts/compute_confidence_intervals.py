"""
scripts/compute_confidence_intervals.py

Bootstrap 95% confidence intervals for key system metrics.

Metrics bootstrapped:
  1. Genre Macro-F1      (AraPoemBERT + window=3, clip-level)
  2. Genre Poem-F1       (majority-vote aggregation)
  3. Arousal Macro-F1    (ArousalMLP, from-scratch)
  4. Emotion Partial-Credit (AraPoemBERT emotion_text classifier)
  5. Retrieval GradedNDCG@10 (per-query bootstrap)

Method: 1000 bootstrap resamples with replacement.
        95% CI = [2.5th percentile, 97.5th percentile].

Output: outputs/reports/confidence_intervals.json
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import f1_score
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.labels import GENRE_CLASSES, EMOTION_CLASSES, ID2EMOTION, get_merged_emotion_classes
from src.evaluation.metrics import emotion_partial_credit, bootstrap_grouped_ci, balanced_accuracy as calc_balanced_accuracy, emotion_ndcg_at_3, top_k_accuracy

PROJECT_ROOT = Path(__file__).parent.parent
N_BOOTSTRAP  = 1000
RANDOM_SEED  = 42
rng          = np.random.default_rng(RANDOM_SEED)


# ─── Bootstrap helper ─────────────────────────────────────────────────────────

def bootstrap_ci(
    values: np.ndarray,
    stat_fn,
    n: int = N_BOOTSTRAP,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """
    Compute mean ± 95% CI via bootstrap resampling.
    Returns (mean, lower, upper).
    """
    n_samples = len(values)
    stats = np.empty(n)
    for i in range(n):
        idx    = rng.integers(0, n_samples, size=n_samples)
        stats[i] = stat_fn(values[idx])
    return float(np.mean(stats)), float(np.percentile(stats, 100 * alpha / 2)), float(np.percentile(stats, 100 * (1 - alpha / 2)))


def macro_f1_from_pairs(pairs: np.ndarray) -> float:
    """pairs is (N,2) array of [y_true, y_pred]."""
    return f1_score(pairs[:, 0], pairs[:, 1], average="macro", zero_division=0)


# ─── Genre inference ──────────────────────────────────────────────────────────

def run_genre_inference() -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Returns (y_true, y_pred, poem_ids) on test set."""
    from scripts.train_text_classifier import NabatiTextDataset

    MODEL_NAME  = "faisalq/bert-base-arapoembert"
    CKPT        = PROJECT_ROOT / "outputs/models/arapoem_genre/arapoem_genre_best.pt"
    N_CLASSES   = len(GENRE_CLASSES)
    MAX_SEQ_LEN = 32
    CONTEXT_WIN = 3
    BATCH_SIZE  = 64
    DEVICE      = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    logger.info("Loading genre model ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=N_CLASSES, ignore_mismatched_sizes=True,
        local_files_only=True,
    ).to(DEVICE)
    state = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval()

    test_ds = NabatiTextDataset(
        PROJECT_ROOT / "data/processed/test.jsonl",
        tokenizer, "genre", MAX_SEQ_LEN, CONTEXT_WIN,
    )
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    all_preds, all_true, all_poems = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
            ).logits
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_true.extend(batch["label"].tolist())
            all_poems.extend(batch["poem_id"])

    logger.info(f"Genre inference done: {len(all_true)} clips")
    return np.array(all_true), np.array(all_preds), all_poems


# ─── Arousal inference ────────────────────────────────────────────────────────

def run_arousal_inference() -> tuple[np.ndarray, np.ndarray]:
    """Returns (y_true, y_pred) on test set using cached features."""
    import torch.nn as nn

    class ArousalMLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, n_layers, dropout, n_classes=3):
            super().__init__()
            layers, in_dim = [], input_dim
            for _ in range(n_layers):
                layers += [nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim),
                           nn.ReLU(), nn.Dropout(dropout)]
                in_dim = hidden_dim
            layers.append(nn.Linear(hidden_dim, n_classes))
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x)

    from src.data.arousal_labels import encode_arousal

    CKPT   = PROJECT_ROOT / "outputs/models/arousal_mlp/arousal_mlp_arousal_best.pt"
    SCALER = PROJECT_ROOT / "outputs/models/arousal_mlp/arousal_scaler.pkl"
    DEVICE = torch.device("cpu")

    logger.info("Loading arousal model ...")
    state = torch.load(CKPT, map_location=DEVICE, weights_only=False)

    # Architecture from arousal_eval.json config (input_dim=34, hidden=128, n_layers=2)
    model = ArousalMLP(input_dim=34, hidden_dim=128, n_layers=2, dropout=0.0, n_classes=3).to(DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()

    with open(SCALER, "rb") as f:
        scaler = pickle.load(f)

    # Load test features
    with open(PROJECT_ROOT / "data/processed/arousal_features.pkl", "rb") as f:
        feature_cache = pickle.load(f)

    X_rows, y_rows = [], []
    for line in open(PROJECT_ROOT / "data/processed/test.jsonl"):
        row   = json.loads(line)
        path  = row.get("audio_filename", "")
        label = encode_arousal(row.get("emotion_audio"))
        feats = feature_cache.get(path)
        if feats is None or label == -1:
            continue
        X_rows.append(feats)
        y_rows.append(label)

    X = scaler.transform(np.array(X_rows, dtype=np.float32))
    X_t = torch.tensor(X)
    with torch.no_grad():
        preds = model(X_t).argmax(dim=-1).numpy()

    logger.info(f"Arousal inference done: {len(y_rows)} clips")
    return np.array(y_rows), preds


# ─── Emotion partial-credit inference ─────────────────────────────────────────

def run_emotion_pc_inference() -> np.ndarray:
    """Returns per-clip partial-credit scores on test set."""
    from scripts.train_text_classifier import NabatiTextDataset

    MODEL_NAME  = "faisalq/bert-base-arapoembert"
    CKPT        = PROJECT_ROOT / "outputs/models/arapoem_emotion/arapoem_emotion_text_best.pt"
    MERGE_PROFILE = "rare_merge_v1"      # K1_merge_v1 — adopted emotion model
    N_CLASSES   = len(get_merged_emotion_classes(MERGE_PROFILE))  # 9 classes
    MAX_SEQ_LEN = 32
    CONTEXT_WIN = 1                      # K1_merge_v1 uses window=1
    BATCH_SIZE  = 64
    DEVICE      = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    logger.info("Loading emotion model ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=N_CLASSES, ignore_mismatched_sizes=True,
        local_files_only=True,
    ).to(DEVICE)
    state = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval()

    test_ds = NabatiTextDataset(
        PROJECT_ROOT / "data/processed/test.jsonl",
        tokenizer, "emotion_text", MAX_SEQ_LEN, CONTEXT_WIN,
    )
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # Build raw row lookup for emotion_audio + genre
    raw_rows: dict[str, dict] = {}
    for line in open(PROJECT_ROOT / "data/processed/test.jsonl"):
        r = json.loads(line)
        key = r["source_poem"] + "|" + r.get("poet_en", "")
        raw_rows[key] = r

    all_preds, all_true, all_poem_ids = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(
                input_ids=batch["input_ids"].to(DEVICE),
                attention_mask=batch["attention_mask"].to(DEVICE),
            ).logits
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_true.extend(batch["label"].tolist())
            all_poem_ids.extend(batch["poem_id"])

    scores = []
    for pred_id, true_id, poem_id in zip(all_preds, all_true, all_poem_ids):
        pred_label = ID2EMOTION.get(pred_id, "")
        true_label = ID2EMOTION.get(true_id, "")
        row        = raw_rows.get(poem_id, {})
        score      = emotion_partial_credit(
            pred_label,
            row.get("emotion_audio", "") or "",
            true_label,
            row.get("genre_en", "") or "",
        )
        scores.append(score)

    logger.info(f"Emotion PC inference done: {len(scores)} clips")
    return np.array(scores)


# ─── Retrieval per-query bootstrap ────────────────────────────────────────────

def load_retrieval_query_scores() -> np.ndarray:
    """
    Compute per-query GradedNDCG@10 by re-running evaluate_retrieval or from stored data.
    Since the report only stores aggregates, rebuild from the retrieval index.
    Falls back to parametric CI from stored aggregate if index is stale.
    """
    # Check if detailed per-query file exists
    detail_path = PROJECT_ROOT / "outputs/reports/retrieval_per_query.json"
    if detail_path.exists():
        data = json.loads(detail_path.read_text())
        return np.array(data["graded_ndcg_10_per_query"])

    # Fall back: load aggregate and return None (will use Gaussian approximation)
    return None


# ─── Poem-level genre CI ──────────────────────────────────────────────────────

def poem_level_f1_ci(y_true: np.ndarray, y_pred: np.ndarray, poem_ids: list[str]) -> tuple[float, float, float]:
    """Bootstrap poem-level Macro-F1 (majority vote per poem)."""
    poems: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    for yt, yp, pid in zip(y_true, y_pred, poem_ids):
        poems[pid][0].append(yt)
        poems[pid][1].append(yp)

    poem_true, poem_pred = [], []
    for pid, (trues, preds) in poems.items():
        # majority vote
        poem_true.append(int(np.bincount(trues).argmax()))
        poem_pred.append(int(np.bincount(preds).argmax()))

    pairs = np.column_stack([poem_true, poem_pred])

    def stat(idx_pairs):
        return f1_score(idx_pairs[:, 0], idx_pairs[:, 1], average="macro", zero_division=0)

    n_poems = len(pairs)
    stats = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx    = rng.integers(0, n_poems, size=n_poems)
        stats[i] = stat(pairs[idx])

    observed = float(f1_score(poem_true, poem_pred, average="macro", zero_division=0))
    return observed, float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def poem_level_balanced_accuracy_ci(
    y_true: np.ndarray, y_pred: np.ndarray, poem_ids: list[str],
) -> tuple[float, float, float]:
    """Bootstrap poem-level balanced accuracy (majority vote per poem)."""
    from sklearn.metrics import balanced_accuracy_score
    poems: dict[str, tuple[list[int], list[int]]] = defaultdict(lambda: ([], []))
    for yt, yp, pid in zip(y_true, y_pred, poem_ids):
        poems[pid][0].append(yt)
        poems[pid][1].append(yp)

    poem_true, poem_pred = [], []
    for pid, (trues, preds) in poems.items():
        poem_true.append(int(np.bincount(trues).argmax()))
        poem_pred.append(int(np.bincount(preds).argmax()))

    pairs = np.column_stack([poem_true, poem_pred])

    def stat(idx_pairs):
        return balanced_accuracy_score(idx_pairs[:, 0], idx_pairs[:, 1])

    n_poems = len(pairs)
    stats = np.empty(N_BOOTSTRAP)
    for i in range(N_BOOTSTRAP):
        idx = rng.integers(0, n_poems, size=n_poems)
        stats[i] = stat(pairs[idx])

    observed = float(balanced_accuracy_score(poem_true, poem_pred))
    return observed, float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    results: dict = {}

    # 1. Genre (poem-level only — clip-level not reported)
    logger.info("=" * 60)
    logger.info("1. Genre Macro-F1 (poem-level)")
    y_true_g, y_pred_g, poem_ids_g = run_genre_inference()

    logger.info("1a. Genre Macro-F1 (poem-level)")
    poem_f1, pg_lo, pg_hi = poem_level_f1_ci(y_true_g, y_pred_g, poem_ids_g)
    logger.info(f"  Poem-level F1 = {poem_f1:.4f}  95% CI: [{pg_lo:.4f}, {pg_hi:.4f}]")
    results["genre_poem_f1"] = {"value": round(poem_f1, 4), "ci_95": [round(pg_lo, 4), round(pg_hi, 4)]}

    logger.info("1c. Genre Balanced Accuracy (poem-level)")
    ba_obs, ba_lo, ba_hi = poem_level_balanced_accuracy_ci(y_true_g, y_pred_g, poem_ids_g)
    logger.info(f"  Poem-level Balanced Acc = {ba_obs:.4f}  95% CI: [{ba_lo:.4f}, {ba_hi:.4f}]")
    results["genre_balanced_accuracy"] = {"value": round(ba_obs, 4), "ci_95": [round(ba_lo, 4), round(ba_hi, 4)]}

    # 2. Arousal Macro-F1
    logger.info("=" * 60)
    logger.info("2. Arousal Macro-F1")
    y_true_a, y_pred_a = run_arousal_inference()
    arousal_f1 = f1_score(y_true_a, y_pred_a, average="macro", zero_division=0)
    pairs_a = np.column_stack([y_true_a, y_pred_a])
    a_mean, a_lo, a_hi = bootstrap_ci(pairs_a, macro_f1_from_pairs)
    logger.info(f"  Arousal F1 = {arousal_f1:.4f}  95% CI: [{a_lo:.4f}, {a_hi:.4f}]")
    results["arousal_f1"] = {"value": round(arousal_f1, 4), "ci_95": [round(a_lo, 4), round(a_hi, 4)]}

    # 3. Emotion — poem-level grouped bootstrap (no clip-level)
    logger.info("=" * 60)
    logger.info("3. Emotion (poem-level grouped bootstrap)")

    # Load poem-level predictions for grouped bootstrap
    poem_preds_path = PROJECT_ROOT / "outputs/reports/poem_emotion_predictions_test.json"
    fusion_report_path = PROJECT_ROOT / "outputs/reports/emotion_fusion_eval.json"
    if poem_preds_path.exists() and fusion_report_path.exists():
        fusion_data = json.loads(fusion_report_path.read_text())
        adopted = fusion_data.get("adopted_variant", "full_fusion")
        poem_preds = json.loads(poem_preds_path.read_text())
        variant_preds = poem_preds.get(adopted, {})
        merge_profile = fusion_data.get("merge_profile", "rare_merge_v1")
        try:
            labels = get_merged_emotion_classes(merge_profile)
        except Exception:
            labels = []

        if variant_preds and labels:
            # Per-poem partial credit
            per_poem_pc: dict[str, float] = {}
            per_poem_ndcg: dict[str, float] = {}
            per_poem_top3: dict[str, float] = {}
            for poem_id, payload in variant_preds.items():
                gold = payload.get("gold_poem_emotion", "")
                pred = payload.get("emotion_poem_final") or payload.get("predicted_poem_emotion", "")
                genre = payload.get("manual_genre", "")
                audio_aux = payload.get("audio_emotion_poem_aux") or ""
                per_poem_pc[poem_id] = emotion_partial_credit(pred, audio_aux, gold, genre)

                probs = payload.get("poem_probabilities")
                if probs and labels:
                    prob_vec = [float(probs.get(label, 0.0)) for label in labels]
                    per_poem_ndcg[poem_id] = emotion_ndcg_at_3(prob_vec, gold, audio_aux, genre, labels)
                    true_id = labels.index(gold) if gold in labels else -1
                    per_poem_top3[poem_id] = top_k_accuracy(prob_vec, true_id, k=3) if true_id >= 0 else 0.0

            # Grouped bootstrap for poem-level PC
            obs_pc, lo_pc, hi_pc = bootstrap_grouped_ci(per_poem_pc, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED)
            logger.info(f"  Poem PC = {obs_pc:.4f}  95% CI: [{lo_pc:.4f}, {hi_pc:.4f}]")
            results["emotion_poem_partial_credit"] = {"value": round(obs_pc, 4), "ci_95": [round(lo_pc, 4), round(hi_pc, 4)]}

            if per_poem_ndcg:
                obs_ndcg, lo_ndcg, hi_ndcg = bootstrap_grouped_ci(per_poem_ndcg, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED)
                logger.info(f"  Poem nDCG@3 = {obs_ndcg:.4f}  95% CI: [{lo_ndcg:.4f}, {hi_ndcg:.4f}]")
                results["emotion_ndcg_at_3"] = {"value": round(obs_ndcg, 4), "ci_95": [round(lo_ndcg, 4), round(hi_ndcg, 4)]}

            if per_poem_top3:
                obs_t3, lo_t3, hi_t3 = bootstrap_grouped_ci(per_poem_top3, n_bootstrap=N_BOOTSTRAP, seed=RANDOM_SEED)
                logger.info(f"  Poem Recall@3 = {obs_t3:.4f}  95% CI: [{lo_t3:.4f}, {hi_t3:.4f}]")
                results["emotion_recall_at_3"] = {"value": round(obs_t3, 4), "ci_95": [round(lo_t3, 4), round(hi_t3, 4)]}
    else:
        logger.warning("Poem-level predictions not found; running clip-level inference as fallback")
        pc_scores = run_emotion_pc_inference()
        pc_mean = float(pc_scores.mean())
        em_mean, em_lo, em_hi = bootstrap_ci(pc_scores, np.mean)
        results["emotion_partial_credit_fallback"] = {"value": round(pc_mean, 4), "ci_95": [round(em_lo, 4), round(em_hi, 4)]}

    # 4. Retrieval GradedNDCG@10
    logger.info("=" * 60)
    logger.info("4. Retrieval GradedNDCG@10")
    retrieval_report = json.loads(
        (PROJECT_ROOT / "outputs/reports/retrieval_eval.json").read_text()
    )
    graded_ndcg = retrieval_report["Genre Retrieval"]["GradedNDCG@10"]
    n_queries   = retrieval_report["Genre Retrieval"]["n_queries"]

    p      = graded_ndcg
    se     = np.sqrt(p * (1 - p) / n_queries)
    r_lo   = max(0.0, p - 1.96 * se)
    r_hi   = min(1.0, p + 1.96 * se)
    logger.info(f"  GradedNDCG@10 = {graded_ndcg:.4f}  95% CI (CLT): [{r_lo:.4f}, {r_hi:.4f}]  (n={n_queries})")
    results["retrieval_graded_ndcg_10"] = {
        "value": round(graded_ndcg, 4),
        "ci_95": [round(r_lo, 4), round(r_hi, 4)],
        "method": "CLT Gaussian (per-query scores not stored)",
        "n_queries": n_queries,
    }

    # 5. Summary table
    logger.info("=" * 60)
    logger.info("SUMMARY — Key Metrics with 95% Bootstrap CI")
    logger.info("=" * 60)
    logger.info("  NOTE: N=13 test poems; CIs are honest but necessarily wide")
    display_rows = [
        ("Genre Macro-F1 (poem)",        results.get("genre_poem_f1")),
        ("Genre Balanced-Acc (poem)",     results.get("genre_balanced_accuracy")),
        ("Arousal Macro-F1",             results.get("arousal_f1")),
        ("Emotion PC (poem)",            results.get("emotion_poem_partial_credit")),
        ("Emotion nDCG@3 (poem)",        results.get("emotion_ndcg_at_3")),
        ("Emotion Recall@3 (poem)",      results.get("emotion_recall_at_3")),
        ("Retrieval GradedNDCG@10",      results.get("retrieval_graded_ndcg_10")),
    ]
    for name, r in display_rows:
        if r is None:
            continue
        lo, hi = r["ci_95"]
        logger.info(f"  {name:30s}: {r['value']:.3f}  [{lo:.3f}, {hi:.3f}]")

    results["note"] = "N=13 test poems; CIs are honest but necessarily wide"

    out_path = PROJECT_ROOT / "outputs/reports/confidence_intervals.json"
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    logger.info(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

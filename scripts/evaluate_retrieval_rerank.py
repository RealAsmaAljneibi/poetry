"""
scripts/evaluate_retrieval_rerank.py

Metadata re-ranking for NabatiRetriever.

Two extra signals on top of AraPoemBERT embedding cosine similarity:
  imagery_overlap  — Jaccard overlap of imagery_tags_en between query and candidate
  value_overlap    — exact match of khaleeji_value (binary 0/1)

Tuning: grid-search β,γ on VAL queries to maximise GradedNDCG@10.
        α is fixed at 1.0 (embedding score is always the primary signal).
        Score = α * embed_score + β * imagery_overlap + γ * value_overlap

Evaluation: best β,γ applied to TEST queries.
            Compare GradedNDCG@10, PoetDiversity@10, ImageryCoherence@10 vs baseline.

Output:
    outputs/reports/retrieval_eval_rerank.json
"""
from __future__ import annotations

import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.retrieval import NabatiRetriever
from src.evaluation.metrics import graded_ndcg_at_k, imagery_coherence_at_k

DATA_DIR      = Path("data/processed")
RETRIEVAL_DIR = Path("outputs/retrieval")
REPORT_DIR    = Path("outputs/reports")

VAL_SPLIT  = DATA_DIR / "val.jsonl"
TEST_SPLIT = DATA_DIR / "test.jsonl"
TRAIN_SPLIT = DATA_DIR / "train.jsonl"

FETCH_K    = 50    # initial candidates before re-ranking
EVAL_K     = 10    # final K for metric


# ── Record helpers ─────────────────────────────────────────────────────────────

def record_id(rec: dict) -> str:
    af = rec.get("audio_filename", "").strip()
    return af if af else rec.get("text_corrected", "")[:100]


def imagery_jaccard(q_tags: str | list, c_tags: str | list) -> float:
    """Jaccard overlap between two sets of imagery tags."""
    def _tokenise(t):
        if isinstance(t, list):
            return {x.lower().strip() for x in t if x.strip()}
        if not t or str(t) in ("nan", "None", ""):
            return set()
        return {x.lower().strip() for x in str(t).split(",") if x.strip()}
    A = _tokenise(q_tags)
    B = _tokenise(c_tags)
    if not A or not B:
        return 0.0
    return len(A & B) / len(A | B)


def value_match(q_val: str, c_val: str) -> float:
    """1.0 if khaleeji_value matches exactly (case-insensitive), else 0.0."""
    if not q_val or not c_val:
        return 0.0
    return float(q_val.strip().lower() == c_val.strip().lower())


# ── Retrieval with re-ranking ──────────────────────────────────────────────────

def search_reranked(
    retriever:  NabatiRetriever,
    all_vecs:   np.ndarray,
    q_pos:      int,
    q_id:       str,
    q_imagery:  str | list,
    q_value:    str,
    top_k:      int,
    fetch_k:    int,
    beta:       float,
    gamma:      float,
) -> list[dict]:
    """Get fetch_k candidates from FAISS, then re-rank using metadata."""
    q = all_vecs[q_pos].reshape(1, -1).astype(np.float32)
    scores, idxs = retriever.text_index.search(q, fetch_k + 10)
    candidates = []
    for embed_score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        rec = retriever.records[idx]
        if record_id(rec) == q_id:
            continue
        img_score = imagery_jaccard(q_imagery, rec.get("imagery_tags_en", ""))
        val_score = value_match(q_value, rec.get("khaleeji_value", ""))
        final_score = float(embed_score) + beta * img_score + gamma * val_score
        candidates.append({**rec, "score": final_score, "_embed_score": float(embed_score)})
    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:top_k]


# ── Per-query GradedNDCG@K ────────────────────────────────────────────────────

def eval_queries(
    retriever:  NabatiRetriever,
    all_vecs:   np.ndarray,
    id_to_pos:  dict[str, int],
    queries:    list[dict],
    beta:       float,
    gamma:      float,
    k:          int = EVAL_K,
) -> tuple[float, float, float]:
    """
    Returns (mean_graded_ndcg, mean_poet_diversity, mean_imagery_coherence).
    """
    graded_ndcgs, diversities, imagery_scores = [], [], []

    for q_rec in queries:
        q_label  = q_rec.get("genre_en", "").strip()
        q_id     = record_id(q_rec)
        if not q_label:
            continue
        pos = id_to_pos.get(q_id)
        if pos is None:
            continue

        q_imagery = q_rec.get("imagery_tags_en", "")
        q_value   = q_rec.get("khaleeji_value", "")
        q_genre   = q_rec.get("genre_en", "")
        q_emotion = q_rec.get("emotion_text", "")

        results = search_reranked(
            retriever, all_vecs, pos, q_id,
            q_imagery, q_value, k, FETCH_K, beta, gamma,
        )
        if not results:
            continue

        # GradedNDCG@10
        gndcg = graded_ndcg_at_k(results, q_genre, q_emotion, k=k,
                                  genre_key="genre_en", emotion_key="emotion_text")
        graded_ndcgs.append(gndcg)

        # Poet Diversity@10
        poets = {r.get("poet_en", "").strip() for r in results if r.get("poet_en", "").strip()}
        diversities.append(len(poets))

        # Imagery Coherence@10
        raw_tags = q_imagery
        if isinstance(raw_tags, str):
            raw_tags = [t.strip() for t in raw_tags.split(",") if t.strip()]
        if raw_tags:
            ic = imagery_coherence_at_k(results, raw_tags, k=k, imagery_key="imagery_tags_en")
            if ic == ic:   # not NaN
                imagery_scores.append(ic)

    len(graded_ndcgs)
    return (
        float(np.mean(graded_ndcgs)) if graded_ndcgs else 0.0,
        float(np.mean(diversities)) if diversities else 0.0,
        float(np.mean(imagery_scores)) if imagery_scores else 0.0,
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.info("Loading retrieval index ...")
    retriever = NabatiRetriever.load(RETRIEVAL_DIR, device="cpu", load_encoder=False)
    n   = retriever.text_index.ntotal
    dim = NabatiRetriever.TEXT_DIM
    all_vecs = np.zeros((n, dim), dtype=np.float32)
    retriever.text_index.reconstruct_n(0, n, all_vecs)
    id_to_pos = {record_id(r): i for i, r in enumerate(retriever.records)}
    logger.info(f"Index: {n} vectors, {len(id_to_pos)} unique IDs")

    # Load queries
    def load_split(path: Path) -> list[dict]:
        rows = []
        for line in path.open():
            r = json.loads(line)
            if r.get("genre_en") and r.get("text_corrected"):
                rows.append(r)
        return rows

    val_queries  = load_split(VAL_SPLIT)
    test_queries = load_split(TEST_SPLIT)
    logger.info(f"Val queries: {len(val_queries)}  |  Test queries: {len(test_queries)}")

    # ── Baseline (β=0, γ=0) ─────────────────────────────────────────────────
    base_gndcg, base_div, base_img = eval_queries(
        retriever, all_vecs, id_to_pos, test_queries, beta=0.0, gamma=0.0
    )
    logger.info(f"Baseline (embed only):  GradedNDCG@10={base_gndcg:.4f}  "
                f"Diversity={base_div:.3f}  ImageryCoherence={base_img:.4f}")

    # ── Tune β,γ on VAL ──────────────────────────────────────────────────────
    beta_grid  = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]
    gamma_grid = [0.0, 0.05, 0.10, 0.15, 0.20]

    best_val_gndcg = -1.0
    best_beta = 0.0
    best_gamma = 0.0

    logger.info(f"Grid search over {len(beta_grid)}×{len(gamma_grid)} β,γ combinations on val ...")
    for beta, gamma in product(beta_grid, gamma_grid):
        val_gndcg, _, _ = eval_queries(
            retriever, all_vecs, id_to_pos, val_queries, beta=beta, gamma=gamma
        )
        if val_gndcg > best_val_gndcg:
            best_val_gndcg = val_gndcg
            best_beta  = beta
            best_gamma = gamma

    logger.info(f"Best val: β={best_beta}  γ={best_gamma}  GradedNDCG@10={best_val_gndcg:.4f}")

    # ── Apply best to TEST ────────────────────────────────────────────────────
    rerank_gndcg, rerank_div, rerank_img = eval_queries(
        retriever, all_vecs, id_to_pos, test_queries, beta=best_beta, gamma=best_gamma
    )
    logger.info(f"Re-ranked (β={best_beta}, γ={best_gamma}):  "
                f"GradedNDCG@10={rerank_gndcg:.4f}  "
                f"Diversity={rerank_div:.3f}  ImageryCoherence={rerank_img:.4f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    improvement = rerank_gndcg - base_gndcg
    adopted = improvement > 0.002   # threshold: >0.2pp improvement

    logger.info("=" * 60)
    logger.info(f"GradedNDCG@10: {base_gndcg:.4f} → {rerank_gndcg:.4f} (Δ={improvement:+.4f})")
    logger.info(f"PoetDiversity@10: {base_div:.3f} → {rerank_div:.3f}")
    logger.info(f"ImageryCoherence@10: {base_img:.4f} → {rerank_img:.4f}")
    logger.info(f"Decision: {'ADOPT re-ranking' if adopted else 'REJECT — no meaningful gain'}")

    report = {
        "method": "Metadata re-ranking: embed_score + β*imagery_jaccard + γ*value_match",
        "fetch_k": FETCH_K,
        "tuning_split": "val",
        "best_beta":  best_beta,
        "best_gamma": best_gamma,
        "baseline": {
            "beta": 0.0, "gamma": 0.0,
            "test_GradedNDCG_10": round(base_gndcg, 4),
            "test_PoetDiversity_10": round(base_div, 3),
            "test_ImageryCoherence_10": round(base_img, 4),
        },
        "reranked": {
            "beta": best_beta, "gamma": best_gamma,
            "val_GradedNDCG_10": round(best_val_gndcg, 4),
            "test_GradedNDCG_10": round(rerank_gndcg, 4),
            "test_PoetDiversity_10": round(rerank_div, 3),
            "test_ImageryCoherence_10": round(rerank_img, 4),
        },
        "delta_GradedNDCG_10": round(improvement, 4),
        "adopted": adopted,
        "interpretation": (
            f"Metadata re-ranking (imagery Jaccard + value match) {'improves' if adopted else 'does not improve'} "
            f"GradedNDCG@10 from {base_gndcg:.3f} to {rerank_gndcg:.3f} "
            f"({'Δ=' + f'{improvement:+.3f}' + ' — adopted' if adopted else 'Δ=' + f'{improvement:+.3f}' + ' — negative result: embedding alone is sufficient'})."
        ),
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "retrieval_eval_rerank.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    logger.success(f"Report → {out}")


if __name__ == "__main__":
    main()

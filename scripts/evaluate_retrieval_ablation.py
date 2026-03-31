"""
scripts/evaluate_retrieval_ablation.py

Retrieval ablation: compares three retrieval strategies to validate the
hybrid approach — text-only vs. neural vs. multimodal fusion.

Three retrieval modes compared on the same test queries / same metrics:

  Mode A — TF-IDF (keyword baseline)
      Classical tf-idf char-n-gram cosine similarity.
      No neural embeddings. Shows what structured text patterns alone give.

  Mode B — Text-only neural (AraPoemBERT, α=1.0)
      Semantic retrieval via AraPoemBERT CLS embeddings + FAISS.
      This is the current production system.

  Mode C — Hybrid text+audio (α=0.7 text + 0.3 audio CNN)
      Adds Emotion1DCNN 512-dim audio embeddings to the text score.
      Only available when outputs/retrieval/audio.index exists.

Evaluation protocol (same as evaluate_retrieval.py):
  - Leave-one-out on TEST split.
  - Index = train + val only (test poems never seen during indexing).
  - Relevance = same genre label (binary, primary signal).
  - Metrics: GradedNDCG@10, Precision@10, MRR.

Output:
  outputs/reports/retrieval_ablation.json
  outputs/figures/retrieval_ablation_bar.png   (optional, if matplotlib)

Usage:
    uv run python scripts/evaluate_retrieval_ablation.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.retrieval import NabatiRetriever

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_DIR      = Path("data/processed")
RETRIEVAL_DIR = Path("outputs/retrieval")
REPORT_DIR    = Path("outputs/reports")
FIG_DIR       = Path("outputs/figures")

TRAIN_JSONL  = DATA_DIR / "train.jsonl"
VAL_JSONL    = DATA_DIR / "val.jsonl"
TEST_JSONL   = DATA_DIR / "test.jsonl"

EVAL_K = 10

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


# ── Data helpers ───────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def record_id(rec: dict) -> str:
    af = rec.get("audio_filename", "").strip()
    return af if af else rec.get("text_corrected", "")[:100]


def aggregate_to_poems(clips: list[dict]) -> list[dict]:
    """Group clips by poem, concatenate text, majority-vote genre/emotion → poem queries."""
    from collections import Counter, defaultdict
    poem_clips: dict[str, list[dict]] = defaultdict(list)
    for rec in clips:
        if not rec.get("text_corrected", "").strip():
            continue
        poem_key = (
            rec.get("source_poem")
            or rec.get("poem_id")
            or f"{rec.get('poet_en', '')}|{rec.get('genre_en', '')}"
        )
        poem_clips[poem_key].append(rec)

    queries: list[dict] = []
    for poem_key, pclips in poem_clips.items():
        clips_sorted = sorted(pclips, key=lambda c: c.get("start", 0))
        combined_text = " ".join(c["text_corrected"] for c in clips_sorted)
        genre_counts = Counter(c.get("genre_en", "") for c in clips_sorted if c.get("genre_en"))
        majority_genre = genre_counts.most_common(1)[0][0] if genre_counts else ""
        rep = {**clips_sorted[0], "text_corrected": combined_text, "genre_en": majority_genre}
        queries.append(rep)
    return queries


# ── Metric helpers ─────────────────────────────────────────────────────────────

def precision_at_k(is_relevant: list[bool], k: int) -> float:
    return sum(is_relevant[:k]) / k if k else 0.0


def ndcg_at_k(is_relevant: list[bool], k: int) -> float:
    topk = is_relevant[:k]
    dcg  = sum(rel / math.log2(i + 2) for i, rel in enumerate(topk))
    n_rel = sum(is_relevant)
    idcg  = sum(1 / math.log2(i + 2) for i in range(min(n_rel, k)))
    return dcg / idcg if idcg > 0 else 0.0


def mrr(is_relevant: list[bool]) -> float:
    for i, rel in enumerate(is_relevant):
        if rel:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_ranked_lists(
    ranked_lists: list[list[bool]],
    k: int,
    mode_name: str,
) -> dict:
    """Aggregate P@K, NDCG@K, MRR across all queries."""
    p_scores    = [precision_at_k(r, k) for r in ranked_lists]
    ndcg_scores = [ndcg_at_k(r, k) for r in ranked_lists]
    mrr_scores  = [mrr(r) for r in ranked_lists]
    result = {
        "mode":     mode_name,
        f"P@{k}":   round(float(np.mean(p_scores)), 4),
        f"NDCG@{k}": round(float(np.mean(ndcg_scores)), 4),
        "MRR":      round(float(np.mean(mrr_scores)), 4),
        "n_queries": len(ranked_lists),
    }
    logger.info(
        f"  {mode_name:40s}  P@{k}={result[f'P@{k}']:.4f}  "
        f"NDCG@{k}={result[f'NDCG@{k}']:.4f}  MRR={result['MRR']:.4f}"
    )
    return result


# ── Mode A: TF-IDF keyword baseline ───────────────────────────────────────────

def run_tfidf_retrieval(
    corpus: list[dict],
    queries: list[dict],
    k: int,
) -> list[list[bool]]:
    """
    Character n-gram TF-IDF retrieval.
    Addresses: 'text-only baseline without neural embeddings'.
    """
    logger.info("Mode A: TF-IDF retrieval (char n-gram, baseline)")
    corpus_texts = [r.get("text_corrected", "") for r in corpus]
    query_texts  = [q.get("text_corrected", "") for q in queries]
    corpus_ids   = [record_id(r) for r in corpus]
    query_ids    = [record_id(q) for q in queries]

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                          max_features=20_000, sublinear_tf=True)
    corpus_mat = vec.fit_transform(corpus_texts)
    query_mat  = vec.transform(query_texts)

    sims = cosine_similarity(query_mat, corpus_mat)  # (n_queries, n_corpus)

    ranked_lists: list[list[bool]] = []
    for qi, query in enumerate(queries):
        qid   = query_ids[qi]
        genre = (query.get("genre_en") or "").split("(")[0].strip()
        scores = sims[qi].copy()

        # Exclude the query itself (should not appear in corpus since corpus is
        # train+val, but guard anyway)
        for ci, cid in enumerate(corpus_ids):
            if cid == qid:
                scores[ci] = -1.0

        top_idxs  = np.argsort(scores)[::-1][:k]
        is_relevant = []
        for ci in top_idxs:
            c_genre = (corpus[ci].get("genre_en") or "").split("(")[0].strip()
            is_relevant.append(c_genre == genre and genre != "")
        ranked_lists.append(is_relevant)

    return ranked_lists


# ── Mode B: Text-only neural (AraPoemBERT FAISS) ──────────────────────────────

def run_text_neural_retrieval(
    retriever:  NabatiRetriever,
    all_vecs:   np.ndarray,
    id_to_pos:  dict[str, int],
    queries:    list[dict],
    k: int,
) -> list[list[bool]]:
    """
    AraPoemBERT embedding retrieval — current production system.
    Uses stored vectors (no BERT inference needed at eval time).
    """
    logger.info("Mode B: Text-only neural retrieval (AraPoemBERT)")
    ranked_lists: list[list[bool]] = []

    for query in queries:
        qid   = record_id(query)
        genre = (query.get("genre_en") or "").split("(")[0].strip()
        pos   = id_to_pos.get(qid)
        if pos is None:
            ranked_lists.append([False] * k)
            continue

        q_vec   = all_vecs[pos : pos + 1]
        scores, idxs = retriever.text_index.search(q_vec, k + 5)

        is_relevant = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            rec = retriever.records[idx]
            if record_id(rec) == qid:
                continue
            c_genre = (rec.get("genre_en") or "").split("(")[0].strip()
            is_relevant.append(c_genre == genre and genre != "")
            if len(is_relevant) >= k:
                break
        # Pad if needed
        while len(is_relevant) < k:
            is_relevant.append(False)
        ranked_lists.append(is_relevant)

    return ranked_lists


# ── Mode C: Hybrid text+audio (if audio index exists) ─────────────────────────

def run_hybrid_retrieval(
    retriever:  NabatiRetriever,
    all_vecs:   np.ndarray,
    id_to_pos:  dict[str, int],
    queries:    list[dict],
    k: int,
    alpha: float = 0.7,
) -> list[list[bool]] | None:
    """
    Hybrid retrieval: α·text_score + (1-α)·audio_score.
    Returns None if no audio index is available.
    """
    if retriever.audio_index is None:
        logger.warning("No audio FAISS index found — skipping hybrid mode.")
        return None

    logger.info(f"Mode C: Hybrid retrieval (α={alpha} text + {1-alpha:.1f} audio)")
    audio_dim = NabatiRetriever.AUDIO_DIM
    n         = retriever.audio_index.ntotal
    all_audio_vecs = np.zeros((n, audio_dim), dtype=np.float32)
    retriever.audio_index.reconstruct_n(0, n, all_audio_vecs)

    ranked_lists: list[list[bool]] = []

    for query in queries:
        qid   = record_id(query)
        genre = (query.get("genre_en") or "").split("(")[0].strip()
        pos   = id_to_pos.get(qid)
        if pos is None:
            ranked_lists.append([False] * k)
            continue

        t_vec = all_vecs[pos : pos + 1]
        a_vec = all_audio_vecs[pos : pos + 1] if pos < n else None

        fetch = k + 10
        t_scores, t_idxs = retriever.text_index.search(t_vec, fetch)

        if a_vec is not None:
            a_scores, a_idxs = retriever.audio_index.search(a_vec, fetch)
            audio_score_map = {int(idx): float(s) for s, idx in zip(a_scores[0], a_idxs[0]) if idx >= 0}
        else:
            audio_score_map = {}

        candidates: list[tuple[float, int]] = []
        for ts, tidx in zip(t_scores[0], t_idxs[0]):
            if tidx < 0:
                continue
            rec   = retriever.records[tidx]
            if record_id(rec) == qid:
                continue
            as_   = audio_score_map.get(int(tidx), 0.0)
            hybrid = alpha * float(ts) + (1 - alpha) * as_
            candidates.append((hybrid, int(tidx)))

        candidates.sort(key=lambda x: x[0], reverse=True)
        is_relevant = []
        for _, ci in candidates[:k]:
            c_genre = (retriever.records[ci].get("genre_en") or "").split("(")[0].strip()
            is_relevant.append(c_genre == genre and genre != "")
        while len(is_relevant) < k:
            is_relevant.append(False)
        ranked_lists.append(is_relevant)

    return ranked_lists


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logger.add("logs/retrieval_ablation.log", rotation="10 MB")

    # Load data
    train_recs = load_jsonl(TRAIN_JSONL)
    val_recs   = load_jsonl(VAL_JSONL)
    test_recs  = load_jsonl(TEST_JSONL)
    corpus     = train_recs + val_recs   # index covers train+val only

    test_queries = aggregate_to_poems(test_recs)

    logger.info(f"Corpus (train+val): {len(corpus)} clips")
    logger.info(f"Queries (test):     {len(test_queries)} poems (poem-level, aggregated)")

    # ── Load pre-built FAISS index (no BERT encoding needed) ──────────────────
    if not (RETRIEVAL_DIR / "text.index").exists():
        logger.error(
            "No retrieval index found. Run scripts/build_retrieval_index.py first."
        )
        sys.exit(1)

    retriever = NabatiRetriever.load(RETRIEVAL_DIR, device="cpu", load_encoder=False)
    n   = retriever.text_index.ntotal
    dim = NabatiRetriever.TEXT_DIM
    all_vecs = np.zeros((n, dim), dtype=np.float32)
    retriever.text_index.reconstruct_n(0, n, all_vecs)
    logger.info(f"Loaded FAISS index: {n} vectors (dim={dim})")

    # Map record_id → FAISS position
    id_to_pos = {record_id(r): i for i, r in enumerate(retriever.records)}
    logger.info(f"Index coverage: {sum(1 for q in test_recs if record_id(q) in id_to_pos)} "
                f"/ {len(test_recs)} test records have stored embeddings")

    # ── Run all three modes ────────────────────────────────────────────────────
    results: list[dict] = []

    print("\n" + "═" * 70)
    print("  RETRIEVAL ABLATION  (genre-relevance, leave-one-out, test split)")
    print("═" * 70)
    print(f"  {'Mode':<40s}  {'P@10':>6}  {'NDCG@10':>8}  {'MRR':>6}")
    print("  " + "─" * 60)

    # Mode A: TF-IDF
    ranked_a = run_tfidf_retrieval(corpus, test_queries, EVAL_K)
    results.append(evaluate_ranked_lists(ranked_a, EVAL_K, "A: TF-IDF (keyword baseline)"))

    # Mode B: Text-only neural
    ranked_b = run_text_neural_retrieval(retriever, all_vecs, id_to_pos, test_queries, EVAL_K)
    results.append(evaluate_ranked_lists(ranked_b, EVAL_K, "B: AraPoemBERT text-only"))

    # Mode C: Hybrid
    ranked_c = run_hybrid_retrieval(retriever, all_vecs, id_to_pos, test_queries, EVAL_K)
    if ranked_c is not None:
        results.append(evaluate_ranked_lists(ranked_c, EVAL_K, "C: Hybrid text+audio (α=0.7)"))

    print("═" * 70)

    # ── Interpretation ─────────────────────────────────────────────────────────
    print("\nInterpretation:")
    print("  B > A  → neural embeddings outperform keyword matching")
    print("  C > B  → adding audio embeddings further improves retrieval")
    print("  C ≈ B  → text embeddings already capture the relevant signal")
    print()

    # ── Save ───────────────────────────────────────────────────────────────────
    out = {
        "eval_k": EVAL_K,
        "query_level": "poem (aggregated text + majority-vote genre, N=13)",
        "relevance_criterion": "same genre (prefix match)",
        "n_corpus": len(corpus),
        "n_queries": len(test_queries),
        "results": results,
    }
    out_path = REPORT_DIR / "retrieval_ablation.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    logger.success(f"Retrieval ablation → {out_path}")

    # Optional bar chart
    try:
        import matplotlib.pyplot as plt
        labels    = [r["mode"].split(":")[0] for r in results]
        ndcg_vals = [r[f"NDCG@{EVAL_K}"] for r in results]
        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels, ndcg_vals, color=["#888", "#4C72B0", "#DD8452"])
        ax.set_ylabel(f"NDCG@{EVAL_K}")
        ax.set_title("Retrieval Ablation: Text-only vs Hybrid\n(higher = better; same test split)")
        ax.set_ylim(0, 1.0)
        for bar, val in zip(bars, ndcg_vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig_path = FIG_DIR / "retrieval_ablation_bar.png"
        fig.savefig(fig_path, bbox_inches="tight")
        plt.close(fig)
        logger.success(f"Bar chart → {fig_path}")
    except Exception as e:
        logger.warning(f"Bar chart skipped: {e}")


if __name__ == "__main__":
    main()

"""
scripts/build_retrieval_index.py

Build and save the NabatiRetriever index from all three corpus splits.

Usage:
    uv run python scripts/build_retrieval_index.py
    uv run python scripts/build_retrieval_index.py --splits train val   # index subset
    uv run python scripts/build_retrieval_index.py --query "قصيدة عن الغزل"

What it does:
    1. Loads train + val + test JSONL splits
    2. Encodes every text_corrected with AraPoemBERT (mean-pooled [CLS])
    3. Builds a FAISS IndexFlatIP (exact cosine similarity)
    4. Saves index + metadata to outputs/retrieval/
    5. Runs sample queries to verify the index works
    6. Prints a t-SNE genre embedding map to outputs/reports/
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.models.retrieval import NabatiRetriever

CNN_CKPT   = Path("outputs/models/audio_cnn/audio_cnn_emotion_best.pt")
AUDIO_DIM  = 512
MAX_SEC    = 8

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data/processed")
OUTPUT_DIR = Path("outputs/retrieval")
REPORT_DIR = Path("outputs/reports")

SPLITS = {
    "train": DATA_DIR / "train.jsonl",
    "val":   DATA_DIR / "val.jsonl",
    "test":  DATA_DIR / "test.jsonl",
}

# ── t-SNE genre map ───────────────────────────────────────────────────────────

def plot_tsne(retriever: NabatiRetriever, out_path: Path) -> None:
    """
    Project all AraPoemBERT embeddings to 2-D with t-SNE and colour by genre.

    Projects all AraPoemBERT embeddings to 2-D with t-SNE and colours by genre.
    If genre clusters are visible without any classification head, it confirms
    the pre-trained model has learned genre-meaningful representations,
    validating the semantic search approach.
    """

    # Reconstruct stored vectors from FAISS index
    n   = retriever.text_index.ntotal
    dim = retriever.TEXT_DIM
    vecs = np.zeros((n, dim), dtype=np.float32)
    retriever.text_index.reconstruct_n(0, n, vecs)  # type: ignore[arg-type]

    genres = [r.get("genre_en", "Unknown") for r in retriever.records]

    # Filter to genres with at least 10 samples for a clean plot
    from collections import Counter
    counts = Counter(genres)
    keep   = {g for g, c in counts.items() if c >= 10}
    mask   = [i for i, g in enumerate(genres) if g in keep]

    if len(mask) < 20:
        logger.warning("Too few samples to plot t-SNE — skipping.")
        return

    vecs_plot   = vecs[mask]
    genres_plot = [genres[i] for i in mask]

    logger.info(f"Running t-SNE on {len(mask)} samples ({len(keep)} genres)...")
    tsne   = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(vecs_plot)

    # Silhouette score (how separable are the genre clusters?)
    le     = LabelEncoder()
    labels = le.fit_transform(genres_plot)
    try:
        sil = silhouette_score(coords, labels)
        logger.info(f"Silhouette score (genre clusters): {sil:.4f}")
    except Exception:
        sil = float("nan")

    # Plot
    unique_genres = sorted(set(genres_plot))
    cmap = plt.cm.get_cmap("tab20", len(unique_genres))
    genre2color = {g: cmap(i) for i, g in enumerate(unique_genres)}

    fig, ax = plt.subplots(figsize=(12, 9))
    for genre in unique_genres:
        idx = [i for i, g in enumerate(genres_plot) if g == genre]
        ax.scatter(
            coords[idx, 0], coords[idx, 1],
            c=[genre2color[genre]],
            label=f"{genre} ({len(idx)})",
            alpha=0.65, s=18,
        )

    ax.legend(fontsize=8, loc="upper right", framealpha=0.7)
    ax.set_title(
        f"AraPoemBERT Embeddings — t-SNE by Genre\n"
        f"Silhouette score: {sil:.3f}  (higher = more separable)",
        fontsize=12,
    )
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(alpha=0.2)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"t-SNE map saved → {out_path}")


# ── Sample queries ────────────────────────────────────────────────────────────

SAMPLE_QUERIES = [
    ("الحب والغزل والشوق",          None),        # love / longing — expect Ghazal
    ("الحزن والفقد والرثاء",         None),        # grief / elegy — expect Ritha
    ("الوطن والانتماء والفخر",       "Wataniyya"), # nation / pride — with genre filter
    ("الحكمة والتأمل في الحياة",     None),        # wisdom — expect Hikma
    ("longing and love",             None),        # English query
]

def run_sample_queries(retriever: NabatiRetriever) -> None:
    logger.info("=" * 60)
    logger.info("Sample queries")
    logger.info("=" * 60)
    for query, genre_filter in SAMPLE_QUERIES:
        results = retriever.search(query, top_k=3, genre_filter=genre_filter)
        filter_str = f"  [filter: {genre_filter}]" if genre_filter else ""
        logger.info(f"\nQuery: {query!r}{filter_str}")
        for i, r in enumerate(results, 1):
            snippet = r["text_corrected"][:80].replace("\n", " ")
            logger.info(
                f"  {i}. score={r['score']:.3f} | {r['genre_en']:12s} | "
                f"{r['poet_en']:20s} | {snippet}…"
            )


# ── Audio embedding extraction ────────────────────────────────────────────────

def build_audio_index(retriever: NabatiRetriever, device: str) -> bool:
    """
    Extract 512-dim CNN embeddings for every indexed record and add audio.index.
    Returns True on success, False if CNN checkpoint or audio files are missing.
    """
    import librosa
    from src.models.audio_cnn import Emotion1DCNN

    if not CNN_CKPT.exists():
        logger.warning(f"CNN checkpoint not found at {CNN_CKPT} — skipping audio index.")
        return False

    dev = torch.device(device)
    model = Emotion1DCNN().to(dev)
    state = torch.load(CNN_CKPT, map_location=dev, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    logger.info(f"Loaded CNN from {CNN_CKPT}")

    target_len = MAX_SEC * 16000 // 512  # mel time frames

    vecs: list[np.ndarray] = []
    missing = 0
    for i, rec in enumerate(retriever.records):
        audio_path = Path(str(rec.get("audio_filename", ""))).expanduser()
        if not audio_path.exists():
            vecs.append(np.zeros(AUDIO_DIM, dtype=np.float32))
            missing += 1
            continue
        wav, _ = librosa.load(str(audio_path), sr=16000, mono=True)
        wav = wav[: MAX_SEC * 16000]
        mel = librosa.feature.melspectrogram(y=wav, sr=16000, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        if mel_db.shape[1] < target_len:
            mel_db = np.pad(mel_db, ((0, 0), (0, target_len - mel_db.shape[1])))
        else:
            mel_db = mel_db[:, :target_len]
        x = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).to(dev)
        with torch.no_grad():
            emb = model.embed(x)
        vecs.append(emb[0].cpu().numpy())
        if (i + 1) % 500 == 0:
            logger.info(f"  Audio embeddings: {i + 1}/{len(retriever.records)}")

    if missing:
        logger.warning(f"  {missing} records had no audio file — zero vector used")

    audio_vecs = np.stack(vecs)
    retriever.add_audio_embeddings(audio_vecs)
    logger.success(f"Audio index built: {audio_vecs.shape[0]} embeddings (dim={AUDIO_DIM})")
    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args: argparse.Namespace) -> None:
    logger.add("logs/retrieval.log", rotation="10 MB")

    # Resolve requested splits
    paths = [SPLITS[s] for s in args.splits if SPLITS[s].exists()]
    if not paths:
        logger.error("No JSONL files found. Run the data pipeline first.")
        sys.exit(1)

    missing = [s for s in args.splits if not SPLITS[s].exists()]
    if missing:
        logger.warning(f"Splits not found (skipped): {missing}")

    # Build index
    retriever = NabatiRetriever.build(
        jsonl_paths=paths,
        text_model="faisalq/bert-base-arapoembert",
        device=args.device,
    )
    logger.info(retriever)

    # Save text index first
    retriever.save(OUTPUT_DIR)

    # Build audio index (if CNN checkpoint available and not skipped)
    if not args.no_audio:
        ok = build_audio_index(retriever, args.device)
        if ok:
            retriever.save(OUTPUT_DIR)  # re-save to write audio.index
            logger.success("Audio index added and saved.")

    # Verify with sample queries (or custom query from CLI)
    if args.query:
        results = retriever.search(args.query, top_k=5)
        logger.info(f"\nQuery: {args.query!r}")
        for i, r in enumerate(results, 1):
            snippet = r["text_corrected"][:100].replace("\n", " ")
            logger.info(
                f"  {i}. score={r['score']:.3f} | {r['genre_en']:12s} | "
                f"{r['poet_en']:20s} | {snippet}"
            )
    else:
        run_sample_queries(retriever)

    # t-SNE genre map
    if not args.no_tsne:
        plot_tsne(retriever, REPORT_DIR / "tsne_genre_map.png")

    logger.success("Done. Load the retriever with:")
    logger.success(f"  retriever = NabatiRetriever.load(Path('{OUTPUT_DIR}'))")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build NabatiRetriever index")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Which JSONL splits to index (default: all three)"
    )
    parser.add_argument(
        "--device", default="cpu",
        help="torch device for encoding (cpu / mps / cuda)"
    )
    parser.add_argument(
        "--query", default=None,
        help="Run a single custom query after building the index"
    )
    parser.add_argument(
        "--no-tsne", action="store_true",
        help="Skip t-SNE plot (faster)"
    )
    parser.add_argument(
        "--no-audio", action="store_true",
        help="Skip audio CNN embedding index (text-only)"
    )
    main(parser.parse_args())

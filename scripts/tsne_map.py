"""
scripts/tsne_map.py

t-SNE Poetry Map — dual-panel visualisation of AraPoemBERT CLS embeddings.

Panel 1: clips coloured by genre  (8 classes)
Panel 2: clips coloured by emotion (12 classes)

Data: pre-computed CLS embeddings at data/processed/embeddings/{train,val,test}.pkl
      Emotion labels joined from data/processed/{train,val,test}.jsonl via
      (poem_id, start) key.

Output: outputs/figures/tsne_poetry_map.png  (300 dpi)
        outputs/figures/tsne_poetry_map_genre.png   (single panel)
        outputs/figures/tsne_poetry_map_emotion.png (single panel)
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

from loguru import logger
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.labels import GENRE_CLASSES, EMOTION_CLASSES

PROJECT_ROOT = Path(__file__).parent.parent
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

PERPLEXITY = 40
N_ITER = 1000
RANDOM_STATE = 42
POINT_SIZE = 6
ALPHA = 0.55

# Colour palettes — visually distinct, colour-blind-friendly where possible
GENRE_COLORS = [
    "#E63946",  # Ghazal — vivid red
    "#457B9D",  # Shajan — steel blue
    "#2DC653",  # Fakhr — green
    "#F4A261",  # Hikma — warm orange
    "#A8DADC",  # Badawa — pale teal
    "#6A0572",  # Wataniyya — deep purple
    "#FFB703",  # Ritha — amber
    "#023E8A",  # Hija — navy
]

EMOTION_COLORS = [
    "#E63946",  # Longing — red
    "#FF758C",  # Delicate Love — pink
    "#457B9D",  # Sorrow — blue
    "#2DC653",  # Pride — green
    "#F4A261",  # Admiration — orange
    "#A8DADC",  # Contemplation — teal
    "#6A0572",  # Disappointment — purple
    "#023E8A",  # Defiance — navy
    "#FFB703",  # Hope — amber
    "#8ECAE6",  # Compassion — light blue
    "#FFBA08",  # Humor — yellow
    "#ADB5BD",  # Neutral — grey
]

ID2EMOTION = {i: label for i, label in enumerate(EMOTION_CLASSES)}
# JSONL stores short labels ("Sorrow"), EMOTION_CLASSES has long form ("Sorrow (Huzn)")
EMOTION2ID: dict[str, int] = {}
for i, label in enumerate(EMOTION_CLASSES):
    EMOTION2ID[label] = i  # full form
    short = label.split("(")[0].strip()  # short form
    EMOTION2ID[short] = i


# ─── Data loading ─────────────────────────────────────────────────────────────


def load_emotion_lookup() -> dict[tuple[str, int], str]:
    """Build (poem_id, start) → emotion_text lookup from all JSONL splits."""
    lookup: dict[tuple[str, int], str] = {}
    for split in ("train", "val", "test"):
        path = PROJECT_ROOT / f"data/processed/{split}.jsonl"
        for line in path.open():
            r = json.loads(line)
            key = (r["source_poem"], int(r["start"]))
            lookup[key] = r.get("emotion_text", "") or ""
    return lookup


def load_embeddings() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load CLS embeddings + genre labels + emotion labels from all splits.
    Returns (X, genre_ids, emotion_ids) where X has shape (N, 768).
    """
    emotion_lookup = load_emotion_lookup()
    all_cls: list[np.ndarray] = []
    all_genre: list[int] = []
    all_emotion: list[int] = []

    for split in ("train", "val", "test"):
        path = PROJECT_ROOT / f"data/processed/embeddings/{split}.pkl"
        with path.open("rb") as f:
            emb_dict = pickle.load(f)

        for poem_id, items in emb_dict.items():
            for item in items:
                cls = np.array(item["cls"], dtype=np.float32)
                genre_id = int(item["label"])
                start = int(item["start"])
                emotion_str = emotion_lookup.get((poem_id, start), "")
                emotion_id = EMOTION2ID.get(emotion_str, -1)

                all_cls.append(cls)
                all_genre.append(genre_id)
                all_emotion.append(emotion_id)

    X = np.stack(all_cls)
    genre_ids = np.array(all_genre)
    emotion_ids = np.array(all_emotion)
    logger.info(
        f"Loaded {len(X)} clips  |  genre labels: {np.bincount(genre_ids)}  |  emotion unknown: {(emotion_ids == -1).sum()}"
    )
    return X, genre_ids, emotion_ids


# ─── t-SNE ────────────────────────────────────────────────────────────────────


def run_tsne(X: np.ndarray) -> np.ndarray:
    logger.info(
        f"Running t-SNE  (n={len(X)}, perplexity={PERPLEXITY}, n_iter={N_ITER}) ..."
    )
    X_scaled = StandardScaler().fit_transform(X)
    tsne = TSNE(
        n_components=2,
        perplexity=PERPLEXITY,
        max_iter=N_ITER,
        random_state=RANDOM_STATE,
        learning_rate="auto",
        init="pca",
    )
    Z = tsne.fit_transform(X_scaled)
    logger.info(f"t-SNE done. Final KL divergence: {tsne.kl_divergence_:.4f}")
    return Z


# ─── Plotting helpers ─────────────────────────────────────────────────────────


def _short_label(label: str) -> str:
    """Return text before first parenthesis/comma/slash."""
    for sep in ("(", ",", "/", "،"):
        if sep in label:
            return label.split(sep)[0].strip()
    return label


def _scatter_panel(
    ax: plt.Axes,
    Z: np.ndarray,
    ids: np.ndarray,
    class_names: list[str],
    colors: list[str],
    title: str,
    skip_unknown: bool = True,
) -> None:
    for cls_id, cls_name in enumerate(class_names):
        mask = ids == cls_id
        if mask.sum() == 0:
            continue
        ax.scatter(
            Z[mask, 0],
            Z[mask, 1],
            c=colors[cls_id],
            s=POINT_SIZE,
            alpha=ALPHA,
            linewidths=0,
            rasterized=True,
        )

    if skip_unknown:
        unk_mask = ids == -1
        if unk_mask.sum() > 0:
            ax.scatter(
                Z[unk_mask, 0],
                Z[unk_mask, 1],
                c="#CCCCCC",
                s=POINT_SIZE * 0.5,
                alpha=0.3,
                linewidths=0,
            )

    legend_patches = [
        mpatches.Patch(color=colors[i], label=_short_label(class_names[i]))
        for i in range(len(class_names))
        if (ids == i).sum() > 0
    ]
    ax.legend(
        handles=legend_patches,
        fontsize=6.5,
        markerscale=1.8,
        loc="lower right",
        framealpha=0.85,
        handlelength=1.2,
        borderpad=0.6,
    )
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    X, genre_ids, emotion_ids = load_embeddings()
    Z = run_tsne(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "t-SNE Map of Nabati Poetry (AraPoemBERT CLS Embeddings)",
        fontsize=13,
        fontweight="bold",
        y=1.01,
    )

    _scatter_panel(
        axes[0], Z, genre_ids, GENRE_CLASSES, GENRE_COLORS, "Coloured by Genre"
    )
    _scatter_panel(
        axes[1], Z, emotion_ids, EMOTION_CLASSES, EMOTION_COLORS, "Coloured by Emotion"
    )

    plt.tight_layout()
    dual_path = FIGURES_DIR / "tsne_poetry_map.png"
    fig.savefig(dual_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved: {dual_path}")

    for panel, ids, classes, colors, name in [
        ("genre", genre_ids, GENRE_CLASSES, GENRE_COLORS, "Genre"),
        ("emotion", emotion_ids, EMOTION_CLASSES, EMOTION_COLORS, "Emotion"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 6.5))
        _scatter_panel(ax, Z, ids, classes, colors, f"Nabati Poetry — t-SNE by {name}")
        plt.tight_layout()
        path = FIGURES_DIR / f"tsne_poetry_map_{panel}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved: {path}")

    logger.info("")
    logger.info("Genre distribution in t-SNE plot:")
    for gid, name in enumerate(GENRE_CLASSES):
        n = (genre_ids == gid).sum()
        logger.info(f"  {_short_label(name):25s}: {n:4d} clips")

    logger.info("")
    logger.info("Emotion distribution in t-SNE plot:")
    for eid, name in enumerate(EMOTION_CLASSES):
        n = (emotion_ids == eid).sum()
        logger.info(f"  {_short_label(name):30s}: {n:4d} clips")
    unk = (emotion_ids == -1).sum()
    if unk:
        logger.info(f"  {'Unknown / unmatched':30s}: {unk:4d} clips")


if __name__ == "__main__":
    main()

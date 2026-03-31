"""
src/models/retrieval.py

NabatiRetriever — semantic search over the Nabati poetry corpus.

How it works:
    1. Each poem's text_corrected is encoded with AraPoemBERT → a 768-dim [CLS] vector.
    2. Vectors are L2-normalised, so inner-product == cosine similarity.
    3. FAISS IndexFlatIP stores all vectors for exact nearest-neighbour search.
    4. Optionally a second FAISS index covers audio CNN embeddings (512-dim),
       enabling hybrid text+audio search  (α·text_score + β·audio_score) —
       this is the α·text + β·acoustic fusion described in the proposal.

Usage:
    # Build once and save
    retriever = NabatiRetriever.build(
        jsonl_paths=[Path("data/processed/train.jsonl"), ...],
        text_model="faisalq/bert-base-arapoembert",
    )
    retriever.save(Path("outputs/retrieval"))

    # Load and query
    retriever = NabatiRetriever.load(Path("outputs/retrieval"))
    results = retriever.search("قصيدة عن الغزل والحب", top_k=5)
    results = retriever.search("longing", top_k=5, genre_filter="Ghazal")
    results = retriever.search("heart journey night", top_k=5, imagery_filter="heart", tag_boost=0.15)
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch
from loguru import logger
from transformers import AutoModel, AutoTokenizer

DEFAULT_TEXT_MODEL = "faisalq/bert-base-arapoembert"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _mean_pool(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Mean-pool token embeddings weighted by the attention mask.
    More robust than [CLS] alone for variable-length poetry verses.
    """
    mask = attention_mask.unsqueeze(-1).float()
    summed = (token_embeddings * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def _l2_normalise(vecs: np.ndarray) -> np.ndarray:
    """Normalise rows to unit length so FAISS inner product = cosine similarity."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-9)
    return (vecs / norms).astype(np.float32)


def _tag_overlap_score(query: str, tags: str) -> float:
    """
    Fraction of query words that appear in the imagery tags string.

    e.g. query="heart journey night", tags="Heart, journey, stars, longing"
         → 2/3 = 0.667

    Used to boost FAISS cosine scores when the query semantically aligns
    with a clip's imagery — a lightweight keyword signal on top of embeddings.
    """
    if not tags or str(tags) in ("nan", "None", ""):
        return 0.0
    q_words = set(query.lower().split())
    t_words = set(tags.lower().replace(",", " ").split())
    overlap = q_words & t_words
    return len(overlap) / max(len(q_words), 1)


# ── Main class ────────────────────────────────────────────────────────────────

class NabatiRetriever:
    """
    Semantic search over Nabati poetry corpus.

    Attributes
    ----------
    text_model_name : str
        HuggingFace model used for text encoding.
    records : list[dict]
        Metadata for each indexed poem (same order as FAISS index rows).
    text_index : faiss.Index
        FAISS IndexFlatIP over 768-dim text embeddings (cosine similarity).
    audio_index : faiss.Index | None
        FAISS IndexFlatIP over 512-dim audio CNN embeddings (optional).
    """

    TEXT_DIM  = 768   # AraPoemBERT hidden size
    AUDIO_DIM = 512   # Emotion1DCNN embed() output size
    BATCH     = 32    # encoding batch size

    def __init__(self) -> None:
        self.text_model_name: str          = ""
        self.records:         list[dict]   = []
        self.text_index:      faiss.Index  = None   # type: ignore[assignment]
        self.audio_index:     Optional[faiss.Index] = None

        # Lazy-loaded encoder (not persisted to disk — reloaded on demand)
        self._tokenizer  = None
        self._text_model = None
        self._device     = "cpu"

    # ── Encoding ──────────────────────────────────────────────────────────────

    def _resolve_model_source(self, model_name: str) -> str:
        local_dir = os.environ.get("NABAT_ARAPOEM_MODEL_DIR", "")
        if model_name == DEFAULT_TEXT_MODEL and local_dir and Path(local_dir).exists():
            return local_dir
        return model_name

    def _load_encoder(self, model_name: str, device: str) -> None:
        if self._text_model is not None:
            return
        model_source = self._resolve_model_source(model_name)
        logger.info(f"Loading text encoder: {model_source}")
        self._tokenizer  = AutoTokenizer.from_pretrained(model_source, local_files_only=True)
        self._text_model = AutoModel.from_pretrained(model_source, local_files_only=True).to(device).eval()
        self._device     = device
        self.text_model_name = model_source

    @torch.no_grad()
    def _encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode a list of Arabic strings → (N, 768) float32 numpy array."""
        all_vecs: list[np.ndarray] = []
        for i in range(0, len(texts), self.BATCH):
            batch = texts[i : i + self.BATCH]
            enc = self._tokenizer(
                batch,
                max_length=32,          # AraPoemBERT hard limit
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids      = enc["input_ids"].to(self._device)
            attention_mask = enc["attention_mask"].to(self._device)

            out = self._text_model(input_ids=input_ids, attention_mask=attention_mask)
            vecs = _mean_pool(out.last_hidden_state, attention_mask).cpu().numpy()
            all_vecs.append(vecs)

        return np.concatenate(all_vecs, axis=0)

    # ── Build ─────────────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        jsonl_paths: list[Path],
        text_model: str = DEFAULT_TEXT_MODEL,
        device: str = "cpu",
    ) -> "NabatiRetriever":
        """
        Build a retrieval index from one or more JSONL corpus files.

        Records with empty text_corrected are skipped.
        All splits (train + val + test) should be indexed so the full
        corpus is searchable — we are not 'training' here, just indexing.
        """
        retriever = cls()
        retriever._load_encoder(text_model, device)

        records: list[dict] = []
        for path in jsonl_paths:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    text = rec.get("text_corrected", "").strip()
                    if text:
                        records.append(rec)

        logger.info(f"Indexing {len(records)} poems from {len(jsonl_paths)} splits...")

        texts = [r["text_corrected"] for r in records]
        vecs  = retriever._encode_texts(texts)
        vecs  = _l2_normalise(vecs)

        # Build FAISS index — IndexFlatIP gives exact cosine similarity search
        index = faiss.IndexFlatIP(cls.TEXT_DIM)
        index.add(vecs)  # type: ignore[arg-type]
        logger.success(f"Text index built: {index.ntotal} vectors, dim={cls.TEXT_DIM}")

        retriever.records    = records
        retriever.text_index = index
        return retriever

    def add_audio_embeddings(self, audio_vecs: np.ndarray) -> None:
        """
        Add pre-computed audio CNN embeddings (512-dim) to enable hybrid search.
        audio_vecs must have the same row order as self.records.

        Call after build() once the audio CNN is trained:
            vecs = extract_audio_embeddings(model, records)
            retriever.add_audio_embeddings(vecs)
        """
        assert len(audio_vecs) == len(self.records), \
            f"audio_vecs rows ({len(audio_vecs)}) ≠ records ({len(self.records)})"
        normed = _l2_normalise(audio_vecs.astype(np.float32))
        index  = faiss.IndexFlatIP(self.AUDIO_DIM)
        index.add(normed)  # type: ignore[arg-type]
        self.audio_index = index
        logger.success(f"Audio index added: {index.ntotal} vectors, dim={self.AUDIO_DIM}")

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        genre_filter:    Optional[str] = None,
        emotion_filter:  Optional[str] = None,
        imagery_filter:  Optional[str] = None,
        tag_boost:       float = 0.15,
        alpha:           float = 1.0,
    ) -> list[dict]:
        """
        Search by text query. Returns top_k results sorted by similarity.

        Parameters
        ----------
        query : str
            Arabic or English search text. Query words are also matched
            against each clip's imagery_tags_en to re-rank results.
        top_k : int
            Number of results to return.
        genre_filter : str, optional
            Hard-filter results to this genre (partial match, e.g. "Ghazal").
        emotion_filter : str, optional
            Hard-filter results to this emotion label (partial match).
        imagery_filter : str, optional
            Hard-filter: only return clips whose imagery_tags_en contain
            this substring (e.g. "heart" keeps only clips tagged with heart).
        tag_boost : float
            Additive bonus applied to the FAISS cosine score when the query
            words overlap with a clip's imagery tags. Default 0.15 means a
            clip with 100% tag overlap ranks as if its cosine score were 0.15
            higher. Set to 0.0 to disable.
        alpha : float
            Weight for text score in hybrid search [0, 1].
            Ignored when no audio index is present.
        """
        # Encode query
        q_vec = self._encode_texts([query])          # (1, 768)
        q_vec = _l2_normalise(q_vec)

        # Over-fetch to allow for post-filter attrition and re-ranking
        fetch_k = min(top_k * 20, len(self.records))
        scores, idxs = self.text_index.search(q_vec, fetch_k)  # type: ignore[arg-type]

        # Build scored candidates with tag boost applied before filtering
        candidates: list[tuple[float, int]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0:
                continue
            rec  = self.records[idx]
            tags = str(rec.get("imagery_tags_en") or "")
            boosted = float(score) + tag_boost * _tag_overlap_score(query, tags)
            candidates.append((boosted, idx))

        # Re-sort by boosted score (FAISS already sorted by raw score)
        candidates.sort(key=lambda x: x[0], reverse=True)

        results: list[dict] = []
        for boosted_score, idx in candidates:
            rec = self.records[idx]

            # Genre / emotion post-filters — partial match
            if genre_filter and not rec.get("genre_en", "").lower().startswith(genre_filter.lower()):
                continue
            if emotion_filter and not rec.get("emotion_text", "").lower().startswith(emotion_filter.lower()):
                continue

            # Imagery hard-filter — substring match against full tags string
            if imagery_filter:
                tags = str(rec.get("imagery_tags_en") or "").lower()
                if imagery_filter.lower() not in tags:
                    continue

            results.append({
                "score":           boosted_score,
                "text_corrected":  rec.get("text_corrected", ""),
                "genre_en":        rec.get("genre_en", ""),
                "emotion_text":    rec.get("emotion_text", ""),
                "emotion_audio":   rec.get("emotion_audio"),
                "poet_en":         rec.get("poet_en", ""),
                "source_poem":     rec.get("source_poem", ""),
                "audio_filename":  rec.get("audio_filename", ""),
                "translation_en":  rec.get("translation_en"),
                "imagery_tags_en": rec.get("imagery_tags_en"),
            })

            if len(results) >= top_k:
                break

        return results

    def hybrid_search(
        self,
        query_text: str,
        audio_vec: np.ndarray,
        top_k: int = 5,
        alpha: float = 0.6,
        genre_filter: Optional[str] = None,
    ) -> list[dict]:
        """
        α·text_score + (1-α)·audio_score — the proposal's weighted similarity fusion.

        Parameters
        ----------
        query_text : str
            Arabic text query (from Whisper ASR or direct input).
        audio_vec : np.ndarray
            Shape (512,) — output of Emotion1DCNN.embed() on the query clip.
        alpha : float
            Weight for text score. Default 0.6 (text slightly dominant).
        """
        if self.audio_index is None:
            logger.warning("No audio index — falling back to text-only search.")
            return self.search(query_text, top_k=top_k, genre_filter=genre_filter)

        fetch_k = min(top_k * 20, len(self.records))

        # Text scores
        q_text = _l2_normalise(self._encode_texts([query_text]))
        t_scores, t_idxs = self.text_index.search(q_text, fetch_k)  # type: ignore[arg-type]

        # Audio scores
        q_audio = _l2_normalise(audio_vec.reshape(1, -1).astype(np.float32))
        a_scores, a_idxs = self.audio_index.search(q_audio, fetch_k)  # type: ignore[arg-type]

        # Combine: build score dict then merge
        score_map: dict[int, float] = {}
        for s, i in zip(t_scores[0], t_idxs[0]):
            if i >= 0:
                score_map[i] = alpha * float(s)
        for s, i in zip(a_scores[0], a_idxs[0]):
            if i >= 0:
                score_map[i] = score_map.get(i, 0.0) + (1 - alpha) * float(s)

        ranked = sorted(score_map.items(), key=lambda x: x[1], reverse=True)

        results: list[dict] = []
        for idx, score in ranked:
            rec = self.records[idx]
            if genre_filter and not rec.get("genre_en", "").lower().startswith(genre_filter.lower()):
                continue
            results.append({
                "score":           score,
                "text_score":      alpha * float(t_scores[0][list(t_idxs[0]).index(idx)] if idx in t_idxs[0] else 0),
                "audio_score":     (1 - alpha) * float(a_scores[0][list(a_idxs[0]).index(idx)] if idx in a_idxs[0] else 0),
                "text_corrected":  rec.get("text_corrected", ""),
                "genre_en":        rec.get("genre_en", ""),
                "emotion_text":    rec.get("emotion_text", ""),
                "emotion_audio":   rec.get("emotion_audio"),
                "poet_en":         rec.get("poet_en", ""),
                "source_poem":     rec.get("source_poem", ""),
                "audio_filename":  rec.get("audio_filename", ""),
                "imagery_tags_en": rec.get("imagery_tags_en"),
            })
            if len(results) >= top_k:
                break

        return results

    def search_poems(
        self,
        query: str,
        top_k: int = 5,
        genre_filter:   Optional[str] = None,
        emotion_filter: Optional[str] = None,
        imagery_filter: Optional[str] = None,
        tag_boost:      float = 0.15,
    ) -> list[dict]:
        """
        Poem-level search: retrieves clips then deduplicates by source_poem.

        For each poem, the best-scoring clip is kept as the representative.
        Returns top_k distinct poems sorted by their highest clip score.

        Result dict keys are identical to search() for drop-in compatibility:
            score, source_poem, poet_en, genre_en, emotion_text, emotion_audio,
            text_corrected (best-matching line), imagery_tags_en, audio_filename,
            translation_en, n_clips (total clips in that poem in the index).
        """
        # Over-fetch clips to ensure we surface top_k *distinct poems*
        fetch_clips = min(top_k * 30, len(self.records))
        clips = self.search(
            query=query,
            top_k=fetch_clips,
            genre_filter=genre_filter,
            emotion_filter=emotion_filter,
            imagery_filter=imagery_filter,
            tag_boost=tag_boost,
        )

        # Group by source_poem — keep clip with highest score per poem
        seen: dict[str, dict] = {}
        for clip in clips:
            pid = clip["source_poem"]
            if pid not in seen or clip["score"] > seen[pid]["score"]:
                seen[pid] = clip

        # Count total clips per poem across the full index (for display)
        poem_clip_counts: dict[str, int] = {}
        for rec in self.records:
            pid = rec.get("source_poem", "")
            poem_clip_counts[pid] = poem_clip_counts.get(pid, 0) + 1

        # Sort by score, return top_k poems
        ranked = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:top_k]
        for poem in ranked:
            poem["n_clips"] = poem_clip_counts.get(poem["source_poem"], 1)

        return ranked

    # ── Persist ───────────────────────────────────────────────────────────────

    def save(self, directory: Path) -> None:
        """Save FAISS index + metadata to directory."""
        directory.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.text_index, str(directory / "text.index"))
        if self.audio_index is not None:
            faiss.write_index(self.audio_index, str(directory / "audio.index"))
        meta = {
            "text_model_name": self.text_model_name,
            "records":         self.records,
        }
        with open(directory / "meta.pkl", "wb") as f:
            pickle.dump(meta, f)
        logger.success(f"Retriever saved → {directory}  ({len(self.records)} records)")

    @classmethod
    def load(
        cls,
        directory: Path,
        device: str = "cpu",
        load_encoder: bool = True,
    ) -> "NabatiRetriever":
        """Load a saved retriever from directory."""
        retriever = cls()
        with open(directory / "meta.pkl", "rb") as f:
            meta = pickle.load(f)
        retriever.records         = meta["records"]
        retriever.text_model_name = meta["text_model_name"]
        retriever.text_index      = faiss.read_index(str(directory / "text.index"))

        audio_path = directory / "audio.index"
        if audio_path.exists():
            retriever.audio_index = faiss.read_index(str(audio_path))

        if load_encoder:
            retriever._load_encoder(retriever.text_model_name, device)

        logger.success(
            f"Retriever loaded: {retriever.text_index.ntotal} text vectors"
            + (f" + {retriever.audio_index.ntotal} audio vectors" if retriever.audio_index else "")
        )
        return retriever

    # ── Stats ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        n_audio = self.audio_index.ntotal if self.audio_index else 0
        return (
            f"NabatiRetriever("
            f"model={self.text_model_name!r}, "
            f"text_vectors={self.text_index.ntotal if self.text_index else 0}, "
            f"audio_vectors={n_audio})"
        )

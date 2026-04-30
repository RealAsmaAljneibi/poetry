"""Shared utilities for analyze_emotion_errors.py and analyze_genre_errors.py."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ARAPOEM_MODEL = "faisalq/bert-base-arapoembert"


def load_arapoem_model(
    checkpoint: Path,
    num_labels: int,
    device: str,
) -> tuple:
    """Load AraPoemBERT + task checkpoint. Returns (tokenizer, model)."""
    tokenizer = AutoTokenizer.from_pretrained(ARAPOEM_MODEL, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        ARAPOEM_MODEL,
        num_labels=num_labels,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state, strict=False)
    return tokenizer, model.to(device).eval()


def batch_infer(
    model,
    tokenizer,
    texts: list[str],
    device: str,
    max_seq_len: int = 32,
    batch_size: int = 16,
) -> np.ndarray:
    """Tokenize and run forward pass in batches. Returns (N, C) float32 prob array."""
    all_probs: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(
            texts[start : start + batch_size],
            max_length=max_seq_len,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            ).logits
        all_probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(all_probs, axis=0)


def tfidf_top_terms(
    label_docs: dict[str, str],
    all_labels: list[str],
    top_k: int = 10,
) -> dict[str, list[str]]:
    """TF-IDF distinctive word extraction. Returns {label: [top_k words]}."""
    labels = [lbl for lbl in all_labels if lbl in label_docs]
    docs = [label_docs[lbl] for lbl in labels]
    vectorizer = TfidfVectorizer(
        token_pattern=r"(?u)\b\w+\b", lowercase=False, max_features=5000
    )
    matrix = vectorizer.fit_transform(docs)
    vocab = np.array(vectorizer.get_feature_names_out())
    top_terms: dict[str, list[str]] = {}
    for idx, label in enumerate(labels):
        row = matrix[idx].toarray().ravel()
        order = row.argsort()[::-1]
        top_terms[label] = [vocab[i] for i in order if row[i] > 0][:top_k]
    return top_terms


def top_confusion_pairs(
    rows: list[dict],
    top_k: int | None = None,
) -> list[tuple[tuple[str, str], int]]:
    """Counter-based confusion pairs sorted by frequency. top_k=None returns all."""
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        if row["true_label"] != row["pred_label"]:
            counts[(row["true_label"], row["pred_label"])] += 1
    return counts.most_common(top_k)

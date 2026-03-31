"""
scripts/poem_tfidf.py

Poem-level TF-IDF + LogReg baseline.

Uses the same build_poems + split_poems logic as train_poem_classifier.py
(same SEED=42 → identical train/val/test poem split).

This establishes the ceiling for bag-of-words features at poem level.
Clip-level TF-IDF: 0.467. Poem-level should be higher (more text per example).

Usage:
    uv run python scripts/poem_tfidf.py
"""

import sys
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.labels import GENRE_CLASSES
from scripts.train_poem_classifier import build_poems, split_poems

DATA_FILE = Path("data/processed/master_dataset.jsonl")


def main():
    logger.info("=== Poem-level TF-IDF + LogReg Baseline ===")

    # ── Build poems (same as AraBERT script) ─────────────────────────────────
    poems  = build_poems(DATA_FILE)
    splits = split_poems(poems)

    train_poems = splits["train"]
    val_poems   = splits["val"]
    test_poems  = splits["test"]

    logger.info(
        f"Split: train={len(train_poems)}  val={len(val_poems)}  test={len(test_poems)} poems"
    )

    X_train = [p["text"] for p in train_poems]
    y_train = [p["label"] for p in train_poems]
    X_val   = [p["text"] for p in val_poems]
    y_val   = [p["label"] for p in val_poems]
    X_test  = [p["text"] for p in test_poems]
    y_test  = [p["label"] for p in test_poems]

    # ── Pipeline: char n-gram + word n-gram TF-IDF ───────────────────────────
    # Same feature engineering as clip-level TF-IDF baseline (run_diagnostics.py)
    char_tfidf = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        max_features=50_000, sublinear_tf=True,
    )
    word_tfidf = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        max_features=50_000, sublinear_tf=True,
    )

    from scipy.sparse import hstack

    logger.info("Fitting TF-IDF features...")
    X_tr_char = char_tfidf.fit_transform(X_train)
    X_tr_word = word_tfidf.fit_transform(X_train)
    X_tr = hstack([X_tr_char, X_tr_word])

    X_va_char = char_tfidf.transform(X_val)
    X_va_word = word_tfidf.transform(X_val)
    X_va = hstack([X_va_char, X_va_word])

    X_te_char = char_tfidf.transform(X_test)
    X_te_word = word_tfidf.transform(X_test)
    X_te = hstack([X_te_char, X_te_word])

    # ── Grid over C ──────────────────────────────────────────────────────────
    best_val_f1, best_C = 0.0, 1.0
    for C in [0.01, 0.1, 0.5, 1.0, 5.0, 10.0]:
        clf = LogisticRegression(
            C=C, class_weight="balanced",
            solver="lbfgs", max_iter=2000,
        )
        clf.fit(X_tr, y_train)
        preds = clf.predict(X_va)
        val_f1 = f1_score(y_val, preds, average="macro", zero_division=0)
        logger.info(f"  C={C:<6}  val Macro-F1={val_f1:.4f}")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_C      = C

    logger.info(f"Best C={best_C}  val Macro-F1={best_val_f1:.4f}")

    # ── Final test evaluation with best C ────────────────────────────────────
    clf_best = LogisticRegression(
        C=best_C, class_weight="balanced",
        solver="lbfgs", max_iter=2000,
    )
    clf_best.fit(X_tr, y_train)
    test_preds = clf_best.predict(X_te)
    test_f1    = f1_score(y_test, test_preds, average="macro", zero_division=0)
    test_acc   = (np.array(test_preds) == np.array(y_test)).mean()

    logger.success(
        f"TEST (poem-level TF-IDF+LR) | Macro-F1={test_f1:.4f} | "
        f"accuracy={test_acc:.4f} | n={len(test_poems)} poems"
    )

    short = [c.split("(")[0].strip() for c in GENRE_CLASSES]
    present_ids = sorted(set(y_test))
    report = classification_report(
        y_test, test_preds,
        labels=present_ids,
        target_names=[short[i] for i in present_ids],
        zero_division=0,
    )
    logger.info(f"\nPer-class report:\n{report}")

    # ── Val TF-IDF (for completeness) ─────────────────────────────────────────
    val_preds = clf_best.predict(X_va)
    val_preds_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
    logger.info(f"Val Macro-F1 (best model): {val_preds_f1:.4f}")

    # Save report
    Path("outputs/reports").mkdir(parents=True, exist_ok=True)
    (Path("outputs/reports") / "poem_tfidf_genre_report.txt").write_text(report)
    logger.info("Report → outputs/reports/poem_tfidf_genre_report.txt")


if __name__ == "__main__":
    logger.add("logs/poem_tfidf.log", rotation="5 MB")
    main()

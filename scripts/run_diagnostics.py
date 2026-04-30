"""
scripts/run_diagnostics.py

Four diagnostic checks:
  1. TF-IDF + LogReg baseline  — is the genre label learnable by bag-of-words?
  2. Genre distribution per split — split balance sanity check
  3. Spot-check examples — taxonomy/mapping bug detection
  4. Tokenizer check — AraPoemBERT input pipeline sanity

Run:
    uv run python scripts/run_diagnostics.py
"""

import json
import random
import sys
from collections import Counter
from pathlib import Path

from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, accuracy_score
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.labels import GENRE_CLASSES, encode_genre

DATA_DIR = Path("data/processed")
TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE = DATA_DIR / "val.jsonl"
TEST_FILE = DATA_DIR / "test.jsonl"

SEPARATOR = "=" * 70


# ── helpers ──────────────────────────────────────────────────────────────────


def load_split(path: Path) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def extract_genre_text(records: list[dict]) -> tuple[list[str], list[int]]:
    texts, labels = [], []
    for rec in records:
        text = rec.get("text_corrected", "").strip()
        genre = rec.get("genre_en", "").strip()
        label = encode_genre(genre)
        if text and label != -1:
            texts.append(text)
            labels.append(label)
    return texts, labels


# ── 1. TF-IDF + LogReg ───────────────────────────────────────────────────────


def run_tfidf_logreg() -> None:
    logger.info(SEPARATOR)
    logger.info("DIAGNOSTIC 1 — TF-IDF + Logistic Regression (genre)")
    logger.info(SEPARATOR)

    train_recs = load_split(TRAIN_FILE)
    val_recs = load_split(VAL_FILE)
    test_recs = load_split(TEST_FILE)

    X_train, y_train = extract_genre_text(train_recs)
    X_val, y_val = extract_genre_text(val_recs)
    X_test, y_test = extract_genre_text(test_recs)

    logger.info(
        f"Train: {len(X_train)} samples | Val: {len(X_val)} | Test: {len(X_test)}"
    )

    # TF-IDF: character 3-5 grams catch morphological variants in Arabic
    tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        max_features=50_000,
        sublinear_tf=True,
    )
    X_tr = tfidf.fit_transform(X_train)
    X_va = tfidf.transform(X_val)
    X_te = tfidf.transform(X_test)

    logger.info(f"TF-IDF vocab size: {len(tfidf.vocabulary_):,}")

    # Also try word-level for comparison
    tfidf_w = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_features=30_000,
        sublinear_tf=True,
    )
    X_tr_w = tfidf_w.fit_transform(X_train)
    X_va_w = tfidf_w.transform(X_val)
    X_te_w = tfidf_w.transform(X_test)
    logger.info(f"Word TF-IDF vocab size: {len(tfidf_w.vocabulary_):,}")

    present_classes = sorted(set(y_train))
    present_names = [GENRE_CLASSES[i].split("(")[0].strip() for i in present_classes]

    for name, Xtr, Xva, Xte, label in [
        ("char 3-5gram", X_tr, X_va, X_te, "char"),
        ("word 1-2gram", X_tr_w, X_va_w, X_te_w, "word"),
    ]:
        clf = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )
        clf.fit(Xtr, y_train)

        val_preds = clf.predict(Xva)
        test_preds = clf.predict(Xte)

        val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
        test_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(y_val, val_preds)
        test_acc = accuracy_score(y_test, test_preds)

        logger.info(f"\n--- TF-IDF ({name}) + LogReg ---")
        logger.info(f"  Val  Macro-F1={val_f1:.4f}  Acc={val_acc:.4f}")
        logger.info(f"  Test Macro-F1={test_f1:.4f}  Acc={test_acc:.4f}")

        report = classification_report(
            y_test,
            test_preds,
            labels=present_classes,
            target_names=present_names,
            zero_division=0,
        )
        logger.info(f"\n  Per-class (test):\n{report}")

        probs = clf.predict_proba(Xte)
        errors = [
            (i, y_test[i], test_preds[i], probs[i].max())
            for i in range(len(y_test))
            if y_test[i] != test_preds[i]
        ]
        errors.sort(key=lambda x: -x[3])  # sort by model confidence
        logger.info("  Top-5 confident mistakes (true → pred, confidence):")
        for idx, true_id, pred_id, conf in errors[:5]:
            true_name = GENRE_CLASSES[true_id].split("(")[0].strip()
            pred_name = GENRE_CLASSES[pred_id].split("(")[0].strip()
            logger.info(
                f"    [{true_name}] → [{pred_name}] conf={conf:.2f} | "
                f"'{X_test[idx][:80]}'"
            )
        break  # print detailed report only for char ngram; summary for word follows above


# ── 2. Genre distribution per split ──────────────────────────────────────────


def run_split_distribution() -> None:
    logger.info(SEPARATOR)
    logger.info("DIAGNOSTIC 2 — Genre Distribution per Split")
    logger.info(SEPARATOR)

    counts: dict[str, Counter] = {}
    totals: dict[str, int] = {}

    for split_name, path in [
        ("train", TRAIN_FILE),
        ("val", VAL_FILE),
        ("test", TEST_FILE),
    ]:
        recs = load_split(path)
        genre_cnt: Counter = Counter()
        for rec in recs:
            g = rec.get("genre_en", "").strip()
            genre_cnt[g] += 1
        counts[split_name] = genre_cnt
        totals[split_name] = sum(genre_cnt.values())
        logger.info(f"{split_name.upper()}: {totals[split_name]} clips total")

    all_genres = sorted(
        set(g for c in counts.values() for g in c),
        key=lambda g: -counts["train"].get(g, 0),
    )

    header = f"{'Genre':42s}  {'Train':>10}  {'Val':>10}  {'Test':>10}"
    logger.info(f"\n{header}")
    logger.info("-" * len(header))

    for genre in all_genres:
        tr = counts["train"].get(genre, 0)
        va = counts["val"].get(genre, 0)
        te = counts["test"].get(genre, 0)

        tr_pct = tr / totals["train"] * 100 if totals["train"] else 0
        va_pct = va / totals["val"] * 100 if totals["val"] else 0
        te_pct = te / totals["test"] * 100 if totals["test"] else 0

        flag = ""
        # Flag if a genre is absent in val or test (train-only leakage)
        if va == 0 and tr > 0:
            flag += " ⚠ ABSENT-VAL"
        if te == 0 and tr > 0:
            flag += " ⚠ ABSENT-TEST"
        # Flag big drift between train% and test%
        if tr > 0 and te > 0 and abs(tr_pct - te_pct) > 10:
            flag += f" ⚠ DRIFT({tr_pct:.0f}%→{te_pct:.0f}%)"

        logger.info(
            f"{genre[:42]:42s}  "
            f"{tr:4d} ({tr_pct:4.1f}%)  "
            f"{va:4d} ({va_pct:4.1f}%)  "
            f"{te:4d} ({te_pct:4.1f}%)"
            f"{flag}"
        )

    logger.info("\nPoet-disjoint check (no poet should appear in >1 split):")
    poet_splits: dict[str, set] = {}
    for split_name, path in [
        ("train", TRAIN_FILE),
        ("val", VAL_FILE),
        ("test", TEST_FILE),
    ]:
        for rec in load_split(path):
            p = rec.get("poet_en", "")
            poet_splits.setdefault(p, set()).add(split_name)

    leaked = [(p, s) for p, s in poet_splits.items() if len(s) > 1]
    if leaked:
        logger.warning(f"  LEAK: {len(leaked)} poets appear in multiple splits!")
        for p, s in leaked[:5]:
            logger.warning(f"    {p}: {s}")
    else:
        logger.info("  OK — every poet appears in exactly one split.")


# ── 3. Spot-check examples ────────────────────────────────────────────────────


def run_spot_check(n_per_split: int = 5) -> None:
    logger.info(SEPARATOR)
    logger.info("DIAGNOSTIC 3 — Spot-check Examples (text + genre label)")
    logger.info(SEPARATOR)

    random.seed(42)
    # Sample from genres we suspect might be mislabeled
    priority_genres = {"Ghazal", "Shajan", "Hikma", "Wataniyya", "Fakhr"}

    for split_name, path in [
        ("train", TRAIN_FILE),
        ("val", VAL_FILE),
        ("test", TEST_FILE),
    ]:
        recs = load_split(path)
        # Prefer records from priority genres, then fill randomly
        priority = [
            r
            for r in recs
            if r.get("genre_en", "").split("(")[0].strip() in priority_genres
        ]
        sample = random.sample(priority, min(n_per_split, len(priority)))

        logger.info(f"\n--- {split_name.upper()} ({n_per_split} samples) ---")
        for rec in sample:
            genre = rec.get("genre_en", "N/A")
            emotion = rec.get("emotion_cat_en", "N/A")
            poet = rec.get("poet_en", "N/A")
            text = rec.get("text_corrected", rec.get("text", "")).strip()
            short = text[:120] + ("…" if len(text) > 120 else "")
            logger.info(f"  [{genre}] [{emotion}] — {poet}\n  Text: {short}")


# ── 4. Tokenizer check ────────────────────────────────────────────────────────


def run_tokenizer_check() -> None:
    logger.info(SEPARATOR)
    logger.info("DIAGNOSTIC 4 — AraPoemBERT Tokenizer Check")
    logger.info(SEPARATOR)

    MODEL = "faisalq/bert-base-arapoembert"
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)

    train_recs = load_split(TRAIN_FILE)
    random.seed(42)
    samples = random.sample(train_recs, min(8, len(train_recs)))

    MAX_LEN = 32  # AraPoemBERT's max_position_embeddings

    len_stats = []
    for rec in train_recs:
        text = rec.get("text_corrected", "").strip()
        if text:
            toks = tokenizer(text, truncation=False)["input_ids"]
            len_stats.append(len(toks))

    len_stats.sort()
    p50 = len_stats[len(len_stats) // 2]
    p90 = len_stats[int(len(len_stats) * 0.90)]
    p99 = len_stats[int(len(len_stats) * 0.99)]
    truncated = sum(1 for length in len_stats if length > MAX_LEN)

    logger.info(f"Tokenized length stats (train, n={len(len_stats)}):")
    logger.info(f"  Median={p50}  P90={p90}  P99={p99}  Max={max(len_stats)}")
    logger.info(
        f"  Truncated (>{MAX_LEN} tokens): {truncated} / {len(len_stats)} "
        f"({truncated / len(len_stats) * 100:.1f}%)"
    )
    logger.info(f"  [AraPoemBERT max_position_embeddings = {MAX_LEN}]")

    logger.info("\nSample tokenizations:")
    for rec in samples[:5]:
        text = rec.get("text_corrected", "").strip()
        genre = rec.get("genre_en", "?")
        enc = tokenizer(text, max_length=MAX_LEN, truncation=True)
        toks = tokenizer.convert_ids_to_tokens(enc["input_ids"])
        n_before = len(tokenizer(text, truncation=False)["input_ids"])
        n_after = len(enc["input_ids"])
        flag = f"  ← TRUNCATED {n_before}→{n_after}" if n_before > MAX_LEN else ""
        logger.info(
            f"\n  Genre: {genre}\n"
            f"  Text:  {text[:80]}\n"
            f"  Tokens ({n_before} total{flag}): {' | '.join(toks)}"
        )

    unk_id = tokenizer.unk_token_id
    total_toks = 0
    unk_toks = 0
    for rec in train_recs:
        text = rec.get("text_corrected", "").strip()
        if text:
            ids = tokenizer(text, max_length=MAX_LEN, truncation=True)["input_ids"]
            total_toks += len(ids)
            unk_toks += ids.count(unk_id)
    logger.info(
        f"\n[UNK] rate: {unk_toks}/{total_toks} tokens = {unk_toks / total_toks * 100:.2f}%"
    )
    if unk_toks / total_toks > 0.05:
        logger.warning("  HIGH UNK rate — tokenizer may not cover your dialect well!")
    else:
        logger.info("  Low UNK rate — tokenizer covers vocabulary well.")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.add("logs/diagnostics.log", rotation="10 MB")
    run_tfidf_logreg()
    run_split_distribution()
    run_spot_check()
    run_tokenizer_check()
    logger.info(SEPARATOR)
    logger.info("All diagnostics complete.")

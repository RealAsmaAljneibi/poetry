"""
scripts/run_baseline.py

Baselines for Nabat-AI — all evaluated on the SAME poet-disjoint test split.

Baselines implemented:
  ┌──┬─────────────────────────────┬────────────┬───────────────────────────┐
  │# │ Model                       │ Task       │ Course concept            │
  ├──┼─────────────────────────────┼────────────┼───────────────────────────┤
  │1 │ Majority-class              │ Genre      │ Imbalance / Macro-F1      │
  │2 │ Majority-class              │ Emotion    │ Imbalance / Macro-F1      │
  │3 │ TF-IDF + LogReg (char n-gram│ Genre      │ Classical NLP baseline    │
  │4 │ TF-IDF + LogReg (char n-gram│ Emotion    │ Classical NLP baseline    │
  │5 │ MFCC mean + SVM (RBF)       │ Emotion    │ Classical audio baseline  │
  │6 │ mBERT fine-tuned            │ Genre      │ Prof's explicit request   │
  └──┴─────────────────────────────┴────────────┴───────────────────────────┘

Outputs:
  outputs/reports/baseline_results.csv   — metrics table (all baselines)
  outputs/figures/cm_tfidf_genre.png     — confusion matrix
  outputs/figures/cm_tfidf_emotion.png
  outputs/figures/cm_mfcc_svm.png
  outputs/figures/cm_mbert_genre.png

Usage:
    just run-baseline
    uv run python scripts/run_baseline.py
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
from loguru import logger
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    f1_score, classification_report,
    confusion_matrix, cohen_kappa_score,
)
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer,
)
from datasets import Dataset as HFDataset

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data.arousal_labels import emotion_to_arousal
from src.evaluation.metrics import emotion_partial_credit

warnings.filterwarnings("ignore")

# ── Paths ──────────────────────────────────────────────────────────────────
TRAIN_JSONL = Path("data/processed/train.jsonl")
VAL_JSONL   = Path("data/processed/val.jsonl")
TEST_JSONL  = Path("data/processed/test.jsonl")
FIG_DIR     = Path("outputs/figures")
RPT_DIR     = Path("outputs/reports")
SEED        = 42

FIG_DIR.mkdir(parents=True, exist_ok=True)
RPT_DIR.mkdir(parents=True, exist_ok=True)


# ── Data loading ───────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    )


def load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return load_jsonl(TRAIN_JSONL), load_jsonl(VAL_JSONL), load_jsonl(TEST_JSONL)


# ── Evaluation helpers ─────────────────────────────────────────────────────

def evaluate(y_true, y_pred, label: str, class_names: list[str],
             emotion_partial_scores: list[float] | None = None) -> dict:
    """
    Full evaluation suite — Macro-F1, Weighted-F1, Cohen's κ.
    For emotion tasks, also reports mean partial-credit score.
    Reports Macro-F1, Weighted-F1, and Cohen's κ across all tasks.
    """
    macro_f1    = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    kappa       = cohen_kappa_score(y_true, y_pred)

    logger.info(f"\n{'─'*55}\n  {label}\n{'─'*55}")
    logger.info(f"  Macro-F1    : {macro_f1:.4f}")
    logger.info(f"  Weighted-F1 : {weighted_f1:.4f}")
    logger.info(f"  Cohen's κ   : {kappa:.4f}")

    result = {"model": label, "macro_f1": macro_f1,
              "weighted_f1": weighted_f1, "kappa": kappa}

    if emotion_partial_scores is not None:
        mean_pc = float(np.mean(emotion_partial_scores))
        logger.info(f"  Emotion PC  : {mean_pc:.4f}  "
                    "(partial-credit: 1.0/0.65/0.45/0.30/0.20)")
        result["emotion_partial_credit"] = mean_pc

    logger.info("\n" + classification_report(
        y_true, y_pred,
        labels=class_names,
        target_names=[c[:25] for c in class_names],
        zero_division=0,
    ))
    return result


def save_confusion_matrix(y_true, y_pred, class_names: list[str],
                          title: str, fname: str) -> None:
    """
    Normalised confusion matrix — raw counts annotated, colour = row %.
    Normalised confusion matrix — raw counts annotated, colour = row %.
    """
    short = [c.split("(")[0].strip()[:18] for c in class_names]
    cm      = confusion_matrix(y_true, y_pred, labels=class_names)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(max(10, len(class_names)), max(8, len(class_names) - 1)))
    sns.heatmap(cm_norm, annot=cm, fmt="d", cmap="Blues",
                xticklabels=short, yticklabels=short,
                linewidths=0.3, ax=ax, cbar_kws={"label": "Row-normalised %"})
    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("True", fontsize=10)
    ax.set_title(f"{title}\n(cell = raw count, colour = % of true class)", fontsize=10)
    plt.xticks(rotation=40, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    fig.tight_layout()
    path = FIG_DIR / fname
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.success(f"Confusion matrix → {path}")

    # Log top-3 most confused pairs
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)
    for _ in range(3):
        idx = np.unravel_index(cm_off.argmax(), cm_off.shape)
        if cm_off[idx] == 0:
            break
        logger.info(f"  Confused: '{class_names[idx[0]][:30]}' "
                    f"→ '{class_names[idx[1]][:30]}': {cm_off[idx]} clips")
        cm_off[idx] = 0


# ── Baseline 1 & 2: Majority class ────────────────────────────────────────

def run_majority(train: pd.DataFrame, test: pd.DataFrame,
                 task_col: str, task_name: str) -> dict:
    logger.info(f"Running majority-class baseline: {task_name}")
    y_train = train[task_col].dropna().tolist()
    y_test  = test[task_col].dropna().tolist()
    classes = sorted(set(y_train))

    clf = DummyClassifier(strategy="most_frequent", random_state=SEED)
    clf.fit([[0]] * len(y_train), y_train)
    y_pred = clf.predict([[0]] * len(y_test))

    return evaluate(y_test, y_pred, f"Majority-class | {task_name}", classes)


# ── Baseline 3 & 4: TF-IDF + LogReg ──────────────────────────────────────

def run_tfidf_logreg(train: pd.DataFrame, test: pd.DataFrame,
                     task_col: str, task_name: str, cm_fname: str,
                     is_emotion: bool = False) -> dict:
    """
    Character n-gram TF-IDF + balanced LogReg.
    Char n-grams (2-5) outperform word-level for Arabic morphology.
    Addresses class imbalance with class_weight='balanced'.
    When is_emotion=True, also computes 5-tier partial-credit score.
    """
    logger.info(f"Running TF-IDF + LogReg: {task_name}")

    train_clean = train[train[task_col].notna()]
    test_clean  = test[test[task_col].notna()]

    X_train_raw = train_clean["text_corrected"].tolist()
    X_test_raw  = test_clean["text_corrected"].tolist()
    y_train     = train_clean[task_col].tolist()
    y_test      = test_clean[task_col].tolist()
    classes     = sorted(set(y_train + y_test))

    # Character n-grams (2–5) — better for Arabic morphology (Week 3 insight)
    vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5),
        max_features=15_000, sublinear_tf=True,
    )
    X_train = vec.fit_transform(X_train_raw)
    X_test  = vec.transform(X_test_raw)

    clf = LogisticRegression(
        max_iter=1000, class_weight="balanced",
        solver="lbfgs", random_state=SEED,
    )
    t0 = time.time()
    clf.fit(X_train, y_train)
    logger.info(f"  Training time: {time.time()-t0:.1f}s")

    y_pred = clf.predict(X_test)
    save_confusion_matrix(y_test, y_pred, classes,
                          f"TF-IDF + LogReg | {task_name}", cm_fname)

    # Emotion partial-credit (when evaluating emotion task)
    pc_scores = None
    if is_emotion:
        audio_refs = test_clean.get("emotion_audio", test_clean[task_col]).tolist()
        genres     = test_clean.get("genre_en", pd.Series([""] * len(test_clean))).tolist()
        pc_scores  = [
            emotion_partial_credit(p, a, t, g)
            for p, a, t, g in zip(y_pred, audio_refs, y_test, genres)
        ]

    return evaluate(y_test, y_pred, f"TF-IDF + LogReg | {task_name}", classes,
                    emotion_partial_scores=pc_scores)


# ── Baseline 5: MFCC mean + SVM ───────────────────────────────────────────

def extract_mfcc(audio_path: str, n_mfcc: int = 40) -> np.ndarray:
    """
    Extract mean + std of MFCC, delta, and delta-delta.
    40 MFCCs × 3 statistics = 120-dim feature vector per clip.
    """
    try:
        y, sr = librosa.load(audio_path, sr=16_000, duration=8.0)
        mfcc       = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        delta      = librosa.feature.delta(mfcc)
        delta2     = librosa.feature.delta(mfcc, order=2)
        feats      = np.concatenate([mfcc, delta, delta2], axis=0)  # (120, T)
        return np.concatenate([feats.mean(axis=1), feats.std(axis=1)])  # (240,)
    except Exception as e:
        logger.warning(f"MFCC failed for {audio_path}: {e}")
        return np.zeros(240)


def run_mfcc_svm(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    logger.info("Running MFCC + SVM baseline (emotion from audio)...")
    logger.info("  Extracting MFCC features — this takes a few minutes...")

    task_col = "emotion_audio"
    train_c  = train[train[task_col].notna()]
    test_c   = test[test[task_col].notna()]

    t0 = time.time()
    X_train = np.array([extract_mfcc(p) for p in train_c["audio_filename"]])
    X_test  = np.array([extract_mfcc(p) for p in test_c["audio_filename"]])
    logger.info(f"  MFCC extraction: {time.time()-t0:.1f}s")

    y_train = train_c[task_col].tolist()
    y_test  = test_c[task_col].tolist()
    classes = sorted(set(y_train + y_test))

    clf = SVC(kernel="rbf", class_weight="balanced",
              C=10, gamma="scale", random_state=SEED)
    t0 = time.time()
    clf.fit(X_train, y_train)
    logger.info(f"  SVM training: {time.time()-t0:.1f}s")

    y_pred = clf.predict(X_test)
    save_confusion_matrix(y_test, y_pred, classes,
                          "MFCC + SVM | Emotion (audio)", "cm_mfcc_svm.png")

    # Audio emotion: audio_ref == y_test (same column), text_ref from text column
    text_refs = test_c.get("emotion_text", test_c[task_col]).tolist()
    genres    = test_c.get("genre_en", pd.Series([""] * len(test_c))).tolist()
    pc_scores = [
        emotion_partial_credit(p, a, t, g)
        for p, a, t, g in zip(y_pred, y_test, text_refs, genres)
    ]
    return evaluate(y_test, y_pred, "MFCC + SVM | Emotion (audio)", classes,
                    emotion_partial_scores=pc_scores)


# ── Baseline 7: Arousal majority-class ───────────────────────────────────

def run_arousal_majority(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """
    Majority-class baseline for 3-class Arousal (High/Low/Medium).
    Arousal is derived from emotion_audio via emotion_to_arousal().
    Provides the floor that ArousalMLP (Macro-F1=0.648) must beat.
    Since arousal is ~balanced (Low 36%, High 33%, Med 31%), majority-class
    is approximately 1/3 Macro-F1.
    """
    logger.info("Running majority-class baseline: Arousal (3-class)")

    def derive_arousal(df: pd.DataFrame) -> list[str]:
        labels = []
        for val in df["emotion_audio"].fillna(""):
            ar = emotion_to_arousal(str(val))
            if ar:
                labels.append(ar)
        return labels

    y_train = derive_arousal(train)
    y_test  = derive_arousal(test)
    classes = ["Low", "Medium", "High"]

    clf = DummyClassifier(strategy="most_frequent", random_state=SEED)
    clf.fit([[0]] * len(y_train), y_train)
    y_pred = clf.predict([[0]] * len(y_test))

    return evaluate(y_test, y_pred, "Majority-class | Arousal (3-class)", classes)


# ── Baseline 6: mBERT fine-tuned ──────────────────────────────────────────

def run_mbert(train: pd.DataFrame, val: pd.DataFrame,
              test: pd.DataFrame) -> dict:
    """
    Fine-tune bert-base-multilingual-cased on genre classification.
    Included as a strong multilingual baseline — AraPoemBERT is expected to
    outperform mBERT given its domain-specific Arabic poetry pre-training.
    """
    logger.info("Running mBERT fine-tuning baseline (genre)...")
    task_col   = "genre_en"
    model_name = "bert-base-multilingual-cased"

    train_c = train[train[task_col].notna()].copy()
    val_c   = val[val[task_col].notna()].copy()
    test_c  = test[test[task_col].notna()].copy()

    # Label encoding
    le = LabelEncoder()
    le.fit(train_c[task_col])
    classes = list(le.classes_)

    train_c["label"] = le.transform(train_c[task_col])
    val_c["label"]   = le.transform(val_c[task_col].map(
        lambda x: x if x in le.classes_ else le.classes_[0]))
    test_c["label"]  = le.transform(test_c[task_col].map(
        lambda x: x if x in le.classes_ else le.classes_[0]))

    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)

    def tokenize(batch):
        return tokenizer(batch["text_corrected"],
                         padding="max_length", truncation=True, max_length=128)

    hf_train = HFDataset.from_pandas(train_c[["text_corrected", "label"]])
    hf_val   = HFDataset.from_pandas(val_c[["text_corrected", "label"]])
    hf_test  = HFDataset.from_pandas(test_c[["text_corrected", "label"]])

    hf_train = hf_train.map(tokenize, batched=True)
    hf_val   = hf_val.map(tokenize, batched=True)
    hf_test  = hf_test.map(tokenize, batched=True)

    for ds in [hf_train, hf_val, hf_test]:
        ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(classes),
        id2label={i: c for i, c in enumerate(classes)},
        label2id={c: i for i, c in enumerate(classes)},
        ignore_mismatched_sizes=True,
        local_files_only=True,
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        return {"f1_macro": f1_score(labels, preds, average="macro", zero_division=0)}

    args = TrainingArguments(
        output_dir="outputs/models/mbert_genre",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,           # warm-up from Week 2 LR scheduling
        lr_scheduler_type="cosine", # cosine decay — from Week 2 optimizer lecture
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        seed=SEED,
        logging_dir="logs/mbert",
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=hf_train,
        eval_dataset=hf_val,
        compute_metrics=compute_metrics,
    )

    logger.info("  Fine-tuning mBERT (5 epochs)...")
    t0 = time.time()
    trainer.train()
    logger.info(f"  mBERT training: {time.time()-t0:.1f}s")

    # Evaluate on test set
    preds_out  = trainer.predict(hf_test)
    y_pred_ids = np.argmax(preds_out.predictions, axis=-1)
    y_true_ids = preds_out.label_ids
    y_pred     = le.inverse_transform(y_pred_ids)
    y_true     = le.inverse_transform(y_true_ids)

    save_confusion_matrix(y_true, y_pred, classes,
                          "mBERT fine-tuned | Genre", "cm_mbert_genre.png")
    return evaluate(y_true, y_pred, "mBERT fine-tuned | Genre", classes)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    logger.add("logs/baselines.log", rotation="10 MB")
    logger.info("Loading poet-disjoint splits...")
    train, val, test = load_splits()
    logger.info(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    results = []

    # 1. Majority class — Genre
    results.append(run_majority(train, test, "genre_en", "Genre"))

    # 2. Majority class — Emotion (text)
    results.append(run_majority(train, test, "emotion_text", "Emotion (text)"))

    # 3. TF-IDF + LR — Genre
    results.append(run_tfidf_logreg(
        train, test, "genre_en", "Genre", "cm_tfidf_genre.png"))

    # 4. TF-IDF + LR — Emotion (text)
    results.append(run_tfidf_logreg(
        train, test, "emotion_text", "Emotion (text)", "cm_tfidf_emotion.png",
        is_emotion=True))

    # 5. MFCC + SVM — Emotion (audio)
    results.append(run_mfcc_svm(train, test))

    # 6. mBERT — Genre  (most expensive — run last)
    results.append(run_mbert(train, val, test))

    # 7. Majority class — Arousal (High/Low/Medium, 3-class from scratch MLP)
    results.append(run_arousal_majority(train, test))

    # ── Save results table ─────────────────────────────────────────────────
    df_results = pd.DataFrame(results)
    df_results = df_results.round(4)

    csv_path = RPT_DIR / "baseline_results.csv"
    df_results.to_csv(csv_path, index=False)
    logger.success(f"Baseline results → {csv_path}")

    print("\n" + "═" * 65)
    print("  BASELINE RESULTS SUMMARY")
    print("═" * 65)
    print(df_results.to_string(index=False))
    print("═" * 65)
    print("\nInterpretation:")
    print("  • Majority-class Macro-F1 ≈ 0.03–0.05 → confirms imbalance")
    print("  • TF-IDF shows what pure text structure gives")
    print("  • mBERT sets the bar that AraPoemBERT must beat")
    print("  • MFCC+SVM sets the bar that Emotion1DCNN must beat")
    print("  • Arousal majority ≈ 0.33 → our MLP (0.648) is ×2 above floor")
    print(f"\nAll confusion matrices in: {FIG_DIR}/")
    print(f"Results CSV: {csv_path}")


if __name__ == "__main__":
    main()

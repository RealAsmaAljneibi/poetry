"""
src/evaluation/metrics.py

Nabat-AI Metrics v4 — unified metrics module.

Implements:
  • Soft-CER — research-informed weighted Levenshtein diagnostic for Gulf Arabic ASR
  • standard_cer/wer   — standard baseline metrics
  • Emotion partial credit — 5-tier scoring (1.0/0.65/0.45/0.30/0.20)
  • Graded nDCG        — 4-level relevance (3/2/1/0) for poetry retrieval
  • Imagery Coherence@K — shared imagery tag fraction in top-K
  • Query Robustness   — Recall@K stability under dialectal spelling variants

Soft-CER is not claimed as a novel metric. It follows the same design direction used in
Arabic ASR evaluation literature where normalization and mapping reduce false penalties
for non-standard orthography, and it is informed by:
  • MGB-3 / MR-WER style Arabic ASR normalization practice
  • CODA / CAPHI work on structured dialect orthography and phonology
  • phonologically weighted / learned edit-distance literature
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Callable

import numpy as np

# ── Arabic preprocessing constants ──────────────────────────────────────────

# Tashkeel (diacritics) — strip before comparison
_TASHKEEL = re.compile(
    r"[\u064B\u064C\u064D\u064E\u064F\u0650\u0651\u0652\u0653\u0654\u0655]"
)

# Punctuation + whitespace to normalise (Arabic + Latin)
_PUNCTUATION = re.compile(r"[،؟؛\.,\?!;:\-\(\)\"'\u2026]")

# Standard normalization shared by WER/CER and Soft-CER.
_HAMZA_MAP = str.maketrans(
    {
        "\u0623": "\u0627",  # أ → ا
        "\u0625": "\u0627",  # إ → ا
        "\u0622": "\u0627",  # آ → ا
        "\u0621": "\u0627",  # ء → ا
    }
)

# Additional normalization used only by Soft-CER.
_SOFT_ONLY_MAP = str.maketrans(
    {
        "\u0649": "\u064A",  # ى → ي
        "\u0629": "\u0647",  # ة → ه
        "\u0624": "\u0648",  # ؤ → و
    }
)

# Multi-char → single canonical char (applied before char-level comparison)
# تش (kashkasha) → چ  |  تس (kasksaka) → پ (reused as placeholder)
_MULTICHAR_SUBS: list[tuple[str, str]] = [
    ("\u062A\u0634", "\u0686"),  # تش → چ
    ("\u062A\u0633", "\u067E"),  # تس → پ  (Najdi kasksaka placeholder)
]


def _strip_poetry_tail(text: str) -> str:
    """Remove a single optional line-final elongation letter when present."""
    text = text.strip()
    if len(text) < 3:
        return text
    if text[-1] in {"ا", "ي", "و"}:
        return text[:-1]
    return text


def _preprocess(text: str) -> str:
    """Standard Arabic normalization used by strict WER/CER."""
    text = _TASHKEEL.sub("", text)
    text = _PUNCTUATION.sub(" ", text)
    text = text.translate(_HAMZA_MAP)
    for old, new in _MULTICHAR_SUBS:
        text = text.replace(old, new)
    return " ".join(text.split())


def _preprocess_soft(text: str) -> str:
    """Soft-CER normalization with extra orthographic and poetry-tail handling."""
    text = _preprocess(text)
    text = text.translate(_SOFT_ONLY_MAP)
    return _strip_poetry_tail(text)


# ── Soft-CER: research-informed default cost table ───────────────────────────

def _build_cost_table() -> dict[tuple[str, str], float]:
    """
    Bidirectional substitution cost table for Gulf-Arabic-weighted CER.

    Orthographic seat variants (أ/إ/آ/ى/ة/ؤ) are normalized in preprocessing rather
    than handled as soft substitutions. The remaining table is a defensible prior, not
    a learned confusion matrix. It is kept intentionally conservative and closer to the
    revised methodology review.
    """
    pairs: list[tuple[str, str, float]] = [
        # Velar/uvular confusions
        ("\u062E", "\u063A", 0.20),   # خ ↔ غ
        ("\u063A", "\u0642", 0.20),   # غ ↔ ق  (Kuwaiti feature)
        ("\u0642", "\u0621", 0.25),   # ق ↔ ء  (urban Gulf qaf→glottal)
        ("\u0642", "\u06AF", 0.12),   # ق ↔ گ  (Bedouin/rural qaf→g)

        # Emphatic pairs
        ("\u062A", "\u0637", 0.15),   # ت ↔ ط  (plain/emphatic alveolar)
        ("\u0636", "\u0638", 0.15),   # ض ↔ ظ  (emphatic stop/fricative merger)
        ("\u0636", "\u0630", 0.50),   # ض ↔ ذ  (keep conservative unless learned)

        # Interdental/sibilant mergers
        ("\u062B", "\u0633", 0.20),   # ث ↔ س  (Gulf merger)
        ("\u0630", "\u0632", 0.20),   # ذ ↔ ز  (voiced interdental/sibilant)
        ("\u062B", "\u062A", 0.20),   # ث ↔ ت  (fronting in Gulf varieties)
        ("\u0633", "\u0634", 0.40),   # س ↔ ش  (keep context-sensitive and conservative)
        ("\u0635", "\u0633", 0.20),   # ص ↔ س  (de-emphasis)
        ("\u0635", "\u0632", 0.60),   # ص ↔ ز  (avoid over-credit)

        # Palatal/affricate shifts
        ("\u062C", "\u064A", 0.10),   # ج ↔ ي
        ("\u0643", "\u0686", 0.12),   # ك ↔ چ
        ("\u0643", "\u067E", 0.20),   # ك ↔ پ  (kasksaka placeholder)
        ("\u0643", "\u062C", 0.15),   # ك ↔ ج  (feminine 2nd person: لك↔لج)

        # Nasal/contextual
        ("\u0646", "\u0645", 0.30),   # ن ↔ م  (nasal assimilation before labials)
    ]

    table: dict[tuple[str, str], float] = {}
    for a, b, cost in pairs:
        table[(a, b)] = cost
        table[(b, a)] = cost  # bidirectional
    return table


SOFT_CER_COST_TABLE: dict[tuple[str, str], float] = _build_cost_table()


# ── Weighted Levenshtein ─────────────────────────────────────────────────────

def _weighted_levenshtein(
    s1: str,
    s2: str,
    cost_table: dict[tuple[str, str], float],
) -> float:
    """
    Character-level weighted Levenshtein.
    Insertion / deletion cost = 1.0.
    Substitution cost = cost_table.get((c1, c2), 1.0).
    """
    m, n = len(s1), len(s2)
    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = float(i)
    for j in range(n + 1):
        dp[0][j] = float(j)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            c1, c2 = s1[i - 1], s2[j - 1]
            if c1 == c2:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                sub_cost = cost_table.get((c1, c2), 1.0)
                dp[i][j] = min(
                    dp[i - 1][j] + 1.0,           # deletion
                    dp[i][j - 1] + 1.0,           # insertion
                    dp[i - 1][j - 1] + sub_cost,  # substitution
                )
    return dp[m][n]


# ── Public ASR metrics ────────────────────────────────────────────────────────

def soft_cer(
    hypothesis: str,
    reference: str,
    cost_table: dict[tuple[str, str], float] | None = None,
) -> float:
    """
    Research-informed weighted character error rate for Gulf Arabic ASR.

    This is an exploratory diagnostic, not a claimed novel metric. It combines:
    1. Arabic orthographic normalization,
    2. a conservative Gulf-Arabic substitution prior,
    3. a light poetry-tail normalization for line-final elongation.

    Normalized by reference length after preprocessing.
    Returns float ≥ 0 (values > 1.0 possible with many insertions).
    """
    if cost_table is None:
        cost_table = SOFT_CER_COST_TABLE
    hyp_chars = _preprocess_soft(hypothesis).replace(" ", "")
    ref_chars  = _preprocess_soft(reference).replace(" ", "")
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    dist = _weighted_levenshtein(hyp_chars, ref_chars, cost_table)
    return dist / len(ref_chars)


def standard_cer(hypothesis: str, reference: str) -> float:
    """Standard CER (all substitutions cost 1.0, no phonemic weighting)."""
    return soft_cer(hypothesis, reference, cost_table={})


def _word_alignment_counts(
    hypothesis: str, reference: str,
) -> tuple[int, int, int, int]:
    """
    Word-level DP alignment returning (hits, substitutions, deletions, insertions).
    Uses _preprocess for normalization.
    """
    hyp = _preprocess(hypothesis).split()
    ref = _preprocess(reference).split()
    if not ref and not hyp:
        return (0, 0, 0, 0)
    if not ref:
        return (0, 0, 0, len(hyp))
    if not hyp:
        return (0, 0, len(ref), 0)
    m, n = len(hyp), len(ref)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if hyp[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    # Backtrace to get S, D, I, C
    i, j = m, n
    subs = dels = ins = hits = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and hyp[i - 1] == ref[j - 1]:
            hits += 1
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            subs += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ins += 1
            i -= 1
        else:
            dels += 1
            j -= 1
    return (hits, subs, dels, ins)


def standard_wer(hypothesis: str, reference: str) -> float:
    """Standard WER: word-level edit distance normalised by reference word count."""
    ref_words = _preprocess(reference).split()
    if not ref_words:
        hyp_words = _preprocess(hypothesis).split()
        return 0.0 if not hyp_words else 1.0
    hits, subs, dels, ins = _word_alignment_counts(hypothesis, reference)
    return (subs + dels + ins) / len(ref_words)


# ── Emotion distance matrix ──────────────────────────────────────────────────

# Short key → canonical label (must match labels.py EMOTION_CLASSES exactly)
_E: dict[str, str] = {
    "longing":       "Longing (Shawq)",
    "love":          "Delicate Love (Hub Raqeeq)",
    "sorrow":        "Sorrow (Huzn)",
    "pride":         "Pride (Fakhr)",
    "admiration":    "Admiration (I'jab)",
    "contemplation": "Contemplation (Ta'ammul)",
    "disappointment":"Disappointment (Khayba)",
    "defiance":      "Defiance (Tahaddi)",
    "hope":          "Hope (Amal)",
    "compassion":    "Compassion (Hanaan)",
    "humor":         "Humor (Turfah)",
    "neutral":       "Neutral / Descriptive (Wasfi)",
}
_E_REV: dict[str, str] = {v: k for k, v in _E.items()}

# Short-form prefix → canonical (for data that stores e.g. "Admiration" not "Admiration (I'jab)")
_E_SHORT: dict[str, str] = {
    canonical.split("(")[0].strip().lower(): canonical
    for canonical in _E.values()
}
# Also add "neutral / descriptive" → neutral
_E_SHORT["neutral / descriptive"] = "Neutral / Descriptive (Wasfi)"


def normalize_emotion(label) -> str:
    """
    Normalise an emotion label to its canonical form (matching EMOTION_CLASSES).
    Accepts: full canonical, short prefix ("Admiration"), or mixed-case variants.
    Handles NaN/None/float gracefully — returns empty string.
    Returns the label unchanged if no match found.
    """
    if label is None or (isinstance(label, float)):
        return ""
    if not label:
        return ""
    stripped = str(label).strip()
    # Already canonical
    if stripped in _E_REV:
        return stripped
    # Exact short-key match
    key = stripped.lower()
    if key in _E_SHORT:
        return _E_SHORT[key]
    # Prefix match (e.g. "Delicate Love" matches "Delicate Love (Hub Raqeeq)")
    for canonical in _E.values():
        if canonical.lower().startswith(key):
            return canonical
    return stripped


# Genre canonical normalization (handles case variants like "Ghazal (Delicate love)")
_GENRE_CANONICAL: dict[str, str] = {
    "Ghazal (Delicate Love)":                    "Ghazal (Delicate Love)",
    "Fakhr (Pride & Honor)":                     "Fakhr (Pride & Honor)",
    "Madih (Praise)":                            "Madih (Praise)",
    "Ritha (Elegy & Lament)":                    "Ritha (Elegy & Lament)",
    "Hikma (Wisdom، Philosophical & Reflection)":"Hikma (Wisdom، Philosophical & Reflection)",
    "Wataniyya (Patriotic & National)":          "Wataniyya (Patriotic & National)",
    "Hija (Satire & Social Critique)":           "Hija (Satire & Social Critique)",
    "Badawa (Bedouin Life & Desert Heritage)":   "Badawa (Bedouin Life & Desert Heritage)",
    "I'tithar (Delicate Apology)":               "I'tithar (Delicate Apology)",
    "Tareef (Humorous)":                         "Tareef (Humorous)",
    "Shajan (Sorrow / Regret)":                  "Shajan (Sorrow / Regret)",
}


def normalize_genre(label) -> str:
    """
    Normalise a genre label to its canonical form (case-insensitive prefix match).
    Handles NaN/None/float gracefully — returns empty string.
    """
    if label is None or isinstance(label, float):
        return ""
    if not label:
        return ""
    stripped = str(label).strip()
    if stripped in _GENRE_CANONICAL:
        return stripped
    low = stripped.lower()
    for canonical in _GENRE_CANONICAL:
        if canonical.lower() == low:
            return canonical
    # Prefix match on the main genre word before the parenthesis
    prefix = low.split("(")[0].strip()
    for canonical in _GENRE_CANONICAL:
        if canonical.lower().startswith(prefix):
            return canonical
    return stripped

# Adjacency edges for BFS distance calculation
# (emotion_a, emotion_b) at distance = 1
_EMOTION_EDGES: list[tuple[str, str]] = [
    # Cluster A — Melancholic/Longing
    ("longing",       "sorrow"),
    ("longing",       "love"),
    ("sorrow",        "disappointment"),
    ("compassion",    "longing"),
    ("compassion",    "love"),
    # Cluster B — Assertive/Powerful
    ("pride",         "defiance"),
    ("pride",         "hope"),
    ("defiance",      "hope"),
    # Cluster C — Gentle/Tender
    ("admiration",    "love"),
    ("humor",         "admiration"),
    # Cluster D — Reflective/Neutral
    ("contemplation", "neutral"),
    # Cross-cluster bridges
    ("sorrow",        "contemplation"),
    ("hope",          "contemplation"),
    ("admiration",    "hope"),
    ("disappointment","compassion"),
]


def _build_emotion_distance_matrix() -> dict[tuple[str, str], int]:
    """BFS-based pairwise emotion distance in the adjacency graph."""
    graph: dict[str, list[str]] = defaultdict(list)
    for a, b in _EMOTION_EDGES:
        graph[a].append(b)
        graph[b].append(a)

    emotions = list(_E.keys())
    dist_matrix: dict[tuple[str, str], int] = {}

    for src in emotions:
        visited: dict[str, int] = {src: 0}
        queue = [src]
        while queue:
            curr = queue.pop(0)
            for nb in graph[curr]:
                if nb not in visited:
                    visited[nb] = visited[curr] + 1
                    queue.append(nb)
        for tgt in emotions:
            d = visited.get(tgt, 999)
            dist_matrix[(src, tgt)] = d
            dist_matrix[(tgt, src)] = d

    return dist_matrix


EMOTION_DISTANCE: dict[tuple[str, str], int] = _build_emotion_distance_matrix()


def emotion_distance(emotion_a: str, emotion_b: str) -> int:
    """
    Distance between two emotion labels (canonical strings from labels.py).
    Accepts short-form labels ("Admiration") and normalises them automatically.
    Returns 0 for identical, 1 for same-cluster adjacent, ≥2 for further.
    Returns 999 for unknown labels.
    """
    a = normalize_emotion(emotion_a)
    b = normalize_emotion(emotion_b)
    if a == b:
        return 0
    ka = _E_REV.get(a, a)
    kb = _E_REV.get(b, b)
    return EMOTION_DISTANCE.get((ka, kb), 999)


# ── Genre → expected emotions mapping ────────────────────────────────────────

# Primary + secondary expected emotions per genre (used for Tier 3 partial credit)
GENRE_EXPECTED_EMOTIONS: dict[str, set[str]] = {
    "Ghazal (Delicate Love)": {
        "Delicate Love (Hub Raqeeq)", "Longing (Shawq)", "Admiration (I'jab)",
        "Compassion (Hanaan)", "Sorrow (Huzn)",
    },
    "Fakhr (Pride & Honor)": {
        "Pride (Fakhr)", "Defiance (Tahaddi)", "Hope (Amal)", "Admiration (I'jab)",
    },
    "Madih (Praise)": {
        "Admiration (I'jab)", "Hope (Amal)", "Pride (Fakhr)",
        "Delicate Love (Hub Raqeeq)",
    },
    "Ritha (Elegy & Lament)": {
        "Sorrow (Huzn)", "Longing (Shawq)", "Disappointment (Khayba)",
        "Compassion (Hanaan)",
    },
    "Hikma (Wisdom، Philosophical & Reflection)": {
        "Contemplation (Ta'ammul)", "Neutral / Descriptive (Wasfi)",
        "Hope (Amal)", "Admiration (I'jab)",
    },
    "Wataniyya (Patriotic & National)": {
        "Pride (Fakhr)", "Hope (Amal)", "Defiance (Tahaddi)", "Admiration (I'jab)",
    },
    "Hija (Satire & Social Critique)": {
        "Defiance (Tahaddi)", "Humor (Turfah)", "Disappointment (Khayba)",
        "Admiration (I'jab)",
    },
    "Badawa (Bedouin Life & Desert Heritage)": {
        "Longing (Shawq)", "Contemplation (Ta'ammul)",
        "Neutral / Descriptive (Wasfi)", "Sorrow (Huzn)",
    },
    "I'tithar (Delicate Apology)": {
        "Compassion (Hanaan)", "Disappointment (Khayba)",
        "Delicate Love (Hub Raqeeq)", "Sorrow (Huzn)",
    },
    "Tareef (Humorous)": {
        "Humor (Turfah)", "Admiration (I'jab)",
        "Neutral / Descriptive (Wasfi)", "Defiance (Tahaddi)",
    },
    "Shajan (Sorrow / Regret)": {
        "Sorrow (Huzn)", "Disappointment (Khayba)", "Longing (Shawq)",
        "Compassion (Hanaan)",
    },
}

# All canonical emotion labels (for Tier 5 validation)
_ALL_EMOTIONS: frozenset[str] = frozenset(_E.values())


def emotion_partial_credit(
    pred:      str,
    audio_ref: str,
    text_ref:  str,
    genre:     str,
) -> float:
    """
    5-tier partial-credit emotion score.

    Tier  Score  Criterion
    ───── ─────  ─────────────────────────────────────────────────────
      1   1.00   Exact match: pred == audio_ref or pred == text_ref
      2   0.65   Cluster-adjacent: emotion_distance == 1 to either ref
      3   0.45   Genre-plausible: pred in GENRE_EXPECTED_EMOTIONS[genre]
      4   0.30   One cluster away: emotion_distance == 2 to either ref
      5   0.20   Any valid emotion label (partial recognition)
      —   0.00   Unknown / empty

    Parameters
    ----------
    pred      : predicted emotion label (canonical string)
    audio_ref : ground-truth emotion from audio annotation
    text_ref  : ground-truth emotion from text annotation
    genre     : genre of the poem (for Tier 3 plausibility check)
    """
    # Normalise all labels to canonical form (handles NaN, short labels, case)
    pred_c      = normalize_emotion(pred)
    audio_ref_c = normalize_emotion(audio_ref)
    text_ref_c  = normalize_emotion(text_ref)
    genre_c     = normalize_genre(genre)

    if not pred_c:
        return 0.0

    # Tier 1 — exact match
    if pred_c == audio_ref_c or pred_c == text_ref_c:
        return 1.0

    # Tier 2 — cluster-adjacent
    d_audio = emotion_distance(pred_c, audio_ref_c)
    d_text  = emotion_distance(pred_c, text_ref_c)
    if d_audio == 1 or d_text == 1:
        return 0.65

    # Tier 3 — genre-plausible
    if pred_c in GENRE_EXPECTED_EMOTIONS.get(genre_c, set()):
        return 0.45

    # Tier 4 — one cluster away
    if d_audio == 2 or d_text == 2:
        return 0.30

    # Tier 5 — any valid emotion
    if pred_c in _ALL_EMOTIONS:
        return 0.20

    return 0.0


def mean_emotion_partial_credit(
    preds:      list[str],
    audio_refs: list[str],
    text_refs:  list[str],
    genres:     list[str],
) -> float:
    """Average emotion_partial_credit over a list of predictions."""
    if not preds:
        return 0.0
    scores = [
        emotion_partial_credit(p, a, t, g)
        for p, a, t, g in zip(preds, audio_refs, text_refs, genres)
    ]
    return float(np.mean(scores))


# ── Graded nDCG for retrieval ─────────────────────────────────────────────────

def graded_relevance(
    result_genre:   str,
    result_emotion: str,
    query_genre:    str,
    query_emotion:  str,
) -> int:
    """
    4-level relevance for graded nDCG.

    Score  Criterion
    ─────  ──────────────────────────────────────────────────
      3    Same genre AND same emotion
      2    Same genre AND emotion_distance == 1
      1    (Same genre, any emotion) OR emotion_distance ≤ 1
      0    Otherwise
    """
    same_genre = normalize_genre(result_genre) == normalize_genre(query_genre)
    e_dist     = emotion_distance(result_emotion, query_emotion)

    if same_genre and e_dist == 0:
        return 3
    if same_genre and e_dist == 1:
        return 2
    if same_genre or e_dist <= 1:
        return 1
    return 0


def graded_ndcg_at_k(
    results:       list[dict],
    query_genre:   str,
    query_emotion: str,
    k:             int = 10,
    genre_key:     str = "genre_en",
    emotion_key:   str = "emotion_text",
) -> float:
    """
    Graded nDCG@K using 4-level relevance (0–3).
    DCG  = Σ_{i=1}^{K}  (2^rel_i - 1) / log2(i+1)
    IDCG = DCG of the ideal re-ranking of the same result list.
    """
    topk = results[:k]

    dcg = 0.0
    rels: list[int] = []
    for i, r in enumerate(topk):
        rel = graded_relevance(
            r.get(genre_key, ""), r.get(emotion_key, ""),
            query_genre, query_emotion,
        )
        rels.append(rel)
        dcg += (2 ** rel - 1) / math.log2(i + 2)

    # IDCG: ideal ordering of the same relevance scores
    ideal_rels = sorted(rels, reverse=True)
    idcg = sum(
        (2 ** rel - 1) / math.log2(i + 2)
        for i, rel in enumerate(ideal_rels)
    )
    return dcg / idcg if idcg > 0 else 0.0


# ── Imagery coherence ─────────────────────────────────────────────────────────

def imagery_coherence_at_k(
    results:            list[dict],
    query_imagery_tags: list[str],
    k:                  int = 10,
    imagery_key:        str = "imagery_tags_en",
) -> float:
    """
    Fraction of top-K results that share ≥1 imagery tag with the query.
    Returns NaN if the query has no imagery tags (excluded from averages).
    """
    query_tags = {t.strip().lower() for t in query_imagery_tags if t.strip()}
    if not query_tags:
        return float("nan")

    topk = results[:k]
    if not topk:
        return 0.0

    hits = 0
    for r in topk:
        raw = r.get(imagery_key, [])
        if isinstance(raw, str):
            raw = [t.strip() for t in raw.split(",")]
        result_tags = {t.strip().lower() for t in raw if t.strip()}
        if query_tags & result_tags:
            hits += 1

    return hits / len(topk)


# ── Query robustness ──────────────────────────────────────────────────────────

# Known Gulf dialectal substitution rules (original → variant)
DIALECT_SUBSTITUTION_RULES: list[tuple[str, str]] = [
    ("\u062C", "\u064A"),  # ج → ي  (Gulf: شجرة→شيرة)
    ("\u0643", "\u0686"),  # ك → چ  (kashkasha: كتاب→چتاب)
    ("\u062E", "\u063A"),  # خ → غ  (voiceless/voiced velar)
    ("\u062B", "\u0633"),  # ث → س  (Gulf interdental merger)
    ("\u0630", "\u0632"),  # ذ → ز  (voiced interdental merger)
    ("\u0642", "\u0621"),  # ق → ء  (urban Gulf glottal stop)
]


def generate_dialect_variants(text: str, n_variants: int = 3) -> list[str]:
    """
    Generate dialectal spelling variants via substitution rules.
    Returns up to n_variants non-duplicate distinct variants.
    """
    variants: list[str] = []
    for old, new in DIALECT_SUBSTITUTION_RULES:
        if old in text:
            variant = text.replace(old, new)
            if variant != text and variant not in variants:
                variants.append(variant)
        if len(variants) >= n_variants:
            break
    return variants


def query_robustness_score(
    original_query: str,
    search_fn:      Callable[[str, int], list[dict]],
    record_id_fn:   Callable[[dict], str],
    k:              int = 10,
    n_variants:     int = 3,
) -> float:
    """
    Recall@K stability under dialectal spelling variants.

    For each dialectal variant of original_query, compute the fraction of
    the original top-K results still retrieved (robust recall).

    Returns: mean robust recall across variants (target: > 0.85).
    Returns NaN when no variants are generatable.

    Parameters
    ----------
    original_query : query string in standard Arabic spelling
    search_fn      : callable(query: str, top_k: int) → list of result dicts
    record_id_fn   : callable(result_dict) → unique string ID
    k              : retrieval depth
    n_variants     : number of dialectal variants to generate
    """
    original_results = search_fn(original_query, k)
    original_ids = {record_id_fn(r) for r in original_results}
    if not original_ids:
        return float("nan")

    variants = generate_dialect_variants(original_query, n_variants)
    if not variants:
        return float("nan")

    recalls: list[float] = []
    for variant in variants:
        variant_results = search_fn(variant, k)
        variant_ids = {record_id_fn(r) for r in variant_results}
        recalls.append(len(original_ids & variant_ids) / len(original_ids))

    return float(np.mean(recalls))


# ── ASR complement metrics (MER, WIL) ───────────────────────────────────────

def match_error_rate(hypothesis: str, reference: str) -> float:
    """
    Match Error Rate: MER = (S + D + I) / (S + D + I + C).
    Unlike WER (normalized by ref length), MER normalizes by total alignment length.
    """
    hits, subs, dels, ins = _word_alignment_counts(hypothesis, reference)
    total = subs + dels + ins + hits
    if total == 0:
        return 0.0
    return (subs + dels + ins) / total


def word_information_lost(hypothesis: str, reference: str) -> float:
    """
    Word Information Lost: WIL = 1 - C² / (N_ref × N_hyp).
    Measures fraction of word information lost between ref and hyp.
    """
    ref_words = _preprocess(reference).split()
    hyp_words = _preprocess(hypothesis).split()
    n_ref = len(ref_words)
    n_hyp = len(hyp_words)
    if n_ref == 0 and n_hyp == 0:
        return 0.0
    if n_ref == 0 or n_hyp == 0:
        return 1.0
    hits, _, _, _ = _word_alignment_counts(hypothesis, reference)
    return 1.0 - (hits * hits) / (n_ref * n_hyp)


# ── Classification helpers ───────────────────────────────────────────────────

def balanced_accuracy(y_true: list[int], y_pred: list[int]) -> float:
    """Balanced accuracy (macro-averaged recall). Wraps sklearn."""
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y_true, y_pred))


def log_loss_safe(
    y_true_ids: list[int],
    prob_matrix: list[list[float]],
    labels: list[int] | None = None,
) -> float:
    """
    Multi-class log loss with epsilon clamping and explicit labels param.
    Prevents crash when a class has 0 samples in the batch.
    """
    from sklearn.metrics import log_loss as sk_log_loss
    arr = np.array(prob_matrix, dtype=np.float64)
    eps = 1e-15
    arr = np.clip(arr, eps, 1.0 - eps)
    # Re-normalize rows after clipping
    arr = arr / arr.sum(axis=1, keepdims=True)
    return float(sk_log_loss(y_true_ids, arr, labels=labels))


def brier_score_multi(y_true_ids: list[int], prob_matrix: list[list[float]]) -> float:
    """
    Multi-class Brier score: mean over samples of sum_k (p_k - 1[k==true])^2.
    Lower is better; 0 = perfect calibration.
    """
    arr = np.array(prob_matrix, dtype=np.float64)
    n_samples, n_classes = arr.shape
    if n_samples == 0:
        return 0.0
    total = 0.0
    for i, true_id in enumerate(y_true_ids):
        for k in range(n_classes):
            indicator = 1.0 if k == true_id else 0.0
            total += (arr[i, k] - indicator) ** 2
    return total / n_samples


# ── Emotion ranking metrics ──────────────────────────────────────────────────

def top_k_accuracy(pred_probs: list[float], true_id: int, k: int = 3) -> float:
    """Returns 1.0 if true_id is in the top-k predictions, else 0.0."""
    top_k_ids = sorted(range(len(pred_probs)), key=lambda i: pred_probs[i], reverse=True)[:k]
    return 1.0 if true_id in top_k_ids else 0.0


def recall_at_k_multi(
    pred_probs: list[float],
    true_ids_set: set[int],
    k: int = 3,
) -> float:
    """Fraction of true label IDs found in top-k predictions (for multi-ref)."""
    if not true_ids_set:
        return 0.0
    top_k_ids = set(
        sorted(range(len(pred_probs)), key=lambda i: pred_probs[i], reverse=True)[:k]
    )
    return len(true_ids_set & top_k_ids) / len(true_ids_set)


def emotion_ndcg_at_3(
    pred_probs: list[float],
    true_label: str,
    audio_ref: str,
    genre: str,
    labels: list[str],
) -> float:
    """
    nDCG@3 for emotion using partial-credit as relevance gain.
    Ranks emotions by predicted probability, computes DCG using
    emotion_partial_credit as the relevance score for each position.
    """
    k = 3
    top_k_indices = sorted(
        range(len(pred_probs)), key=lambda i: pred_probs[i], reverse=True
    )[:k]

    # DCG with partial-credit relevance
    dcg = 0.0
    rels: list[float] = []
    for rank, idx in enumerate(top_k_indices):
        pred_label = labels[idx] if idx < len(labels) else ""
        rel = emotion_partial_credit(pred_label, audio_ref, true_label, genre)
        rels.append(rel)
        dcg += rel / math.log2(rank + 2)

    # IDCG: best possible ordering of the same relevance values
    ideal_rels = sorted(rels, reverse=True)
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal_rels))
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def lrap_score(
    y_true_binary: list[list[int]],
    y_score: list[list[float]],
) -> float:
    """Label Ranking Average Precision. Wraps sklearn."""
    from sklearn.metrics import label_ranking_average_precision_score
    return float(label_ranking_average_precision_score(
        np.array(y_true_binary), np.array(y_score),
    ))


# ── Calibration & agreement ─────────────────────────────────────────────────

def expected_calibration_error(
    prob_rows: list[list[float]],
    true_ids: list[int],
    n_bins: int = 10,
) -> float:
    """
    Expected Calibration Error (ECE).
    Bins predictions by confidence, measures gap between confidence and accuracy.
    """
    if not prob_rows:
        return 0.0
    preds = [int(np.argmax(prob)) for prob in prob_rows]
    confidences = np.array([float(np.max(prob)) for prob in prob_rows])
    correctness = np.array(
        [int(pred == true) for pred, true in zip(preds, true_ids)], dtype=float
    )
    ece = 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() > 0:
            ece += mask.sum() / len(true_ids) * abs(
                correctness[mask].mean() - confidences[mask].mean()
            )
    return float(ece)


def inter_annotator_kappa(labels_a: list, labels_b: list) -> float:
    """Cohen's kappa between two annotator label sequences. Wraps sklearn."""
    from sklearn.metrics import cohen_kappa_score
    return float(cohen_kappa_score(labels_a, labels_b))


def krippendorff_alpha_nominal(labels_a: list, labels_b: list) -> float:
    """
    Krippendorff's alpha for nominal data (two annotators).
    Coincidence-matrix implementation — no extra dependency.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("Annotator sequences must have equal length")
    n = len(labels_a)
    if n == 0:
        return 1.0
    # Build coincidence matrix
    categories = sorted(set(labels_a) | set(labels_b))
    cat2idx = {c: i for i, c in enumerate(categories)}
    nc = len(categories)
    coincidence = np.zeros((nc, nc), dtype=np.float64)
    for a, b in zip(labels_a, labels_b):
        i, j = cat2idx[a], cat2idx[b]
        coincidence[i][j] += 1.0
        coincidence[j][i] += 1.0
    # Marginals
    n_c = coincidence.sum(axis=1)
    total = n_c.sum()
    if total == 0:
        return 1.0
    # Observed disagreement
    d_o = 0.0
    for i in range(nc):
        for j in range(nc):
            if i != j:
                d_o += coincidence[i][j]
    d_o /= total
    # Expected disagreement
    d_e = 0.0
    for i in range(nc):
        for j in range(nc):
            if i != j:
                d_e += n_c[i] * n_c[j]
    d_e /= total * (total - 1) if total > 1 else 1.0
    if d_e == 0.0:
        return 1.0
    return 1.0 - d_o / d_e


# ── Grouped bootstrap ───────────────────────────────────────────────────────

def bootstrap_grouped_ci(
    metric_per_group: dict[str, float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Bootstrap CI by resampling group keys (e.g. poem IDs).
    Returns (observed_mean, lower, upper).
    """
    rng = np.random.default_rng(seed)
    keys = list(metric_per_group.keys())
    values = np.array([metric_per_group[k] for k in keys])
    n_groups = len(keys)
    if n_groups == 0:
        return (0.0, 0.0, 0.0)
    observed = float(values.mean())
    stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n_groups, size=n_groups)
        stats[i] = values[idx].mean()
    lower = float(np.percentile(stats, 100 * alpha / 2))
    upper = float(np.percentile(stats, 100 * (1 - alpha / 2)))
    return (observed, lower, upper)

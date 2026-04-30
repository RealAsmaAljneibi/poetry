from loguru import logger

EMOTION_CLASSES = [
    "Longing (Shawq)",
    "Delicate Love (Hub Raqeeq)",
    "Sorrow (Huzn)",
    "Pride (Fakhr)",
    "Admiration (I'jab)",
    "Contemplation (Ta'ammul)",
    "Disappointment (Khayba)",
    "Defiance (Tahaddi)",
    "Hope (Amal)",
    "Compassion (Hanaan)",
    "Humor (Turfah)",
    "Neutral / Descriptive (Wasfi)",
]

# 8-genre taxonomy.
# Full mapping of all 11 raw genres → 8 merged classes:
#   Ghazal (Delicate love)                    → Ghazal       [kept]
#   Shajan (Sorrow / Regret)                  → Shajan       [kept]
#   Fakhr (Pride & Honor)                     → Fakhr        [kept]
#   Hikma (Wisdom، Philosophical & Reflection) → Hikma        [kept]
#   Badawa (Bedouin Life & Desert Heritage)   → Badawa       [kept]
#   Wataniyya (Patriotic & National)          → Wataniyya    [kept]
#   Ritha (Elegy & Lament)                    → Ritha        [kept]
#   Hija (Satire & Social Critique)           → Hija         [kept]
#   Madih (Praise)           → Fakhr   (praise/honour family)
#   I'tithar (Delicate Apology) → Ghazal (emotion-profile cosine=0.934; love-register, tender address to beloved)
#   Tareef (Humorous)           → Hija   (wit/satire share the socially-observant register)
# Ordered by total corpus size descending.
GENRE_CLASSES = [
    "Ghazal (Delicate love)",
    "Shajan (Sorrow / Regret)",
    "Fakhr (Pride & Honor)",
    "Hikma (Wisdom، Philosophical & Reflection)",
    "Badawa (Bedouin Life & Desert Heritage)",
    "Wataniyya (Patriotic & National)",
    "Ritha (Elegy & Lament)",
    "Hija (Satire & Social Critique)",
]

# Maps removed genre labels to their surviving merge targets.
# encode_genre() applies this before label lookup.
GENRE_MERGE_MAP: dict[str, str] = {
    "Madih (Praise)": "Fakhr (Pride & Honor)",
    "I'tithar (Delicate Apology)": "Ghazal (Delicate love)",
    "Tareef (Humorous)": "Hija (Satire & Social Critique)",
}

EMOTION2ID = {label: idx for idx, label in enumerate(EMOTION_CLASSES)}
ID2EMOTION = {idx: label for label, idx in EMOTION2ID.items()}

GENRE2ID = {label: idx for idx, label in enumerate(GENRE_CLASSES)}
ID2GENRE = {idx: label for label, idx in GENRE2ID.items()}


def encode_emotion(emotion_str: str) -> int:
    """Converts an emotion string to its integer ID via partial matching."""
    if not emotion_str or not str(emotion_str).strip():
        logger.error("Unknown Emotion label: <empty>")
        return -1
    emotion_str = emotion_str.strip().lower()
    for label, idx in EMOTION2ID.items():
        if label.lower() == emotion_str or emotion_str.split(" ")[0] in label.lower():
            return idx
    logger.error(f"Unknown Emotion label: {emotion_str}")
    return -1


def encode_genre(genre_str: str) -> int:
    """Converts a genre string to its integer ID, applying the merge map first."""
    if not genre_str or not str(genre_str).strip():
        logger.error("Unknown Genre label: <empty>")
        return -1
    genre_str = genre_str.strip()
    # Apply merge map (exact match, then partial)
    for old, new in GENRE_MERGE_MAP.items():
        if (
            genre_str.lower() == old.lower()
            or genre_str.lower().split(" ")[0] == old.lower().split(" ")[0]
        ):
            genre_str = new
            break
    g = genre_str.lower()
    for label, idx in GENRE2ID.items():
        if label.lower() == g or g.split(" ")[0] in label.lower():
            return idx
    logger.error(f"Unknown Genre label: {genre_str!r}")
    return -1


def merge_genre_label(genre_str: str) -> str:
    """Returns the canonical (post-merge) genre string for a raw genre_en value."""
    genre_str = genre_str.strip()
    for old, new in GENRE_MERGE_MAP.items():
        if (
            genre_str.lower() == old.lower()
            or genre_str.lower().split(" ")[0] == old.lower().split(" ")[0]
        ):
            return new
    return genre_str


# ── Emotion merge profiles ─────────────────────────────────────────────────
# Maps profile_name → {short_prefix (case-insensitive): target full class name}
# Rationale for rare_merge_v1:
#   Longing (9 test clips)    → Sorrow    (semantically adjacent: yearning ≈ grief)
#   Compassion (2 test clips) → Delicate Love  (tender care ≈ gentle love)
#   Humor (2 test clips)      → Neutral   (light/non-emotive framing in Nabati)
# These three classes score near-zero F1 due to extreme rarity and penalize macro avg.
EMOTION_MERGE_PROFILES: dict[str, dict[str, str]] = {
    "none": {},
    "rare_merge_v1": {
        "Longing": "Sorrow (Huzn)",
        "Compassion": "Delicate Love (Hub Raqeeq)",
        "Humor": "Neutral / Descriptive (Wasfi)",
    },
}


def apply_emotion_merge(emotion_str: str, profile: str = "none") -> str:
    """
    Apply a merge profile to an emotion label string.
    Matches on the first word of the emotion (before parenthesis), case-insensitive.
    Returns the target class string if matched, else the original.
    """
    if profile == "none" or profile not in EMOTION_MERGE_PROFILES:
        return emotion_str
    mapping = EMOTION_MERGE_PROFILES[profile]
    short = emotion_str.strip().split("(")[0].strip().lower()
    for prefix, target in mapping.items():
        if short == prefix.lower():
            return target
    return emotion_str


def get_merged_emotion_classes(profile: str = "none") -> list[str]:
    """Return the ordered list of emotion classes after applying a merge profile."""
    if profile == "none" or profile not in EMOTION_MERGE_PROFILES:
        return list(EMOTION_CLASSES)
    mapping = EMOTION_MERGE_PROFILES[profile]
    merged_away: set[str] = set()
    for prefix in mapping:
        for cls in EMOTION_CLASSES:
            if cls.split("(")[0].strip().lower() == prefix.lower():
                merged_away.add(cls)
    return [cls for cls in EMOTION_CLASSES if cls not in merged_away]


def encode_emotion_with_profile(emotion_str: str, profile: str = "none") -> int:
    """
    Encode an emotion string to its integer ID, applying a merge profile first.
    Returns -1 if unknown after merging.
    """
    merged = apply_emotion_merge(emotion_str, profile)
    merged_classes = get_merged_emotion_classes(profile)
    merged_lower = merged.strip().lower()
    for i, label in enumerate(merged_classes):
        if label.lower() == merged_lower or merged_lower.split(" ")[0] in label.lower():
            return i
    logger.error(
        f"Unknown emotion after merge ({profile!r}): {emotion_str!r} -> {merged!r}"
    )
    return -1


# ── Genre → expected emotion sets ─────────────────────────────────────────────
# Culturally grounded mapping: which emotions are plausible for each genre.
# Estimated from corpus statistics (train split) + Nabati tradition knowledge.
# Used for:
#   1. genre-constrained decoding  (re-rank emotion outputs)
#   2. partial-credit tier T3 (genre_plausible)
#   3. genre-plausibility evaluation
GENRE_EXPECTED_EMOTIONS: dict[str, list[str]] = {
    "Ghazal (Delicate love)": [
        "Delicate Love (Hub Raqeeq)",
        "Longing (Shawq)",
        "Sorrow (Huzn)",
        "Hope (Amal)",
        "Disappointment (Khayba)",
    ],
    "Shajan (Sorrow / Regret)": [
        "Sorrow (Huzn)",
        "Longing (Shawq)",
        "Disappointment (Khayba)",
        "Contemplation (Ta'ammul)",
    ],
    "Fakhr (Pride & Honor)": [
        "Pride (Fakhr)",
        "Defiance (Tahaddi)",
        "Admiration (I'jab)",
    ],
    "Hikma (Wisdom\u060c Philosophical & Reflection)": [
        "Contemplation (Ta'ammul)",
        "Hope (Amal)",
        "Admiration (I'jab)",
        "Pride (Fakhr)",
    ],
    "Badawa (Bedouin Life & Desert Heritage)": [
        "Contemplation (Ta'ammul)",
        "Pride (Fakhr)",
        "Neutral / Descriptive (Wasfi)",
        "Admiration (I'jab)",
        "Longing (Shawq)",
    ],
    "Wataniyya (Patriotic & National)": [
        "Pride (Fakhr)",
        "Hope (Amal)",
        "Admiration (I'jab)",
        "Defiance (Tahaddi)",
    ],
    "Ritha (Elegy & Lament)": [
        "Sorrow (Huzn)",
        "Longing (Shawq)",
        "Compassion (Hanaan)",
        "Admiration (I'jab)",
    ],
    "Hija (Satire & Social Critique)": [
        "Defiance (Tahaddi)",
        "Humor (Turfah)",
        "Sorrow (Huzn)",
        "Disappointment (Khayba)",
    ],
}


# Merge-profile-aware version: returns expected emotions under a given merge profile
def get_genre_expected_emotions(genre: str, profile: str = "none") -> list[str]:
    """
    Returns the expected emotions for a genre, mapped through a merge profile.
    Duplicates are removed; order is preserved.
    """
    raw = GENRE_EXPECTED_EMOTIONS.get(genre, [])
    seen: set[str] = set()
    result: list[str] = []
    for em in raw:
        mapped = apply_emotion_merge(em, profile)
        if mapped not in seen:
            seen.add(mapped)
            result.append(mapped)
    return result


if __name__ == "__main__":
    logger.add("logs/labels.log", rotation="10 MB")
    for g in [
        "Madih (Praise)",
        "I'tithar (Delicate Apology)",
        "Tareef (Humorous)",
        "Ghazal (Delicate love)",
        "Hija (Satire & Social Critique)",
    ]:
        idx = encode_genre(g)
        logger.info(f"'{g}' → id={idx} → '{ID2GENRE.get(idx)}'")

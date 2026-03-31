"""
src/data/arousal_labels.py

Maps the 12 human-annotated emotion_audio labels to 3-level Arousal.

Why Arousal instead of 12-class emotion from audio:
  - Audio reliably encodes *delivery energy* (loud/forceful vs soft/calm),
    but NOT the fine-grained semantic emotion (that lives in the words).
  - 84.5% of clips have different text vs audio emotion labels — this is
    a cultural feature of Nabati oral poetry (ironic/proud delivery of sad
    content).  A 12-class audio model therefore approaches chance.
  - 3-class Arousal is what the vocal features actually carry:
      High  — forceful, energetic delivery (Defiance, Pride, Humor)
      Medium — moderate, expressive delivery (Love, Admiration, Longing…)
      Low   — gentle, quiet, reflective delivery (Contemplation, Neutral, Sorrow)
  - Distribution from the 3,340-clip corpus (after removing 105 null labels):
      High   1079 (33%)   Defiance 596 + Pride 382 + Humor 101
      Medium  987 (31%)   Delicate Love 299 + Admiration 198 + Disappointment 194
                           + Compassion 167 + Hope 81 + Longing 48
      Low    1169 (36%)   Neutral 592 + Sorrow 458 + Contemplation 119
    → Almost perfectly balanced: no class weighting needed.

Usage:
    from src.data.arousal_labels import (
        AROUSAL_CLASSES, AROUSAL2ID, ID2AROUSAL,
        emotion_to_arousal, encode_arousal,
    )
"""

AROUSAL_CLASSES: list[str] = ["Low", "Medium", "High"]

AROUSAL2ID: dict[str, int] = {label: i for i, label in enumerate(AROUSAL_CLASSES)}
ID2AROUSAL:  dict[int, str] = {i: label for label, i in AROUSAL2ID.items()}

# --- Mapping from canonical emotion_audio label → Arousal level ---
#
# Rationale per group:
#   HIGH   — delivery is forceful, rhythmically prominent, loud
#            Defiance (Tahaddi): confrontational tone, strong prosody
#            Pride    (Fakhr):   chest-voice, assertive cadence
#            Humor    (Turfah):  animated, raised pitch, playful energy
#
#   MEDIUM — expressive but not forceful; variable energy
#            Delicate Love   (Hub Raqeeq): tender, slight tremor, moderate pace
#            Admiration      (I'jab):      warm, rising intonation, moderate energy
#            Disappointment  (Khayba):     heavier than neutral, not as loud as defiance
#            Compassion      (Hanaan):     soft warmth, slight breathiness
#            Hope            (Amal):       lighter than pride, upward inflection
#            Longing         (Shawq):      breathy, drawn-out vowels, medium pace
#
#   LOW    — gentle, slow, minimal energy variation
#            Neutral/Descriptive (Wasfi):    flat delivery, recitative style
#            Sorrow          (Huzn):         heavy, falling intonation, slow tempo
#            Contemplation   (Ta'ammul):     meditative, near-whisper, minimal dynamics

_EMOTION_TO_AROUSAL: dict[str, str] = {
    # High
    "Defiance (Tahaddi)":              "High",
    "Pride (Fakhr)":                   "High",
    "Humor (Turfah)":                  "High",
    # Medium
    "Delicate Love (Hub Raqeeq)":      "Medium",
    "Admiration (I'jab)":              "Medium",
    "Disappointment (Khayba)":         "Medium",
    "Compassion (Hanaan)":             "Medium",
    "Hope (Amal)":                     "Medium",
    "Longing (Shawq)":                 "Medium",
    # Low
    "Neutral / Descriptive (Wasfi)":   "Low",
    "Sorrow (Huzn)":                   "Low",
    "Contemplation (Ta'ammul)":        "Low",
}


def emotion_to_arousal(emotion_label: str | None) -> str | None:
    """Convert a canonical emotion_audio string to its Arousal level.

    Returns None for null/unknown labels (do not train on these).
    Partial-match: if the label starts with a known prefix it still resolves.
    """
    if not emotion_label:
        return None
    label = emotion_label.strip()
    if label in _EMOTION_TO_AROUSAL:
        return _EMOTION_TO_AROUSAL[label]
    # Partial prefix match (handles short labels like "Sorrow")
    low = label.lower()
    for canonical, arousal in _EMOTION_TO_AROUSAL.items():
        if canonical.lower().startswith(low) or low.startswith(canonical.lower().split("(")[0].strip()):
            return arousal
    return None


def encode_arousal(emotion_label: str | None) -> int:
    """Convert emotion_audio label → Arousal integer ID.  Returns -1 for unknown."""
    arousal = emotion_to_arousal(emotion_label)
    if arousal is None:
        return -1
    return AROUSAL2ID[arousal]

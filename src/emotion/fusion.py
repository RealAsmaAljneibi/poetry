"""Re-export shim: the canonical implementation lives in src/models/emotion/fusion.py."""
from models.emotion.fusion import *  # noqa: F401, F403
from models.emotion.fusion import (
    apply_genre_constrained,
    apply_genre_prior,
    compute_delivery_metadata,
    decide_final_emotion,
    estimate_genre_emotion_prior,
    map_audio_emotion_to_core,
    map_text_emotion_to_core,
)

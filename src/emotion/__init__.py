# Compatibility shim — canonical location is src/models/emotion/
from src.models.emotion import *  # noqa: F401, F403
from src.models.emotion import (
    aggregate_confidence_weighted,
    aggregate_logits_mean,
    aggregate_probs_mean,
    aggregate_topk_vote,
    build_poem_emotion_summary,
    group_by_poem_id,
    poem_id_from_row,
    apply_genre_constrained,
    apply_genre_prior,
    compute_delivery_metadata,
    decide_final_emotion,
    estimate_genre_emotion_prior,
    map_audio_emotion_to_core,
    map_text_emotion_to_core,
)

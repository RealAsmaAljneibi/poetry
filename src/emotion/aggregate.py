"""Re-export shim: the canonical implementation lives in src/models/emotion/aggregate.py."""
from models.emotion.aggregate import *  # noqa: F401, F403
from models.emotion.aggregate import (
    RankedLabel,
    aggregate_confidence_weighted,
    aggregate_logits_mean,
    aggregate_probs_mean,
    aggregate_topk_vote,
    build_poem_emotion_summary,
    clip_support_summary,
    group_by_poem_id,
    poem_id_from_row,
    ranked_topk,
)

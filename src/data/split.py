import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from loguru import logger

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.labels import merge_genre_label


def split_dataset(
    input_jsonl: str | Path,
    output_dir: str | Path,
    seed: int = 42,
    low_poet_threshold: int = 3,
) -> None:
    """
    Strict poet-disjoint 80/10/10 split.

    This function follows the hard project requirement from `CLAUDE.md`:
    no poet may appear in more than one split. We therefore do NOT use any
    verse-level exceptions for rare genres. Instead, poets are sorted by clip
    count (largest first) and greedily assigned to the split with the largest
    current deficit relative to the 80/10/10 targets.
    """
    random.seed(seed)

    # ── Load & normalise ──────────────────────────────────────────────────────
    poet_to_records: dict[str, list] = defaultdict(list)
    with open(input_jsonl, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            rec["_genre_merged"] = merge_genre_label(rec.get("genre_en", "").strip())
            poet_to_records[rec["poet_en"]].append(rec)

    total_clips = sum(len(r) for r in poet_to_records.values())
    global_target = {
        "train": int(0.80 * total_clips),
        "val":   int(0.10 * total_clips),
        "test":  total_clips - int(0.80 * total_clips) - int(0.10 * total_clips),
    }

    assignment: dict[str, str] = {}
    split_counts = {"train": 0, "val": 0, "test": 0}
    split_to_poets: dict[str, list[str]] = {"train": [], "val": [], "test": []}

    poets_sorted = sorted(poet_to_records, key=lambda poet: (-len(poet_to_records[poet]), poet))
    split_order = ("train", "val", "test")

    def _deficit(split_name: str) -> int:
        return global_target[split_name] - split_counts[split_name]

    for poet in poets_sorted:
        chosen = max(split_order, key=lambda split_name: (_deficit(split_name), -split_counts[split_name], split_name == "train"))
        assignment[poet] = chosen
        split_to_poets[chosen].append(poet)
        split_counts[chosen] += len(poet_to_records[poet])

    # ── Write splits ──────────────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    for split_name in split_order:
        out_path = Path(output_dir) / f"{split_name}.jsonl"
        written = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for poet in split_to_poets[split_name]:
                for rec in poet_to_records[poet]:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
        counts[split_name] = written

    # ── Report ────────────────────────────────────────────────────────────────
    logger.info(
        f"\nSplit sizes: "
        f"train={counts['train']} ({counts['train']/total_clips:.1%})  "
        f"val={counts['val']} ({counts['val']/total_clips:.1%})  "
        f"test={counts['test']} ({counts['test']/total_clips:.1%})"
    )

    # Per-genre breakdown (using merged labels from all records)
    genre_split: dict[str, dict[str, int]] = defaultdict(lambda: {"train": 0, "val": 0, "test": 0})

    def _tally(recs_list: list, split_name: str) -> None:
        for rec in recs_list:
            g = rec.get("_genre_merged", rec.get("genre_en", "")).strip()
            if g:
                genre_split[g][split_name] += 1

    for split_name in split_order:
        for poet in split_to_poets[split_name]:
            _tally(poet_to_records[poet], split_name)

    logger.info(f"\n{'Genre':<48} {'Train':>10} {'Val':>8} {'Test':>8}")
    logger.info("-" * 80)
    for g in sorted(genre_split, key=lambda x: -sum(genre_split[x].values())):
        gc  = genre_split[g]
        tot = sum(gc.values())
        if tot == 0:
            continue
        logger.info(
            f"{g[:48]:<48} "
            f"{gc['train']:4d} ({gc['train']/tot*100:3.0f}%)  "
            f"{gc['val']:4d} ({gc['val']/tot*100:3.0f}%)  "
            f"{gc['test']:4d} ({gc['test']/tot*100:3.0f}%)"
        )


if __name__ == "__main__":
    logger.add("logs/split.log", rotation="10 MB")
    split_dataset("data/processed/master_dataset.jsonl", "data/processed/")

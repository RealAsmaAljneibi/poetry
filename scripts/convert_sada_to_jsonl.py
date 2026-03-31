"""
scripts/convert_sada_to_jsonl.py

Convert SADA (Saudi Arabic Audio Dataset) to the project's JSONL format
for use as intermediate fine-tuning data in the two-stage ASR adaptation.

SADA Dataset: https://www.kaggle.com/datasets/sdaiancai/sada2022
  - ~668 hours, Khaleeji/Gulf dialect coverage
  - Expected directory structure (from Kaggle download):
      sada/
        metadata.csv        (or multiple CSV files per split/dialect)
        audio/
          <speaker_id>/
            <utterance_id>.wav

Alternatively SADA may be organized as:
      sada/
        train/
          metadata.csv
          audio/
            *.wav
        dev/
          ...

This script handles both structures. Run:
    uv run python scripts/convert_sada_to_jsonl.py --sada-dir /path/to/sada

Output:
    data/processed/sada_train.jsonl  (~90% of SADA, for stage-1 pre-training)
    data/processed/sada_val.jsonl    (~10% of SADA, for stage-1 validation)

The JSONL records match NabatiASRDataset requirements:
  - audio_filename: absolute path to the full audio file
  - audio_offset_sec / audio_duration_sec: segment boundaries within the full file
  - text_corrected: human transcript (used as ASR target)
  - text_whisper:   "" (no Whisper baseline for SADA; not used in stage-1 training)
  - poet_en:        speaker_id from SADA (used as dummy poet id for grouping)
  - genre_en:       "SADA" (placeholder — SADA is not poetry)
  - source_poem:    utterance_id from SADA

NOTE: The SADA data is used ONLY for stage-1 domain anchoring. All evaluation
      is done against the Nabati test split. SADA clips are never mixed into
      the Nabati test or val splits.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

# loguru for consistent logging with the rest of the project
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# Known SADA column name variants across different release versions
AUDIO_COL_VARIANTS = ["audio_path", "audio", "file", "filename", "wav_path", "path"]
TEXT_COL_VARIANTS = [
    "groundtruthtext",
    "processedtext",
    "transcription",
    "text",
    "transcript",
    "sentence",
    "label",
    "normalized_text",
]
SPEAKER_COL_VARIANTS = ["speaker_id", "speaker", "client_id", "spkid"]
SEGMENT_ID_COL_VARIANTS = ["segmentid", "segment_id", "utt_id", "utterance_id"]
SEGMENT_START_COL_VARIANTS = ["segmentstart", "start", "offset", "segment_start"]
SEGMENT_END_COL_VARIANTS = ["segmentend", "end", "segment_end"]
SEGMENT_LENGTH_COL_VARIANTS = ["segmentlength", "duration", "segment_length"]

FILLER_EXACT = {
    "هاهاها",
    "الو",
    "السلام عليكم",
    "مع السلامة",
    "يا سلام",
    "اه",
    "اها",
}
NOISE_KEYWORDS = {
    "ضحك",
    "تصفيق",
    "غيرواضح",
}


def _find_column(header: list[str], variants: list[str]) -> str | None:
    """Return first matching column name (case-insensitive)."""
    lower = [h.lower().strip() for h in header]
    for v in variants:
        if v.lower() in lower:
            return header[lower.index(v.lower())]
    return None


def _discover_metadata_files(sada_dir: Path) -> list[Path]:
    """Find all CSV/TSV metadata files under sada_dir."""
    candidates = []
    for ext in ("*.csv", "*.tsv"):
        candidates.extend(sada_dir.rglob(ext))
    # Prefer files with 'train' / 'all' / 'metadata' in name
    def priority(p: Path) -> int:
        name = p.stem.lower()
        if "test" in name:
            return 3
        if "dev" in name or "val" in name:
            return 2
        return 1
    candidates.sort(key=priority)
    return candidates


def _parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _normalize_text(text: str) -> str:
    return " ".join(str(text).strip().split())


def _keep_record(
    record: dict,
    min_duration_sec: float,
    max_duration_sec: float,
) -> tuple[bool, str | None]:
    """Filter obviously low-value stage-1 ASR samples before writing JSONL."""
    text = _normalize_text(record.get("text_corrected", ""))
    duration = record.get("audio_duration_sec")

    if not text:
        return False, "empty_text"

    if duration is not None:
        duration = float(duration)
        if duration < min_duration_sec:
            return False, "too_short"
        if duration > max_duration_sec:
            return False, "too_long"

    if text in FILLER_EXACT:
        return False, "filler_exact"

    if any(keyword in text for keyword in NOISE_KEYWORDS):
        return False, "noise_keyword"

    # Collapse laughter-only or interjection-only strings.
    if re.fullmatch(r"(ها)+", text) or re.fullmatch(r"(ه+|ا+|اه+|اها+)", text):
        return False, "interjection_only"

    return True, None


def _parse_csv(csv_path: Path, sada_dir: Path) -> list[dict]:
    """Parse a SADA metadata CSV and return list of JSONL-ready dicts."""
    records = []
    delimiter = "\t" if csv_path.suffix == ".tsv" else ","

    with csv_path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delimiter)
        header = reader.fieldnames or []

        audio_col = _find_column(list(header), AUDIO_COL_VARIANTS)
        speaker_col = _find_column(list(header), SPEAKER_COL_VARIANTS)
        segment_id_col = _find_column(list(header), SEGMENT_ID_COL_VARIANTS)
        segment_start_col = _find_column(list(header), SEGMENT_START_COL_VARIANTS)
        segment_end_col = _find_column(list(header), SEGMENT_END_COL_VARIANTS)
        segment_length_col = _find_column(list(header), SEGMENT_LENGTH_COL_VARIANTS)
        preferred_text_col = _find_column(list(header), ["processedtext", "groundtruthtext"])
        text_col = preferred_text_col or _find_column(list(header), TEXT_COL_VARIANTS)

        if audio_col is None or text_col is None:
            logger.warning(
                "Skipping {}: could not detect audio/text columns in header {}",
                csv_path.name,
                header,
            )
            return []

        logger.info(
            "Parsing {} — audio_col='{}', text_col='{}', speaker_col='{}'",
            csv_path.name,
            audio_col,
            text_col,
            speaker_col or "(none)",
        )

        missing_audio = 0
        for row in reader:
            raw_audio = (row.get(audio_col) or "").strip()
            text = (row.get(text_col) or "").strip()
            speaker = (row.get(speaker_col or "") or "sada_speaker").strip()
            segment_id = (row.get(segment_id_col or "") or "").strip()
            start_sec = _parse_float(row.get(segment_start_col or ""))
            end_sec = _parse_float(row.get(segment_end_col or ""))
            duration_sec = _parse_float(row.get(segment_length_col or ""))

            if not raw_audio or not text:
                continue

            # Resolve audio path: may be absolute, relative to CSV, or relative to sada_dir
            audio_path = Path(raw_audio)
            if not audio_path.is_absolute():
                # Try relative to csv directory first, then sada_dir
                for base in (csv_path.parent, sada_dir):
                    candidate = base / raw_audio
                    if candidate.exists():
                        audio_path = candidate.resolve()
                        break
                else:
                    # Try stripping leading separators
                    stripped = raw_audio.lstrip("/\\")
                    for base in (csv_path.parent, sada_dir):
                        candidate = base / stripped
                        if candidate.exists():
                            audio_path = candidate.resolve()
                            break
                    else:
                        missing_audio += 1
                        continue

            # Derive utterance id from filename stem
            utt_id = segment_id or audio_path.stem
            speaker_id = f"{audio_path.stem}:{speaker}" if speaker else audio_path.stem
            if duration_sec is None and start_sec is not None and end_sec is not None:
                duration_sec = max(0.0, end_sec - start_sec)

            records.append({
                "audio_filename": str(audio_path),
                "audio_offset_sec": start_sec or 0.0,
                "audio_duration_sec": duration_sec,
                "text_corrected": text,
                "text_whisper":   "",          # no Whisper baseline for SADA
                "poet_en":        speaker_id,  # speaker ids are only file-local in SADA
                "poet_ar":        "",
                "source_poem":    utt_id,
                "poem_title":     None,
                "start":          int((start_sec or 0.0) * 1000),
                "end":            int((end_sec or ((start_sec or 0.0) + (duration_sec or 0.0))) * 1000),
                "genre_en":       "SADA",      # placeholder
                "genre_ar":       "",
                "emotion_text":   "Neutral / Descriptive (Wasfi)",
                "emotion_text_ar": "",
                "emotion_audio":  None,
                "khaleeji_value": None,
                "audio_quality":  "clean",
            })

    if missing_audio:
        logger.warning("{} rows skipped because audio files were not found under {}", missing_audio, sada_dir)
    return records


def _discover_audio_txt_pairs(sada_dir: Path) -> list[dict]:
    """
    Fallback: walk directory for .wav/.mp3 files and pair with .txt transcripts.
    Used when no metadata CSV is found.
    """
    records = []
    audio_exts = {".wav", ".mp3", ".flac", ".ogg"}

    for audio_path in sorted(sada_dir.rglob("*")):
        if audio_path.suffix.lower() not in audio_exts:
            continue
        txt_path = audio_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        text = txt_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        speaker = audio_path.parent.name  # assume parent dir = speaker_id
        records.append({
            "audio_filename": str(audio_path.resolve()),
            "audio_offset_sec": 0.0,
            "audio_duration_sec": None,
            "text_corrected": text,
            "text_whisper":   "",
            "poet_en":        speaker,
            "poet_ar":        "",
            "source_poem":    audio_path.stem,
            "poem_title":     None,
            "start":          0,
            "end":            0,
            "genre_en":       "SADA",
            "genre_ar":       "",
            "emotion_text":   "Neutral / Descriptive (Wasfi)",
            "emotion_text_ar": "",
            "emotion_audio":  None,
            "khaleeji_value": None,
            "audio_quality":  "clean",
        })

    return records


def convert(
    sada_dir: Path,
    output_dir: Path,
    val_ratio: float = 0.10,
    max_records: int | None = None,
    seed: int = 42,
    min_duration_sec: float = 1.5,
    max_duration_sec: float = 20.0,
) -> tuple[int, int]:
    """
    Convert SADA dataset to sada_train.jsonl + sada_val.jsonl.

    Returns (n_train, n_val).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Scanning SADA directory: {}", sada_dir)
    records: list[dict] = []

    # Try CSV/TSV metadata first
    meta_files = _discover_metadata_files(sada_dir)
    if meta_files:
        logger.info("Found {} metadata file(s): {}", len(meta_files), [f.name for f in meta_files])
        for mf in meta_files:
            batch = _parse_csv(mf, sada_dir)
            logger.info("  {} → {} records", mf.name, len(batch))
            records.extend(batch)
    else:
        logger.info("No CSV/TSV found — falling back to audio+txt pair discovery")
        records = _discover_audio_txt_pairs(sada_dir)

    if not records:
        logger.error("No records found in {}. Check directory structure.", sada_dir)
        sys.exit(1)

    # Deduplicate by full segment identity, not just parent audio file
    seen: set[tuple[str, str, int, int]] = set()
    unique: list[dict] = []
    for r in records:
        key = (
            str(r["audio_filename"]),
            str(r.get("source_poem", "")),
            int(r.get("start", 0)),
            int(r.get("end", 0)),
        )
        if key not in seen:
            seen.add(key)
            unique.append(r)
    logger.info("Total unique records before filtering: {}", len(unique))

    filtered: list[dict] = []
    filter_counts: dict[str, int] = {}
    for record in unique:
        keep, reason = _keep_record(
            record,
            min_duration_sec=min_duration_sec,
            max_duration_sec=max_duration_sec,
        )
        if keep:
            filtered.append(record)
        else:
            filter_counts[reason or "unknown"] = filter_counts.get(reason or "unknown", 0) + 1
    unique = filtered

    logger.info(
        "Filtered records for stage-1 quality: kept {} rows with duration in [{}, {}] sec",
        len(unique),
        min_duration_sec,
        max_duration_sec,
    )
    if filter_counts:
        logger.info("Filter breakdown: {}", filter_counts)

    if max_records and len(unique) > max_records:
        logger.info("Capping at {} records (--max-records)", max_records)
        rng = random.Random(seed)
        rng.shuffle(unique)
        unique = unique[:max_records]

    # Speaker-disjoint split (mirrors the project's poet-disjoint split policy)
    speakers = list({r["poet_en"] for r in unique})
    rng = random.Random(seed)
    rng.shuffle(speakers)
    n_val_speakers = max(1, int(len(speakers) * val_ratio))
    val_speakers   = set(speakers[:n_val_speakers])
    train_speakers = set(speakers[n_val_speakers:])

    train_records = [r for r in unique if r["poet_en"] in train_speakers]
    val_records   = [r for r in unique if r["poet_en"] in val_speakers]

    logger.info(
        "Speaker-disjoint split: {} train records ({} speakers), {} val records ({} speakers)",
        len(train_records),
        len(train_speakers),
        len(val_records),
        len(val_speakers),
    )

    # Write output
    train_path = output_dir / "sada_train.jsonl"
    val_path   = output_dir / "sada_val.jsonl"

    for path, split_records in [(train_path, train_records), (val_path, val_records)]:
        with path.open("w", encoding="utf-8") as fh:
            for rec in split_records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logger.info("Wrote {} → {} records", path, len(split_records))

    return len(train_records), len(val_records)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert SADA dataset to Nabat-AI JSONL format for stage-1 ASR pre-training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--sada-dir",
        type=Path,
        required=True,
        help="Path to the root of the downloaded SADA dataset.",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write sada_train.jsonl and sada_val.jsonl.",
    )
    p.add_argument(
        "--val-ratio",
        type=float,
        default=0.10,
        help="Fraction of speakers to put in the validation split.",
    )
    p.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Cap total records (useful for quick smoke tests). None = no cap.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    p.add_argument(
        "--min-duration-sec",
        type=float,
        default=1.5,
        help="Drop clips shorter than this threshold.",
    )
    p.add_argument(
        "--max-duration-sec",
        type=float,
        default=20.0,
        help="Drop clips longer than this threshold to avoid unstable stage-1 supervision.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.sada_dir.exists():
        logger.error("SADA directory not found: {}", args.sada_dir)
        sys.exit(1)

    n_train, n_val = convert(
        sada_dir=args.sada_dir,
        output_dir=args.output_dir,
        val_ratio=args.val_ratio,
        max_records=args.max_records,
        seed=args.seed,
        min_duration_sec=args.min_duration_sec,
        max_duration_sec=args.max_duration_sec,
    )

    logger.success("Done. sada_train.jsonl: {} records, sada_val.jsonl: {} records", n_train, n_val)
    logger.info("Output: {}", args.output_dir)
    logger.info("Next steps:")
    logger.info("  just train-asr-run5-sada-stage1   # Gulf dialect anchoring (stage 1)")
    logger.info("  just train-asr-run5-sada-stage2   # Nabati fine-tuning from stage-1 checkpoint")


if __name__ == "__main__":
    main()

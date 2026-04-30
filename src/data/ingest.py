from pathlib import Path

from loguru import logger
import pandas as pd
from pydantic import ValidationError

from src.data.schema import PoetrySample


def _safe_str(val) -> str:
    return str(val).strip() if pd.notna(val) else ""


def ingest_and_validate(
    excel_path: str | Path, audio_dir: str | Path, output_jsonl: str | Path
) -> None:
    logger.info(f"Loading raw dataset from {excel_path}")
    df = pd.read_excel(excel_path)

    valid_samples = []
    audio_dir_path = Path(audio_dir)

    for index, row in df.iterrows():
        try:
            clip_path = audio_dir_path / str(row["audio_filename"])
            if not clip_path.exists():
                continue

            sample = PoetrySample(
                audio_filename=clip_path,
                source_poem=_safe_str(row["source_poem"]),
                poem_title=_safe_str(row["poem_title"]) or None,
                start=int(row["start"]),
                end=int(row["end"]),
                text_whisper=_safe_str(row["text_whisper"]),
                text_corrected=_safe_str(row["text_corrected"]),
                poet_en=_safe_str(row["poet_en"]),
                poet_ar=_safe_str(row["poet_ar"]),
                genre_en=_safe_str(row["genre_en"]),
                genre_ar=_safe_str(row["genre_ar"]),
                emotion_text=_safe_str(row["emotion_text"]),
                emotion_text_ar=_safe_str(row["emotion_text_ar"]),
                emotion_audio=_safe_str(row["emotion_audio"]) or None,
                khaleeji_value=_safe_str(row["khaleeji_value"]) or None,
                khaleeji_value_ar=_safe_str(row["khaleeji_value_ar"]) or None,
                audio_quality=_safe_str(row["audio_quality"]) or None,
                translation_en=_safe_str(row["translation_en"]) or None,
                imagery_tags_en=_safe_str(row["imagery_tags_en"]) or None,
                poem_date=_safe_str(row["poem_date"]) or None,
            )
            valid_samples.append(sample)

        except ValidationError as e:
            logger.error(f"Row {index}: Validation failed: {e}")
        except KeyError as e:
            logger.error(f"Row {index}: Missing column: {e}")
            return

    output_path = Path(output_jsonl)
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in valid_samples:
            f.write(sample.model_dump_json() + "\n")

    logger.info(
        f"Successfully re-validated {len(valid_samples)} samples with rich metadata."
    )


if __name__ == "__main__":
    EXCEL_FILE = "data/processed/master_dataset_full.xlsx"
    AUDIO_FOLDER = (
        "/Users/Asma Salem Mubarak Najem Aljneibi/data-poetry-annotator/clips_dataset"
    )
    OUTPUT_FILE = "data/processed/validated_dataset.jsonl"

    logger.add("logs/ingest.log", rotation="10 MB")
    ingest_and_validate(EXCEL_FILE, AUDIO_FOLDER, OUTPUT_FILE)

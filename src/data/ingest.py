import pandas as pd
from pathlib import Path
from loguru import logger
from pydantic import ValidationError
from src.data.schema import PoetrySample

def ingest_and_validate(excel_path: str | Path, audio_dir: str | Path, output_jsonl: str | Path) -> None:
    logger.info(f"Loading raw dataset from {excel_path}")
    df = pd.read_excel(excel_path)
    
    valid_samples = []
    audio_dir_path = Path(audio_dir)
    
    for index, row in df.iterrows():
        try:
            clip_path = audio_dir_path / str(row['audio_filename'])
            if not clip_path.exists():
                continue
            
            # Helper to handle potential empty/NaN cells gracefully
            def safe_str(val):
                return str(val).strip() if pd.notna(val) else ""

            sample = PoetrySample(
                audio_path=clip_path,
                transcription=safe_str(row['text_corrected']),
                genre=safe_str(row['genre_en']),
                emotion=safe_str(row['emotion_cat_en']),
                poet=safe_str(row['poet_en']),
                translation=safe_str(row['translation_en']), 
                khaleeji_value=safe_str(row['gcc_value_cat_en'])
            )
            valid_samples.append(sample)
            
        except ValidationError as e:
            logger.error(f"Row {index}: Validation failed: {e}")
        except KeyError as e:
            logger.error(f"Row {index}: Missing column: {e}")
            return
            
    output_path = Path(output_jsonl)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in valid_samples:
            f.write(sample.model_dump_json() + '\n')
            
    logger.info(f"Successfully re-validated {len(valid_samples)} samples with rich metadata.")

if __name__ == "__main__":
    EXCEL_FILE = "data/processed/master_dataset_full.xlsx"
    AUDIO_FOLDER = "/Users/Asma Salem Mubarak Najem Aljneibi/data-poetry-annotator/clips_dataset"
    OUTPUT_FILE = "data/processed/validated_dataset.jsonl"
    
    logger.add("logs/ingest.log", rotation="10 MB")
    ingest_and_validate(EXCEL_FILE, AUDIO_FOLDER, OUTPUT_FILE)
"""Run analyze + search smoke tests and record latency + success."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent))
import app as demo_app  # noqa: E402
from demo import run_demo  # noqa: E402

TEST_PATH = PROJECT_ROOT / "data/processed/test.jsonl"
REPORT_PATH = PROJECT_ROOT / "outputs/reports/demo_smoke_test.json"


def main() -> None:
    logger.add(PROJECT_ROOT / "logs/demo_smoke_test.log", rotation="10 MB")
    rows = [
        json.loads(line)
        for line in TEST_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    clips = []
    seen = set()
    for row in rows:
        audio = row.get("audio_filename")
        if audio and audio not in seen:
            seen.add(audio)
            clips.append(audio)
        if len(clips) == 2:
            break

    analyze_results = []
    search_results = []
    if not clips:
        raise RuntimeError("No clips found in test split for smoke test.")

    corpus_clip = clips[0]
    external_source_clip = clips[1] if len(clips) > 1 else clips[0]

    with tempfile.TemporaryDirectory() as tmpdir:
        external_sim_path = (
            Path(tmpdir) / f"external_sim_{Path(external_source_clip).stem}.mp3"
        )
        shutil.copy2(external_source_clip, external_sim_path)

        scenarios = [
            {
                "label": "corpus",
                "clip": corpus_clip,
                "external_simulation": False,
            },
            {
                "label": "external_simulation",
                "clip": str(external_sim_path),
                "external_simulation": True,
                "source_clip": external_source_clip,
            },
        ]

        for idx, scenario in enumerate(scenarios, start=1):
            clip = scenario["clip"]
            out_path = PROJECT_ROOT / f"outputs/demo_smoke_result_{idx}.json"
            try:
                result = run_demo(
                    audio_path=Path(clip),
                    top_k=5,
                    out_path=out_path,
                    device="cpu",
                    use_lora=False,
                )
                analyze_results.append(
                    {
                        "mode": "analyze",
                        "scenario": scenario["label"],
                        "clip": clip,
                        "clip_id": Path(clip).stem,
                        "success": True,
                        "latency_ms": result.inference_ms,
                        "poem_id": result.poem_id,
                        "final_poem_emotion": result.emotion_poem_final,
                        "transcript_source": "corrected_corpus"
                        if result.asr_model == "corpus_corrected_text"
                        else "whisper_asr",
                        "external_simulation": scenario["external_simulation"],
                        "source_clip": scenario.get("source_clip"),
                    }
                )
                if result.poem_id and not scenario["external_simulation"]:
                    demo_app.build_poem_card_from_id(
                        result.poem_id,
                        top_k=5,
                        imagery_filter="",
                        device="cpu",
                    )
            except Exception as exc:
                logger.exception(exc)
                analyze_results.append(
                    {
                        "mode": "analyze",
                        "scenario": scenario["label"],
                        "clip": clip,
                        "clip_id": Path(clip).stem,
                        "success": False,
                        "error": str(exc),
                        "external_simulation": scenario["external_simulation"],
                        "source_clip": scenario.get("source_clip"),
                    }
                )

            try:
                search_payload = demo_app.run_audio_search_query(
                    audio_path=clip,
                    top_k=5,
                    genre_filter="",
                    imagery_filter="",
                    poet_filter="",
                    device="cpu",
                    use_lora=False,
                )
                search_results.append(
                    {
                        "mode": "search",
                        "scenario": scenario["label"],
                        "clip": clip,
                        "clip_id": Path(clip).stem,
                        "success": bool(search_payload["ok"]),
                        "poem_id": search_payload["rows"][0]["poem_id"]
                        if search_payload["rows"]
                        else None,
                        "top_candidate_score": search_payload["rows"][0]["score"]
                        if search_payload["rows"]
                        else None,
                        "query_source": search_payload["transcript_source"],
                        "query_text_preview": search_payload["query_text"][:120],
                        "error": search_payload["error"] or None,
                        "external_simulation": scenario["external_simulation"],
                        "source_clip": scenario.get("source_clip"),
                    }
                )
            except Exception as exc:
                logger.exception(exc)
                search_results.append(
                    {
                        "mode": "search",
                        "scenario": scenario["label"],
                        "clip": clip,
                        "clip_id": Path(clip).stem,
                        "success": False,
                        "error": str(exc),
                        "external_simulation": scenario["external_simulation"],
                        "source_clip": scenario.get("source_clip"),
                    }
                )

    all_results = analyze_results + search_results
    payload = {
        "n_examples": len(all_results),
        "success_rate": round(
            sum(int(item["success"]) for item in all_results)
            / max(len(all_results), 1),
            4,
        ),
        "analyze_results": analyze_results,
        "search_results": search_results,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.success("Saved demo smoke report → {}", REPORT_PATH)


if __name__ == "__main__":
    main()

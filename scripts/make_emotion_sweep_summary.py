"""
scripts/make_emotion_sweep_summary.py

Consolidates all emotion knob-sweep experiment JSONs into one summary table.

Usage:
    uv run python scripts/make_emotion_sweep_summary.py

Output:
    outputs/reports/emotion_experiments_summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
REPORT_DIR   = PROJECT_ROOT / "outputs/reports"

# Known experiments in order (will be skipped if file missing)
EXPECTED_REPORTS = [
    # Knob 0 — known baseline (from previous session, no new json)
    {
        "run_id": "baseline_none_window1",
        "report_path": None,    # use hardcoded values from previous run
        "hardcoded": {
            "run_id": "baseline_none_window1",
            "merge_profile": "none",
            "loss": "ce",
            "dropout": 0.1,
            "weight_decay": 0.01,
            "unfreeze_last_n": "full",
            "multitask": False,
            "seed": 42,
            "context_window": 1,
            "val_hard_macro_f1_clip": 0.2629,
            "test_hard_macro_f1_clip": 0.2124,
            "test_hard_macro_f1_poem": 0.2923,
            "test_top2_acc": None,
            "test_ece": None,
            "test_partial_credit": 0.687,
            "checkpoint_path": "outputs/models/arapoem_emotion/arapoem_emotion_text_best.pt",
            "confusion_matrix_path": "outputs/reports/arapoem_emotion_text_confusion.png",
            "note": "Existing best from window=1 run (bo2k8boyf). Macro-F1 and partial-credit from separate eval.",
        },
    },
]

# Dynamic experiment files to scan
SCAN_PATTERNS = [
    "emotion_eval_none.json",
    "emotion_eval_rare_merge_v1.json",
    "emotion_eval_merge_ce.json",
    "emotion_eval_merge_focal_g2.json",
    "emotion_sweep_reg_*.json",
    "emotion_sweep_unfreeze_*.json",
    "multitask_eval.json",
]


def load_report(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception as e:
        print(f"  Warning: could not load {path}: {e}")
        return None


def normalise(raw: dict) -> dict:
    """Map a report dict to the canonical summary schema."""
    is_multitask = raw.get("run_id", "").startswith("multitask")
    return {
        "run_id":                    raw.get("run_id", "?"),
        "merge_profile":             raw.get("emotion_merge_profile", raw.get("merge_profile", "?")),
        "loss":                      raw.get("loss", "ce"),
        "focal_gamma":               raw.get("focal_gamma"),
        "dropout":                   raw.get("dropout"),
        "weight_decay":              raw.get("weight_decay"),
        "unfreeze_last_n":           raw.get("unfreeze_last_n", "full"),
        "multitask":                 is_multitask,
        "genre_weight":              raw.get("genre_weight"),
        "emotion_weight":            raw.get("emotion_weight"),
        "seed":                      raw.get("seed", 42),
        "context_window":            raw.get("context_window", 1),
        "val_hard_macro_f1_clip":    raw.get("val_hard_macro_f1_clip"),
        "test_hard_macro_f1_clip":   raw.get("test_hard_macro_f1_clip",
                                             raw.get("test_emotion_clip_f1")),
        "test_hard_macro_f1_poem":   raw.get("test_hard_macro_f1_poem",
                                             raw.get("test_emotion_poem_f1")),
        "test_top2_acc":             raw.get("test_top2_acc",
                                             raw.get("test_emotion_top2_acc")),
        "test_ece":                  raw.get("test_ece",
                                             raw.get("test_emotion_ece")),
        "test_partial_credit":       raw.get("test_partial_credit"),
        "checkpoint_path":           raw.get("checkpoint_path"),
        "confusion_matrix_path":     raw.get("confusion_matrix_path",
                                             raw.get("emotion_confusion_matrix")),
        "note":                      raw.get("note", ""),
    }


def main() -> None:
    rows: list[dict] = []

    # Add hardcoded baseline
    rows.append(EXPECTED_REPORTS[0]["hardcoded"])

    # Scan dynamic report files
    for pattern in SCAN_PATTERNS:
        for p in sorted(REPORT_DIR.glob(pattern)):
            raw = load_report(p)
            if raw is None:
                continue
            if any(r["run_id"] == raw.get("run_id") for r in rows):
                continue  # deduplicate
            rows.append(normalise(raw))
            print(f"  Loaded: {p.name}  →  run_id={raw.get('run_id', '?')}")

    # Sort: multitask last, otherwise by test clip F1 desc
    def sort_key(r):
        f1 = r.get("test_hard_macro_f1_clip") or 0.0
        return (int(bool(r.get("multitask"))), -f1)

    rows.sort(key=sort_key)

    summary = {
        "generated_by":   "scripts/make_emotion_sweep_summary.py",
        "primary_metric": "test_hard_macro_f1_clip (Macro-F1 on 343 test clips)",
        "n_experiments":  len(rows),
        "best_run": max(
            (r for r in rows if r.get("test_hard_macro_f1_clip")),
            key=lambda r: r.get("test_hard_macro_f1_clip") or 0.0,
            default={},
        ).get("run_id", "?"),
        "experiments": rows,
    }

    out = REPORT_DIR / "emotion_experiments_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nSummary → {out}")
    print(f"Total experiments: {len(rows)}")
    print("\nRanking by test clip Macro-F1:")
    for r in rows:
        f1  = r.get("test_hard_macro_f1_clip")
        f1s = f"{f1:.4f}" if f1 is not None else "  —  "
        print(f"  {f1s}  {r['run_id']}")


if __name__ == "__main__":
    main()

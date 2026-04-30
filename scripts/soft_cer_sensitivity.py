from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.metrics import (
    SOFT_CER_COST_TABLE,
    _preprocess_soft,
    soft_cer,
    standard_cer,
)

PROJECT_ROOT = Path(__file__).parent.parent
REPORT_JSON = PROJECT_ROOT / "outputs/reports/soft_cer_sensitivity.json"
REPORT_MD = PROJECT_ROOT / "outputs/reports/soft_cer_sensitivity.md"


def unique_pairs(cost_table: dict[tuple[str, str], float]) -> list[tuple[str, str]]:
    pairs = {tuple(sorted((a, b))) for a, b in cost_table}
    return sorted(pairs)


def scaled_cost_table(scale: float) -> dict[tuple[str, str], float]:
    table: dict[tuple[str, str], float] = {}
    for (a, b), cost in SOFT_CER_COST_TABLE.items():
        table[(a, b)] = min(1.0, round(cost * scale, 4))
    return table


def uniform_cost_table(value: float) -> dict[tuple[str, str], float]:
    table: dict[tuple[str, str], float] = {}
    for a, b in SOFT_CER_COST_TABLE:
        table[(a, b)] = value
    return table


def align_substitutions(hypothesis: str, reference: str) -> list[tuple[str, str]]:
    hyp = _preprocess_soft(hypothesis).replace(" ", "")
    ref = _preprocess_soft(reference).replace(" ", "")
    m, n = len(hyp), len(ref)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if hyp[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    substitutions: list[tuple[str, str]] = []
    i, j = m, n
    tracked = set(SOFT_CER_COST_TABLE)
    while i > 0 or j > 0:
        if (
            i > 0
            and j > 0
            and hyp[i - 1] == ref[j - 1]
            and dp[i][j] == dp[i - 1][j - 1]
        ):
            i -= 1
            j -= 1
            continue

        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + 1:
            pair = (hyp[i - 1], ref[j - 1])
            if pair in tracked:
                substitutions.append(pair)
            i -= 1
            j -= 1
            continue

        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
            continue

        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            j -= 1
            continue

        break

    return substitutions


def corpus_frequency_cost_table(records: list[dict]) -> dict[tuple[str, str], float]:
    tracked_pairs = unique_pairs(SOFT_CER_COST_TABLE)
    counts: Counter[tuple[str, str]] = Counter()
    total = 0

    canonical_lookup = {tuple(sorted((a, b))): (a, b) for a, b in SOFT_CER_COST_TABLE}

    for row in records:
        for pair in align_substitutions(row["hyp"], row["ref"]):
            undirected = tuple(sorted(pair))
            if undirected in canonical_lookup:
                counts[undirected] += 1
                total += 1

    if total == 0:
        logger.warning(
            "No tracked dialect substitutions found; falling back to hand-assigned costs."
        )
        return dict(SOFT_CER_COST_TABLE)

    max_freq = max(counts.values()) if counts else 1
    empirical: dict[tuple[str, str], float] = {}
    for undirected in tracked_pairs:
        a, b = undirected
        relative = counts[undirected] / max_freq if max_freq else 0.0
        cost = round(1.0 - 0.95 * relative, 4)
        empirical[(a, b)] = cost
        empirical[(b, a)] = cost
        if undirected not in counts:
            empirical[(a, b)] = 1.0
            empirical[(b, a)] = 1.0
    return empirical


def load_records(split: str = "test") -> list[dict]:
    rows: list[dict] = []
    path = PROJECT_ROOT / f"data/processed/{split}.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        hyp = (rec.get("text_whisper") or "").strip()
        ref = (rec.get("text_corrected") or "").strip()
        if hyp and ref:
            rows.append({"hyp": hyp, "ref": ref})
    return rows


def evaluate(
    records: list[dict], name: str, cost_table: dict[tuple[str, str], float]
) -> dict:
    cer_scores = [standard_cer(row["hyp"], row["ref"]) for row in records]
    soft_scores = [
        soft_cer(row["hyp"], row["ref"], cost_table=cost_table) for row in records
    ]
    mean_cer = float(np.mean(cer_scores))
    mean_soft = float(np.mean(soft_scores))
    dps = (mean_cer - mean_soft) / max(mean_cer, 1e-9) * 100
    return {
        "name": name,
        "n": len(records),
        "mean_cer": round(mean_cer, 4),
        "mean_soft_cer": round(mean_soft, 4),
        "dialect_proximity_score_pct": round(dps, 2),
    }


def compare_tables(empirical_table: dict[tuple[str, str], float]) -> list[dict]:
    rows = []
    seen = set()
    for (a, b), hand_cost in SOFT_CER_COST_TABLE.items():
        undirected = tuple(sorted((a, b)))
        if undirected in seen:
            continue
        seen.add(undirected)
        rows.append(
            {
                "pair": f"{undirected[0]}↔{undirected[1]}",
                "hand_cost": hand_cost,
                "empirical_cost": empirical_table[(undirected[0], undirected[1])],
            }
        )
    rows.sort(
        key=lambda row: abs(row["hand_cost"] - row["empirical_cost"]), reverse=True
    )
    return rows


def render_markdown(summary: dict) -> str:
    variants = summary["variants"]
    lines = [
        "# Soft-CER Sensitivity Analysis",
        "",
        "Weighted Levenshtein is treated here as an exploratory diagnostic rather than a claimed novel metric.",
        "",
        "Research grounding: Arabic ASR normalization practice (e.g. MGB-3 / MR-WER),",
        "CODA/CAPHI-style dialect standardization, and phonologically weighted edit-distance literature.",
        "",
        f"Under cost perturbation, the Dialect Proximity Score ranges from `{summary['dps_range_pct'][0]:.2f}%` to `{summary['dps_range_pct'][1]:.2f}%`.",
        "",
        "## Variant Summary",
        "",
        "| Variant | CER | Soft-CER | Dialect Proximity Score |",
        "|---|---:|---:|---:|",
    ]
    for item in variants:
        lines.append(
            f"| {item['name']} | {item['mean_cer']:.4f} | {item['mean_soft_cer']:.4f} | {item['dialect_proximity_score_pct']:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Hand vs Empirical Costs",
            "",
            "| Pair | Hand cost | Empirical cost |",
            "|---|---:|---:|",
        ]
    )
    for row in summary["cost_comparison"][:15]:
        lines.append(
            f"| {row['pair']} | {row['hand_cost']:.2f} | {row['empirical_cost']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    logger.add(PROJECT_ROOT / "logs/soft_cer_sensitivity.log", rotation="10 MB")

    records = load_records("test")
    empirical = corpus_frequency_cost_table(records)
    variants = [
        evaluate(records, "hand_assigned", dict(SOFT_CER_COST_TABLE)),
        evaluate(records, "scaled_x0.5", scaled_cost_table(0.5)),
        evaluate(records, "scaled_x1.5", scaled_cost_table(1.5)),
        evaluate(records, "uniform_0.5", uniform_cost_table(0.5)),
        evaluate(records, "corpus_frequency", empirical),
    ]

    dps_values = [item["dialect_proximity_score_pct"] for item in variants]
    summary = {
        "variants": variants,
        "dps_range_pct": [round(min(dps_values), 2), round(max(dps_values), 2)],
        "cost_comparison": compare_tables(empirical),
        "references": [
            "MGB-3 / MR-WER style Arabic ASR normalization",
            "CODA and CAPHI dialect orthography / phonology frameworks",
            "phonologically weighted and learned edit-distance literature",
            "AraPoemBERT as the poetry-domain semantic tier context",
        ],
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    REPORT_MD.write_text(render_markdown(summary), encoding="utf-8")

    logger.success(f"Soft-CER sensitivity report → {REPORT_MD}")
    logger.success(f"Soft-CER sensitivity JSON → {REPORT_JSON}")


if __name__ == "__main__":
    main()

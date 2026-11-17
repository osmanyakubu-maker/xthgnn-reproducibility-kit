#!/usr/bin/env python3
"""Reproduce the coverage-aware nonempty-pair sensitivity analysis.

Run from any directory after cloning the repository:
    python supplementary_analyses/compute_coverage_sensitivity.py

The script reads the Version 2 per-seed explanation_metrics.json files under
v2_ablations/{dataset}/full/seed_{seed}/ and writes:
    supplementary_analyses/results/coverage_sensitivity_summary.csv
    supplementary_analyses/results/coverage_sensitivity_pairs.csv

For each seed, A_s is the set of target_node_ids and E_s is the set of
selected_event_rows in the shared evaluation window. The complete-pair
descriptive Jaccard uses 1 for empty/empty and 0 for empty/nonempty sets.
The nonempty-pair sensitivity excludes a seed pair if either seed has an
empty alert set or an empty evidence set. Spearman correlations are descriptive
only: the 45 pairwise observations share seeds, and no inferential p-values are
computed. A correlation is blank when either vector has zero variance.

Only Python's standard library is required.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Any

DATASETS = ("cadets", "theia", "trace", "streamspot")
SEEDS = (11, 22, 33, 44, 55, 66, 77, 88, 99, 110)


def rank_average(values: list[float]) -> list[float]:
    """Return one-based average ranks, assigning average ranks to ties."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        for pos in range(start, end):
            ranks[order[pos]] = average_rank
        start = end
    return ranks


def spearman(x: list[float], y: list[float]) -> float | None:
    """Spearman rho with average ranks; return None for a constant vector."""
    if len(x) != len(y) or len(x) < 2:
        return None
    rx, ry = rank_average(x), rank_average(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    dx = [v - mx for v in rx]
    dy = [v - my for v in ry]
    xx, yy = sum(v * v for v in dx), sum(v * v for v in dy)
    if xx == 0.0 or yy == 0.0:
        return None
    return sum(a * b for a, b in zip(dx, dy)) / math.sqrt(xx * yy)


def mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, None
    sd = math.sqrt(sum((value - mean) ** 2 for value in values) / (len(values) - 1))
    return mean, sd


def jaccard(left: set[Any], right: set[Any]) -> float:
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def load_seed_sets(root: Path, dataset: str, seed: int) -> tuple[set[Any], set[Any]]:
    path = root / "v2_ablations" / dataset / "full" / f"seed_{seed}" / "explanation_metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing archived explanation output: {path}")
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    records = payload.get("window_records")
    if not isinstance(records, list) or len(records) != 1:
        raise ValueError(f"Expected one shared evaluation window in {path}; got {len(records or [])}")
    record = records[0]
    if "target_node_ids" not in record or "selected_event_rows" not in record:
        raise ValueError(f"Missing target_node_ids or selected_event_rows in {path}")
    return set(record["target_node_ids"]), set(record["selected_event_rows"])


def analyze_dataset(root: Path, dataset: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed_sets = {seed: load_seed_sets(root, dataset, seed) for seed in SEEDS}
    rows: list[dict[str, Any]] = []
    for seed_a, seed_b in combinations(SEEDS, 2):
        alerts_a, events_a = seed_sets[seed_a]
        alerts_b, events_b = seed_sets[seed_b]
        eligible = bool(alerts_a and alerts_b and events_a and events_b)
        rows.append({
            "dataset": dataset,
            "seed_a": seed_a,
            "seed_b": seed_b,
            "alert_count_a": len(alerts_a),
            "alert_count_b": len(alerts_b),
            "event_row_count_a": len(events_a),
            "event_row_count_b": len(events_b),
            "alert_jaccard": jaccard(alerts_a, alerts_b),
            "evidence_jaccard": jaccard(events_a, events_b),
            "nonempty_pair_eligible": eligible,
        })

    eligible_rows = [row for row in rows if row["nonempty_pair_eligible"]]
    alert_all = [float(row["alert_jaccard"]) for row in rows]
    evidence_all = [float(row["evidence_jaccard"]) for row in rows]
    alert_valid = [float(row["alert_jaccard"]) for row in eligible_rows]
    evidence_valid = [float(row["evidence_jaccard"]) for row in eligible_rows]

    alert_all_mean, alert_all_sd = mean_sd(alert_all)
    evidence_all_mean, evidence_all_sd = mean_sd(evidence_all)
    alert_valid_mean, alert_valid_sd = mean_sd(alert_valid)
    evidence_valid_mean, evidence_valid_sd = mean_sd(evidence_valid)
    alert_counts = [len(seed_sets[seed][0]) for seed in SEEDS]
    evidence_counts = [len(seed_sets[seed][1]) for seed in SEEDS]

    summary = {
        "dataset": dataset,
        "seed_count": len(SEEDS),
        "zero_alert_seed_count": sum(count == 0 for count in alert_counts),
        "zero_evidence_seed_count": sum(count == 0 for count in evidence_counts),
        "all_pair_count": len(rows),
        "all_pair_alert_jaccard_mean": alert_all_mean,
        "all_pair_alert_jaccard_sample_sd": alert_all_sd,
        "all_pair_evidence_jaccard_mean": evidence_all_mean,
        "all_pair_evidence_jaccard_sample_sd": evidence_all_sd,
        "all_pair_spearman_rho_descriptive": spearman(alert_all, evidence_all),
        "nonempty_pair_count": len(eligible_rows),
        "nonempty_alert_jaccard_mean": alert_valid_mean,
        "nonempty_alert_jaccard_sample_sd": alert_valid_sd,
        "nonempty_evidence_jaccard_mean": evidence_valid_mean,
        "nonempty_evidence_jaccard_sample_sd": evidence_valid_sd,
        "nonempty_spearman_rho_descriptive": spearman(alert_valid, evidence_valid),
    }
    return summary, rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"No rows to write to {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1],
                        help="Repository root containing v2_ablations/ (default: inferred from script path)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results",
                        help="Directory for summary and pairwise CSV outputs")
    args = parser.parse_args()

    summaries, pair_rows = [], []
    for dataset in DATASETS:
        summary, rows = analyze_dataset(args.root, dataset)
        summaries.append(summary)
        pair_rows.extend(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "coverage_sensitivity_summary.csv"
    pairs_path = args.output_dir / "coverage_sensitivity_pairs.csv"
    write_csv(summary_path, summaries)
    write_csv(pairs_path, pair_rows)
    print(f"Wrote {summary_path}")
    print(f"Wrote {pairs_path}")


if __name__ == "__main__":
    main()

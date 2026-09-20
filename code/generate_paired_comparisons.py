#!/usr/bin/env python3
"""Generate seed-matched X-THGNN versus classical-control contrasts."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from run_resource_bounded_baselines import load_workflow


ROOT = Path(__file__).resolve().parent
WORK = ROOT / "evidence_work"
DATASETS = ("cadets", "theia", "trace", "streamspot")
MODELS = ("LogisticRegression", "RandomForest", "HistGradientBoosting")


def main() -> None:
    workflow = load_workflow()
    all_xthgnn = pd.read_csv(WORK / "summary" / "seed_metrics.csv")
    for dataset in DATASETS:
        baseline_dir = WORK / "baselines_resource_bounded" / dataset
        candidate_path = baseline_dir / "xthgnn_seed_metrics.csv"
        all_xthgnn[all_xthgnn["dataset"] == dataset].to_csv(candidate_path, index=False)
        for model in MODELS:
            workflow.compare_seed_tables(
                candidate_path,
                baseline_dir / "baseline_seed_metrics.csv",
                baseline_dir / "comparisons" / model,
                candidate_name="X-THGNN",
                reference_name=model,
                reference_model=model,
                permutations=20000,
                seed=20260821,
            )


if __name__ == "__main__":
    main()

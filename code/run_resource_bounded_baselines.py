#!/usr/bin/env python3
"""Run the workflow's declared resource-bounded classical controls.

This wrapper calls the unmodified public ``run_baselines`` entry point with
``fast=True`` and records the exact estimator budget in the resulting metadata.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path


# Apply the worker cap before importing NumPy, scikit-learn, or PyTorch through
# the workflow module. Four dataset jobs may still be run concurrently.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(variable, "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")


ROOT = Path(__file__).resolve().parent
WORKFLOW = ROOT / "upload" / "X_THGNN_Reproduction_v3(3)(1).py"
SEEDS = [11, 22, 33, 44, 55]


def load_workflow():
    spec = importlib.util.spec_from_file_location("xthgnn_workflow", WORKFLOW)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load workflow: {WORKFLOW}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--events", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    workflow = load_workflow()
    output = workflow.run_baselines(
        args.events,
        args.output,
        seeds=SEEDS,
        train_fraction=0.8,
        validation_fraction=0.1,
        fast=True,
    )

    metadata_path = Path(output) / "baseline_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "protocol": "resource-bounded classical falsification controls",
            "resource_budget": {
                "LogisticRegression_max_iter": 2000,
                "RandomForest_n_estimators": 40,
                "HistGradientBoosting_max_iter": 50,
                "logical_workers_per_dataset_process": 1,
            },
            "full_budget_attempt": (
                "Stopped uniformly before any numerical result was written; "
                "no partial full-budget results were retained."
            ),
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

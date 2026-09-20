#!/usr/bin/env python3
"""Run or resume the Version 2 graph-baseline matrix with bounded concurrency."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml


DATASETS = ("cadets", "theia", "trace", "streamspot")
MODELS = ("gcn", "gat", "temporal_gat", "rgcn")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = root / "configuration" / "manuscript_config_v4.yaml"
    payload = yaml.safe_load(protocol.read_text(encoding="utf-8"))
    seeds = [int(value) for value in payload["training_seeds"]]
    runner = root / "code" / "run_graph_baselines_v2.py"
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"})
    tasks = []
    for dataset in DATASETS:
        events = root / "normalized" / f"{dataset}_5000.csv"
        for model in MODELS:
            for seed in seeds:
                output = root / "v2_baselines" / dataset / model / f"seed_{seed}"
                tasks.append((dataset, model, seed, events, output))

    def run(task):
        dataset, model, seed, events, output = task
        metrics = output / "metrics.json"
        if metrics.exists():
            return f"skip {dataset} {model} {seed}"
        command = [
            sys.executable, str(runner), "--events", str(events), "--protocol", str(protocol),
            "--output", str(output), "--model", model, "--seed", str(seed),
        ]
        completed = subprocess.run(command, env=env, text=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError(
                f"{dataset} {model} seed {seed} failed\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
        return f"done {dataset} {model} {seed}"

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(run, task): task for task in tasks}
        for future in as_completed(futures):
            print(future.result(), flush=True)


if __name__ == "__main__":
    main()

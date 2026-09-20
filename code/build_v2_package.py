#!/usr/bin/env python3
"""Validate, manifest, and package the focused Version 2 evidence archive."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from pathlib import Path


SEEDS = {11, 22, 33, 44, 55, 66, 77, 88, 99, 110}
DATASETS = {"cadets", "theia", "trace", "streamspot"}
ABLATIONS = {
    "full",
    "minus_temporal_stream",
    "minus_heterogeneous_stream",
    "minus_joint_optimization",
    "minus_consistency_regularization",
    "minus_prototype_learning",
    "minus_temporal_attention",
    "minus_gating_mechanism",
}
BASELINES = {"gcn", "gat", "temporal_gat", "rgcn"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate(root: Path) -> None:
    ablation_metrics = list((root / "v2_ablations").glob("*/*/seed_*/metrics.json"))
    baseline_metrics = list((root / "v2_baselines").glob("*/*/seed_*/metrics.json"))
    scale_files = list((root / "v2_scale_benchmarks").glob("*/scalability_benchmark.csv"))
    assert len(ablation_metrics) == 320, len(ablation_metrics)
    assert len(baseline_metrics) == 160, len(baseline_metrics)
    assert len(scale_files) == 4, len(scale_files)
    observed_datasets = {p.parts[-4] for p in ablation_metrics}
    observed_ablations = {p.parts[-3] for p in ablation_metrics}
    observed_seeds = {int(p.parts[-2].removeprefix("seed_")) for p in ablation_metrics}
    assert observed_datasets == DATASETS, observed_datasets
    assert observed_ablations == ABLATIONS, observed_ablations
    assert observed_seeds == SEEDS, observed_seeds
    observed_baselines = {p.parts[-3] for p in baseline_metrics}
    baseline_seeds = {int(p.parts[-2].removeprefix("seed_")) for p in baseline_metrics}
    assert observed_baselines == BASELINES, observed_baselines
    assert baseline_seeds == SEEDS, baseline_seeds
    diagnostics = json.loads((root / "v2_summary/optimization_diagnostics.json").read_text())
    assert diagnostics["runs"] == 40
    assert diagnostics["runs_reaching_50_epoch_limit"] == 10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--zip", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.zip.resolve()
    validate(root)

    manifest_path = root / "archive_manifest.csv"
    files = sorted(
        path for path in root.rglob("*")
        if path.is_file() and path != manifest_path
    )
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["relative_path", "bytes", "sha256"])
        for path in files:
            writer.writerow([path.relative_to(root).as_posix(), path.stat().st_size, sha256(path)])

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            archive.write(path, (Path(root.name) / path.relative_to(root)).as_posix())
    checksum = output.with_suffix(output.suffix + ".sha256")
    checksum.write_text(f"{sha256(output)}  {output.name}\n", encoding="utf-8")
    print(json.dumps({
        "files": len(files) + 1,
        "zip": str(output),
        "zip_bytes": output.stat().st_size,
        "sha256_file": str(checksum),
    }, indent=2))


if __name__ == "__main__":
    main()

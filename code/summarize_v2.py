#!/usr/bin/env python3
"""Summarize Version 2 ablation and graph-baseline evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


def exact_sign_flip(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    observed = abs(float(differences.mean()))
    n = len(differences)
    values = []
    for mask in range(1 << n):
        signs = np.asarray([1.0 if mask & (1 << i) else -1.0 for i in range(n)])
        values.append(abs(float((differences * signs).mean())))
    return float(np.mean(np.asarray(values) >= observed - 1e-15))


def paired(candidate: pd.DataFrame, reference: pd.DataFrame, metric: str) -> dict:
    merged = candidate[["seed", metric]].merge(
        reference[["seed", metric]], on="seed", suffixes=("_candidate", "_reference")
    )
    delta = merged[f"{metric}_candidate"].to_numpy() - merged[f"{metric}_reference"].to_numpy()
    mean = float(delta.mean())
    sd = float(delta.std(ddof=1))
    half = float(stats.t.ppf(0.975, len(delta) - 1) * sd / np.sqrt(len(delta))) if sd else 0.0
    return {
        "metric": metric,
        "matched_seeds": len(delta),
        "paired_difference": mean,
        "ci95_low": mean - half,
        "ci95_high": mean + half,
        "paired_effect_dz": mean / sd if sd else None,
        "exact_sign_flip_pvalue_two_sided": exact_sign_flip(delta),
    }


def holm(values: list[float]) -> list[float]:
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * values[index]))
        adjusted[index] = running
    return adjusted.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    out = root / "v2_summary"
    out.mkdir(parents=True, exist_ok=True)

    ablation_rows = []
    for path in sorted((root / "v2_ablations").glob("*/ablation_seed_metrics.csv")):
        dataset = path.parent.name
        frame = pd.read_csv(path)
        frame = frame.rename(columns={
            "configuration": "ablation",
            "fidelity": "explanation_fidelity",
        })
        frame.insert(0, "dataset", dataset)
        ablation_rows.append(frame)
    if ablation_rows:
        ablations = pd.concat(ablation_rows, ignore_index=True)
        ablations.to_csv(out / "ablation_seed_metrics.csv", index=False)
        numeric = [column for column in ablations.columns if column not in {"dataset", "ablation", "seed"}]
        ablations.groupby(["dataset", "ablation"])[numeric].agg(["mean", "std"]).to_csv(
            out / "ablation_summary.csv"
        )
        comparisons = []
        for dataset, block in ablations.groupby("dataset"):
            full = block[block["ablation"] == "full"]
            for ablation, candidate in block.groupby("ablation"):
                if ablation == "full":
                    continue
                for metric in ("node_f1", "edge_auroc", "subgraph_f1", "explanation_fidelity"):
                    if metric not in block:
                        continue
                    row = paired(candidate, full, metric)
                    row.update({"dataset": dataset, "candidate": ablation, "reference": "full"})
                    comparisons.append(row)
        if comparisons:
            comp = pd.DataFrame(comparisons)
            comp["holm_pvalue_within_metric"] = np.nan
            for metric, indices in comp.groupby("metric").groups.items():
                comp.loc[indices, "holm_pvalue_within_metric"] = holm(
                    comp.loc[indices, "exact_sign_flip_pvalue_two_sided"].tolist()
                )
            comp.to_csv(out / "ablation_paired_comparisons.csv", index=False)

    graph_rows = []
    for path in sorted((root / "v2_baselines").glob("*/*/seed_*/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["dataset"] = path.parents[2].name
        graph_rows.append(payload)
    if graph_rows:
        graph = pd.DataFrame(graph_rows)
        graph.to_csv(out / "graph_baseline_seed_metrics.csv", index=False)
        numeric = [column for column in graph.columns if column not in {
            "dataset", "model", "seed", "primary_outcome", "implementation_scope"
        } and pd.api.types.is_numeric_dtype(graph[column])]
        graph.groupby(["dataset", "model"])[numeric].agg(["mean", "std"]).to_csv(
            out / "graph_baseline_summary.csv"
        )

    full_rows = []
    for metrics_path in sorted((root / "v2_ablations").glob("*/full/seed_*/metrics.json")):
        run = metrics_path.parent
        dataset = metrics_path.parents[2].name
        seed = int(run.name.removeprefix("seed_"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        explanation = json.loads((run / "explanation_metrics.json").read_text(encoding="utf-8"))
        config = json.loads((run / "config_resolved.json").read_text(encoding="utf-8"))
        history = pd.read_csv(run / "training_history.csv")
        validation_column = "validation_loss" if "validation_loss" in history else "validation_total"
        best_epoch = int(history.loc[history[validation_column].idxmin(), "epoch"])
        full_rows.append({
            "dataset": dataset,
            "model": "xthgnn",
            "seed": seed,
            "f1": metrics["node"]["f1"],
            "auroc": metrics["node"]["auroc"],
            "auprc": metrics["node"]["auprc"],
            "edge_f1_derived": metrics["edge"]["f1"],
            "edge_auroc_derived": metrics["edge"]["auroc"],
            "subgraph_f1_derived": metrics["subgraph"]["f1"],
            "explanation_fidelity": explanation["fidelity"],
            "explanation_stability": explanation["stability"],
            "deletion_auc": explanation.get("deletion_auc"),
            "insertion_probability_auc": explanation.get("insertion_probability_auc"),
            "normalized_insertion_gain_auc": explanation.get("normalized_insertion_gain_auc"),
            "epochs_executed": int(config["epochs_completed"]),
            "best_epoch": best_epoch,
            "reached_epoch_limit": int(config["epochs_completed"]) >= int(config["optimization"]["epochs"]),
        })
    if full_rows:
        full = pd.DataFrame(full_rows)
        full.to_csv(out / "xthgnn_full_seed_metrics.csv", index=False)
        full_numeric = [column for column in full.columns if column not in {"dataset", "model", "seed"}]
        full.groupby(["dataset", "model"])[full_numeric].agg(["mean", "std"]).to_csv(
            out / "xthgnn_full_summary.csv"
        )

        diagnostics = {
            "runs": int(len(full)),
            "runs_reaching_50_epoch_limit": int(full["reached_epoch_limit"].sum()),
            "median_epochs_executed": float(full["epochs_executed"].median()),
            "minimum_epochs_executed": int(full["epochs_executed"].min()),
            "maximum_epochs_executed": int(full["epochs_executed"].max()),
            "median_best_epoch": float(full["best_epoch"].median()),
            "seeds_per_dataset": full.groupby("dataset")["seed"].nunique().astype(int).to_dict(),
        }
        (out / "optimization_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2), encoding="utf-8"
        )

        if graph_rows:
            graph = pd.DataFrame(graph_rows)
            comparisons = []
            for dataset in sorted(set(full["dataset"]) & set(graph["dataset"])):
                candidate = full[full["dataset"] == dataset]
                for model, reference in graph[graph["dataset"] == dataset].groupby("model"):
                    for metric in ("f1", "auroc", "auprc"):
                        row = paired(candidate, reference, metric)
                        row.update({"dataset": dataset, "candidate": "xthgnn", "reference": model})
                        comparisons.append(row)
            graph_comp = pd.DataFrame(comparisons)
            graph_comp["holm_pvalue_within_metric"] = np.nan
            for metric, indices in graph_comp.groupby("metric").groups.items():
                graph_comp.loc[indices, "holm_pvalue_within_metric"] = holm(
                    graph_comp.loc[indices, "exact_sign_flip_pvalue_two_sided"].tolist()
                )
            graph_comp.to_csv(out / "graph_baseline_paired_comparisons.csv", index=False)

    classical_rows = []
    for path in sorted((root / "v2_classical_baselines").glob("*/baseline_seed_metrics.csv")):
        frame = pd.read_csv(path)
        frame.insert(0, "dataset", path.parent.name)
        classical_rows.append(frame)
    if classical_rows:
        classical = pd.concat(classical_rows, ignore_index=True)
        classical.to_csv(out / "classical_baseline_seed_metrics.csv", index=False)
        numeric = [column for column in classical.columns if column not in {"dataset", "model", "seed"}]
        classical.groupby(["dataset", "model"])[numeric].agg(["mean", "std"]).to_csv(
            out / "classical_baseline_summary.csv"
        )

    scale_rows = []
    for path in sorted((root / "v2_scale_benchmarks").glob("*/scalability_benchmark.csv")):
        frame = pd.read_csv(path)
        frame.insert(0, "dataset", path.parent.name)
        scale_rows.append(frame)
    if scale_rows:
        pd.concat(scale_rows, ignore_index=True).to_csv(out / "scalability_benchmark.csv", index=False)


if __name__ == "__main__":
    main()

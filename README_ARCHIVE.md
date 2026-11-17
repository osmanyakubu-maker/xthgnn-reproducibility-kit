# X-THGNN Reproducibility Archive

This repository preserves the focused Version 2 evidence archive and adds the corrected CRA expanded-analysis extension. The expanded package and matching supplementary report are also linked to the Zenodo record below.

- GitHub repository: https://github.com/osmanyakubu-maker/xthgnn-reproducibility-kit
- Expanded evidence package DOI: [10.5281/zenodo.24167813](https://doi.org/10.5281/zenodo.24167813)

## Scope

The archive supports methodological reproducibility. It does not claim state-of-the-art performance, causal component identification, certified robustness, independently adjudicated malicious-event labels, or deployment readiness. The original Version 2 experiments use fixed 5,000-edge extracts. The expanded CRA analysis evaluates additional fixed, class-aware 20,000- and 100,000-edge samples; these samples are not full-source releases and do not preserve natural prevalence.

## Version 2 baseline archive

Version 2 contains the ten-seed focused evaluation of X-THGNN:

- ten prespecified seeds: 11, 22, 33, 44, 55, 66, 77, 88, 99, and 110;
- 50 maximum epochs with patience-10 early stopping;
- seven component removals plus the full model across four datasets (320 runs);
- four parameter-matched graph references across four datasets and ten seeds (160 runs): GCN, GAT, Temporal-GAT, and R-GCN;
- unclipped per-target deletion and insertion explanation curves;
- exact paired sign-flip tests with Holm adjustment;
- single-thread CPU forward-latency measurements at 1,000, 2,500, and 5,000 edges.

Entity/node detection is the primary source-label-anchored outcome. Event-edge and evaluation-subgraph labels are derived and are secondary pipeline diagnostics. The original scale experiment characterizes runtime only within the available extracts and does not establish larger-graph accuracy.

## Expanded CRA extension

`expanded_CRA_v2_1/` adds the corrected explainer comparison, THEIA threshold sensitivity, correction notes, matching supplementary report, and a SHA-256 update manifest. Key results:

- The earlier explainer comparison is superseded because some scoring occurred while the model remained in training mode. The corrected rerun enforces evaluation mode, fixes evaluation randomness, and retains target-seed pairs with at least five eligible candidate edges. Nine targets qualify at 20K and eleven at 100K. No method has a Holm-adjusted advantage; this small, uneven sample does not establish equivalence.
- In the post hoc THEIA threshold sensitivity, a uniform +0.05 change to each seed's existing validation-selected threshold reduces alert-producing runs from 10/10 to 8/10 (28/45 seed-pair support) and mean test F1 from 0.961 to 0.703, with AUROC/AUPRC unchanged. This is run-level alert support, not population-level detection coverage and not evidence that the locked default threshold fails.
- The unreconciled older 5K latency table was removed from the manuscript because the checkpoint and benchmark protocols could not be reconciled from the available artifacts. The newer 100K-trained single-process CPU measurements remain in the supplementary report as bounded runtime measurements, not deployment guarantees.

The full normalized samples, checkpoints, execution outputs, provenance, and scripts are in the DOI-linked expanded package. The repository extension contains the corrected analysis scripts and result tables for quick review; see `expanded_CRA_v2_1/README.md` for its interpretation and download route.

## Main directories

- `configuration/manuscript_config_v4.yaml` - locked Version 2 protocol.
- `documentation/VERSION_2_PROTOCOL.md` - scope, outcome hierarchy, and decision rules.
- `code/X_THGNN_Reproduction_v3.py` - full-model, ablation, explanation, robustness, and scale workflow.
- `code/run_graph_baselines_v2.py` and `code/run_v2_graph_baseline_matrix.py` - graph-reference workflows.
- `code/summarize_v2.py` - aggregate summaries and paired inference.
- `normalized/` - four fixed 5,000-edge inputs and sidecars.
- `v2_ablations/`, `v2_baselines/`, `v2_scale_benchmarks/`, `v2_summary/` - Version 2 runs, runtime measurements, and summaries.
- `expanded_CRA_v2_1/` - corrected 20K/100K explainer and threshold analyses, supplementary report, correction notes, and checksums.
- `archive_manifest.csv` - SHA-256 and byte size for the original repository archive snapshot.
- `expanded_CRA_v2_1/UPDATE_MANIFEST.csv` - SHA-256 and byte size for files in the expanded CRA extension, excluding the manifest itself.

The earlier Version 1 directories remain for provenance. Use `v2_*` for the Version 2 conclusions and `expanded_CRA_v2_1/` plus the DOI-linked full package for the corrected larger-extract CRA analyses.

## Limitations

- The original Version 2 accuracy experiments use fixed 5,000-edge extracts; expanded results use fixed class-aware samples rather than full source releases or natural-prevalence test sets.
- Ten of forty full-model Version 2 runs reached the 50-epoch limit; ten seeds remain limited for small heterogeneous effects.
- Event-edge and evaluation-subgraph outcomes use derived labels.
- Detection and explanation results vary by dataset; no general detector or explainer superiority is established.
- The corrected explainer subset is small and uneven across datasets; absence of a detected difference is not evidence of equivalence.
- The CPU benchmarks do not establish deployment performance.

Start with `documentation/VERSION_2_PROTOCOL.md` for the baseline study and `expanded_CRA_v2_1/README.md` for the corrected extension.

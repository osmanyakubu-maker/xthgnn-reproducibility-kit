# X-THGNN Focused Version 2 Evidence Archive

This archive contains the executable evidence for the ten-seed focused Version 2
evaluation of X-THGNN. The operative public reproducibility location is:

https://github.com/osmanyakubu-maker/xthgnn-reproducibility-kit

No repository DOI is claimed unless and until a deposited release exists.

## Scope

The archive supports a methodological reproducibility claim. It does not claim
state-of-the-art performance, causal component identification, certified
robustness, independently adjudicated malicious-event labels, or accuracy beyond
the available 5,000-edge extracts.

Version 2 adds:

- ten prespecified seeds: 11, 22, 33, 44, 55, 66, 77, 88, 99, and 110;
- 50 maximum epochs with patience-10 early stopping;
- seven component removals plus the full model across four datasets (320 runs);
- four parameter-matched PyTorch Geometric graph references across four datasets
  and ten seeds (160 runs): GCN, GAT, Temporal-GAT, and R-GCN;
- unclipped per-target deletion and insertion explanation curves;
- exact paired sign-flip tests with Holm adjustment;
- single-thread CPU forward-latency measurements at 1,000, 2,500, and 5,000
  edges.

## Primary and secondary outcomes

Entity/node detection is the primary source-label-anchored outcome. Event-edge
and evaluation-subgraph labels are derived and must be interpreted as secondary
pipeline diagnostics. The scale experiment characterizes forward latency only
within the available extracts and does not establish larger-graph accuracy.

## Main directories

- `configuration/manuscript_config_v4.yaml` - locked Version 2 protocol.
- `documentation/VERSION_2_PROTOCOL.md` - scope, outcome hierarchy, and decision
  rules.
- `code/X_THGNN_Reproduction_v3.py` - full-model, ablation, explanation,
  robustness, and scale workflow.
- `code/run_graph_baselines_v2.py` - one competitive graph-reference run.
- `code/run_v2_graph_baseline_matrix.py` - resumable graph-reference matrix.
- `code/summarize_v2.py` - aggregate summaries and paired inference.
- `normalized/` - four fixed 5,000-edge inputs and sidecars.
- `v2_ablations/` - 320 full-model and component-removal run directories.
- `v2_baselines/` - 160 graph-reference run directories.
- `v2_scale_benchmarks/` - measured prefix-latency results and metadata.
- `v2_summary/` - seed-level tables, aggregate tables, exact tests, and
  optimization diagnostics.
- `archive_manifest.csv` - SHA-256 and byte size for every archived file other
  than the manifest itself.

The earlier Version 1 directories are retained for provenance and comparison.
Version 2 conclusions must be taken from the `v2_*` directories.

## Principal limitations preserved in the evidence

- All accuracy experiments use fixed 5,000-edge extracts.
- Ten of forty full-model runs reached the 50-epoch limit.
- Ten seeds improve inference but remain limited for small heterogeneous effects.
- Event-edge and evaluation-subgraph outcomes use derived labels.
- Explanation influence is weak on CADETS and StreamSpot.
- Competitive performance is dataset dependent; no paired node-F1 contrast
  survives Holm adjustment.

Start with `documentation/VERSION_2_PROTOCOL.md`, then inspect
`v2_summary/optimization_diagnostics.json` and the complete seed tables.


# X-THGNN submission artifact checklist

The items below are included in `X_THGNN_Reproduction_Anonymous_v1.zip`. Independent
reproducibility should be claimed only after that complete archive is deposited
in a permanent repository and its DOI or versioned link is added to the paper.

## Included in this revised package

- [x] Manuscript intentionally excluded for double-anonymous peer review
- [x] Executable Python workflow
- [x] Locked manuscript configuration
- [x] Label-source integrity audit summary
- [x] Reproduction guide
- [x] Dependency specification

## Completed evidence archive

- [x] CADETS, THEIA, and TRACE source UUID label files
- [x] Four normalized 5,000-edge extracts
- [x] Extraction sidecars and normalized-file SHA-256 hashes
- [x] Twenty X-THGNN seed directories (four datasets x five seeds)
- [x] `config_resolved.json` and `training_history.csv` for every run
- [x] Node, edge, and subgraph prediction CSVs for every run
- [x] Explanation and robustness JSON outputs for every run
- [x] Model checkpoints
- [x] Classical-baseline seed metrics and summaries
- [x] Matched-seed confidence intervals, paired effect sizes, and sign-flip tests
- [x] Exact resource-budget and worker-cap metadata for every comparator dataset
- [x] `seed_metrics.csv` and `seed_summary.csv`
- [x] Figure-generation metadata and source hashes
- [x] Successful full PyTorch self-test report in the locked environment
- [x] Complete archive file manifest with SHA-256 checksums
- [x] Training-budget diagnostic derived from all twenty training histories
- [x] Explanation-metric ceiling diagnostic derived from all twenty explanation files
- [x] Five-seed statistical-resolution diagnostic with explicit assumptions
- [x] Public-source repository commits, licences, and graph-archive hashes

## Claim-extension experiments

These are not required to support the manuscript's bounded Tier A audit claim,
because the paper makes neither a component-causality nor a state-of-the-art
ranking claim. They become mandatory before advancing the indicated stronger
claims:

- [ ] Parameter-matched temporal-GNN, heterogeneous-GNN, and MAGIC comparators
      before any competitive-superiority claim
- [ ] Full component ablation matrix before any component-mechanism claim
- [ ] GNNExplainer and GraphMask comparisons before any explainer-ranking claim
- [ ] At least one natural-prevalence or full-release validation

The evidence archive above remains mandatory for submission even under the
bounded Tier A claim. The remaining action is external deposition and insertion
of the permanent DOI or versioned repository link in the manuscript.

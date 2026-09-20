# X-THGNN Focused Version 2 Protocol

This protocol was fixed before inspecting any Version 2 test result. It preserves the Version 1 archive and treats the ten-seed extension as a separate experiment.

## Confirmatory changes

- Ten training seeds: 11, 22, 33, 44, 55, 66, 77, 88, 99, and 110.
- Maximum 50 epochs with early-stopping patience 10.
- Entity/node F1 is the primary source-label-anchored outcome.
- Endpoint-derived event-edge and evaluation-subgraph/window outcomes are secondary sensitivity analyses.
- All seven implemented component removals are evaluated against the full model.
- Competitive graph baselines are evaluated under the same ordered partitions, validation-only threshold selection, seed list, and reporting contract.
- Explanation evaluation archives unclipped baseline, deletion, and insertion scores over a fixed retained-fraction grid.
- Paired inference reports effect sizes, confidence intervals, exact sign-flip tests, Holm-adjusted p-values, and complete seed-level data.

## Unchanged boundary

The available normalized files contain 5,000 edges per extract. Prefix benchmarks characterize runtime only. No result from these files is presented as full-release accuracy, natural-prevalence deployment performance, independently adjudicated malicious-event detection, or attack-stage classification.

## Fail-closed rule

If a configured run, seed, baseline, metric, or provenance record is absent, its associated claim remains unsupported. Weak or negative outcomes are retained rather than replaced through post-test tuning.

# CRA expanded analysis extension (20K/100K)

This folder adds the corrected explainer comparison and THEIA threshold sensitivity to the preserved 5K Version 2 repository archive. The complete expanded data-and-code package, including normalized samples, run outputs, scripts, provenance, and the matching supplement, is deposited at [Zenodo DOI 10.5281/zenodo.24167813](https://doi.org/10.5281/zenodo.24167813).

## Corrected explainer comparison

Use the `20k/` and `100k/` fixed-evaluation outputs in this folder; these supersede the earlier exploratory comparison. The rerun explicitly sets the model to evaluation mode before explanation and scoring, fixes evaluation randomness, and includes only target-seed pairs with at least five eligible candidate edges. Nine targets qualify at 20K and eleven at 100K. No explainer has a Holm-adjusted advantage over the learned ranking. This small, uneven subset does not establish equivalence. The scripts are included; the complete matching input graphs, checkpoints, and execution outputs are in the Zenodo package.

The one-edge diagnostic is `100k/explainer_comparison_fixed_eval/one_edge_diagnostic.csv`; all five rankings tie when only one candidate edge exists.

## THEIA threshold sensitivity

The post hoc +0.05 shift uses saved held-out predictions and each seed's validation-selected threshold. It leaves AUROC/AUPRC unchanged, while alert-producing runs drop from 10/10 to 8/10, seed-pair support is 28/45, and mean F1 falls from 0.961 to 0.703. This is a run-level alert-support sensitivity, not a population-level detection-coverage estimate or evidence that the locked default threshold fails.

## Latency correction

The unreconciled older 5K latency table was removed from the manuscript because the checkpoint and benchmark protocols could not be reconciled from the available artifacts. The newer 100K-trained single-process CPU measurements remain in the supplementary report as bounded runtime measurements, not deployment guarantees.

`UPDATE_MANIFEST.csv` records SHA-256 and byte size for each file in this extension other than the manifest itself. The original `archive_manifest.csv` continues to describe the original repository archive.

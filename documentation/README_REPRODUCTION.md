# X-THGNN label-provenance-audited reproduction package

This package supports a reproducibility-audit case study. X-THGNN is the audit
target, not a proposed state-of-the-art detector. The package does not contain
hard-coded result tables and does not infer missing experiments.

## Claim contract

- **Tier A — asserted:** bounded whole-system reproducibility audit against
  transparent falsification controls.
- **Tier B — not asserted:** component-mechanism attribution; this requires the
  locked ablation matrix.
- **Tier C — not asserted:** competitive research-system superiority; this
  requires matched temporal/heterogeneous GNN, MAGIC, and explainer baselines.

Results and conclusions must not be generalized across these tiers.

## 1. Install the locked environment

Use Python 3.10 and install `requirements-xthgnn.txt`.

## 2. Audit the supplied DARPA entity-label files

Run `audit-ground-truth` with the CADETS, THEIA, and TRACE UUID text files.
The command validates every identifier and writes file-level and canonical
SHA-256 hashes, unique counts, and duplicate counts.

This is an integrity and provenance audit, not independent semantic
adjudication. The source files use the ThreaTrace/MAGIC labeling convention,
which may include context-expanded neighbouring entities. Treat positive UUIDs
as benchmark anomaly/entity labels rather than unqualified proof that every
listed entity directly participated in an attack.

## 3. Normalize data

- Use `normalize-tc-ground-truth` for raw DARPA CDM JSONL shards. Entity labels
  are direct source-file membership indicators; event-edge labels are derived
  when an endpoint is listed.
- Use `normalize-magic` for public MAGIC graph objects. Pass `--ground-truth`
  for a file-integrity and label-source audit. The sidecar records whether an
  identity-level UUID-to-serialized-node mapping could be verified.
- Use `normalize-magic --dataset streamspot` for StreamSpot scenario graphs.

## 4. Execute the locked protocol

Generate the exact YAML with `manuscript-config`. For each normalized dataset,
run `train` under seeds 11, 22, 33, 44, and 55. The deposited classical controls
were produced with `run_resource_bounded_baselines.py`: logistic regression uses
`max_iter=2000`, random forest uses 40 trees, histogram gradient boosting uses
50 iterations, and each dataset process is capped at one logical worker.
Use `ablate` only when producing a component-analysis extension. Ablation
outputs are not part of the present Tier A audit evidence.

## 5. Aggregate and verify

Use `summarize-runs` to create seed-level and summary tables, `make-figures`
to regenerate Figures 2 and 3, and `generate_paired_comparisons.py` for the
deposited matched-seed confidence intervals, paired effect sizes, and sign-flip
tests.
Deposit normalized-file hashes, extraction sidecars, complete seed directories,
prediction CSVs, and figure-generation metadata with the manuscript.

## 6. Mandatory submission bundle

The following items are required before the numerical tables can be treated as
independently auditable: `requirements-xthgnn.txt`; the three source UUID files;
four normalized extracts and their sidecars; twenty X-THGNN seed directories;
baseline seed metrics; `seed_metrics.csv`; `seed_summary.csv`; prediction CSVs;
checkpoints; and figure-generation metadata. The manuscript reports completed
runs, but the script, configuration, and summary audit alone are not substitutes
for this archive. Archive deposition is a submission gate.

The completed local bundle is `X_THGNN_Evidence_Archive_v1.zip`. It contains the
twenty X-THGNN seed directories, matched classical-baseline outputs, normalized
extracts and sidecars, audited source labels, summary CSVs, regenerated figures,
locked-environment report, public-source provenance, licences, and a complete
SHA-256 file manifest. Deposit the ZIP in a permanent repository and replace the
manuscript's repository placeholder with its DOI or versioned URL.

## Scientific boundary

The reported manuscript results concern fixed 5,000-edge public extracts and
source-defined benchmark labels.
Natural-prevalence deployment, full-release scaling, attack-stage inference,
cross-organization effectiveness, and independently adjudicated event-level
maliciousness are not claimed.

# Correction notes

The earlier explainer comparison could score the model while it remained in training mode. Dropout and batch-normalization state could affect repeated scores. The corrected analysis explicitly restores evaluation mode, fixes evaluation randomness, and filters to target-seed pairs with at least five eligible candidate edges. Use only the fixed-evaluation results here; the prior comparison is superseded. The qualifying sample is 9 targets at 20K and 11 at 100K. No Holm-adjusted advantage was detected, but this small sample does not establish equivalence.

The THEIA +0.05 threshold shift is an exploratory operating-point sensitivity using saved held-out scores and existing validation-selected thresholds; it is not test-set threshold fitting and does not replace the locked operating point. Alert support is measured at the run/seed level as whether at least one alert exists; it is not an estimate of the fraction of attack events detected.

The earlier 5K latency table was removed because its protocol and checkpoints were not comparable with the newer 100K-trained CPU benchmark. The available artifacts did not isolate the cause of the gap. The updated manuscript retains the newer bounded measurements in its supplement.

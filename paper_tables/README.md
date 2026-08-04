# Paper-Facing Aggregate Results

This directory contains only aggregate results that support the claims in the
submitted manuscript. It contains no raw clinical text, patient identifiers,
per-episode predictions, model checkpoints, or embedding caches.

- `main_test_performance.csv` corresponds to the manuscript's main test-set
  comparison table.
- `representation_ablation_validation.csv` corresponds to the controlled
  validation representation ablation.
- `patient_cluster_bootstrap_ci.csv` supports the paired patient-cluster
  bootstrap contrasts reported in the Results section.

For the main performance table, values are means and sample standard
deviations across supervised-training seeds 17, 23, and 42. Precision, recall,
and F1 use a threshold selected on validation F1 for each seed and applied
unchanged to test. AUPRC, AUROC, Brier score, and ECE use continuous
probabilities.

The representation ablation used validation AUPRC for selection. The two mixed
configurations were screened with seed 17 only; the single-summary control and
full multi-span configuration were subsequently evaluated across all three
seeds.

The bootstrap file reports seed-matched paired differences using 1,000 common
patient-cluster resamples. Each sampled patient's complete set of test episodes
was retained with cluster multiplicity.

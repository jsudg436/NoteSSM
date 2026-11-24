# Data and Artifact Policy

This code release does not redistribute MIMIC-IV, MIMIC-IV-Note, raw discharge summaries, or any derived raw-text representation.

Users must obtain credentialed access to the required clinical datasets through the official data providers and configure local paths with `configs/paths.local.yaml`.

Large experiment artifacts should be stored outside git:

- DAPT checkpoints.
- Token shards.
- Embedding caches.
- Per-example predictions.
- Frozen result archives.

For the final Mamba3 joint D S2/S2 aux=0.0 experiment, the expected external archive layout is:

```text
dapt_mamba3_joint_D_final_valtest_3seed_aux00_20260803/
  FREEZE_MANIFEST.json
  SHA256SUMS.txt
  artifact/
    D_final_valtest_3seed_aggregate.csv
    D_final_valtest_threshold_tuned_metrics_all_seeds.csv
    notesm_v2_hybrid_threshold_tuned_metrics_seed17.csv
    notesm_v2_hybrid_threshold_tuned_metrics_seed23.csv
    notesm_v2_hybrid_threshold_tuned_metrics_seed42.csv
```

Verify external artifacts with:

```bash
sha256sum -c SHA256SUMS.txt
```

## Public experimental metadata

[docs/EXPERIMENT_PROVENANCE.md](docs/EXPERIMENT_PROVENANCE.md) and its JSON
companion publish an allowlisted subset of original experiment settings,
run identifiers, and hashes. They do not publish the private archive or its
complete manifests. The public source-file checksum manifest and this
selected experimental metadata serve different purposes.

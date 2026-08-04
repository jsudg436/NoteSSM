# NoteSSM

Code release scaffold for:

**NoteSSM: Reducing Local Representation Compression in Long Discharge Summaries via Hierarchical Multi-Span Modeling**

This repository contains the model, preprocessing, extraction, downstream training, validation, and aggregation code used for the NoteSSM experiments. It intentionally does not include raw clinical text, MIMIC-IV tables, model checkpoints, embedding caches, prediction CSVs, or other large/private artifacts.

## What Is Included

- `src/notesm_v2/`: core package code for benchmark construction, section parsing, model components, training helpers, and evaluation.
- `scripts/`: command-line scripts for benchmark construction, DAPT, embedding extraction, joint S1/S2 multi-span extraction, downstream training, validation, and freezing aggregate results.
- `tests/`: targeted tests for the public pipeline.
- `configs/`: public example configuration files with placeholder paths.
- `paper_tables/`: manuscript-facing aggregate performance, representation-ablation, and patient-cluster bootstrap tables, without per-example predictions.
- `docs/`: protocol and benchmark notes that are safe to publish.

## Not Included

The following are deliberately excluded and should not be committed:

- Raw MIMIC-IV or MIMIC-IV-Note data.
- Derived raw-text files.
- Token shards, embedding caches, `.npy`, `.parquet`, `.pt`, `.pth`, `.safetensors`, checkpoints, and logs.
- Per-example prediction CSVs.
- Hugging Face tokens, local `.env` files, RunPod runtime env files, and other credentials.

## Environment

Create an environment with Python 3.10+ or 3.12. Install a PyTorch build
compatible with the target CPU/CUDA platform first, then install the remaining
dependencies and the package without dependency resolution:

```bash
python -m pip install -r requirements.txt
python -m pip install -e . --no-deps
```

The versions of non-CUDA packages recorded for the paper can be installed with
`requirements-paper.txt`. For Mamba3 extraction or DAPT, the runtime also
needs compatible CUDA, PyTorch, `mamba-ssm`, `causal-conv1d`, Triton, and
Hugging Face model access. These GPU dependencies are intentionally not
installed through the package metadata because they depend on the CUDA image.
The paper runs used:

- Python 3.12
- PyTorch `2.8.0+cu128`
- CUDA `12.8`
- Triton `3.7.1`
- `mamba-ssm==2.3.2.post1`
- `causal-conv1d==1.6.2.post1`
- `transformers==5.14.1`
- `pandas==3.0.5`, `pyarrow==25.0.0`, `scikit-learn==1.9.0`

## Configuration

Copy and edit the examples:

```bash
cp .env.example .env
cp configs/paths.example.yaml configs/paths.local.yaml
cp configs/data_lampr.example.yaml configs/data_lampr.local.yaml
```

Do not commit the edited local files.

## Main Pipeline

The high-level reproducibility flow is:

1. Build the LAMP-R benchmark from local MIMIC-IV and MIMIC-IV-Note files.
2. Build section and fallback-chunk caches.
3. Prepare DAPT token shards and train the frozen-backbone checkpoint.
4. Extract note-level embeddings and joint local S1/S2 embeddings.
5. Train/evaluate the downstream NoteSSM heads.
6. Freeze aggregate results and compare against baselines.

Example entry points:

```bash
python scripts/build_lampr_benchmark.py --config configs/data_lampr.local.yaml
python scripts/train_dapt_from_shards.py --help
python scripts/extract_mamba3_joint_s1_span2.py --help
python scripts/train_downstream.py --help
python scripts/tune_validation_threshold.py --help
python scripts/freeze_notesm_multispan_patient_bootstrap.py --help
```

Paper-setting reproduction commands are provided in
`docs/PAPER_REPRODUCTION_COMMANDS.md`. They include clean train-only Mamba3
domain adaptation, controlled single-summary/multi-span extraction, final
multi-span three-seed validation/test evaluation, and patient-cluster
bootstrap steps.

## Final Reported Results

The manuscript-facing aggregate files in `paper_tables/` report the selected
multi-span NoteSSM system, its external baseline comparisons, the controlled
single-summary versus multi-span ablation, and paired patient-cluster bootstrap
intervals. The final test result across seeds 17, 23, and 42 is:

- test AUPRC: `0.3499 +/- 0.0016`
- test AUROC: `0.7045 +/- 0.0015`
- test F1: `0.4148 +/- 0.0004`

The full frozen result artifact and per-episode predictions are not stored in
git. They should be archived separately with their manifests and SHA256
checksums.

## Public Release Checks

Before creating a GitHub repository, run:

```bash
python scripts/check_public_release.py . --write-manifest
python scripts/check_public_release.py .
python -m pytest -q tests
```

Review `RELEASE_CHECKLIST.md` before pushing.

## License

This package is released under the MIT License. Before upload, verify the
copyright holder string in `LICENSE` matches the author group or institution.

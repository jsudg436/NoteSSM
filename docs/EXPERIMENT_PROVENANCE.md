# Experimental provenance: original NoteSSM extraction

Prepared: 2026-11-24

This document publishes a selected, non-patient-level subset of the preserved
records for the **original joint single-summary/two-span local extraction**
and the **original final D three-seed selection lock**. The accompanying
[EXPERIMENT_PROVENANCE.json](EXPERIMENT_PROVENANCE.json) contains the exact
forward-contract objects and recorded experiment identifiers.

This is distinct from [PUBLIC_RELEASE_MANIFEST.txt](../PUBLIC_RELEASE_MANIFEST.txt),
which checksums the public source files. It is not a manifest of every experiment
in the camera-ready paper. In particular, the later A6000 re-extractions,
additional controls, and convergence suites are outside this document's scope;
their artifacts must not be identified by the original hashes below.

## Evidence and verification

- Values were checked against four preserved extraction provenance records:
  section and chunk, each for training/validation and test.
- The original final-head settings were checked against the preserved
  `FINAL_SELECTION_LOCK.txt`. This records the selected recipe; it does not
  independently prove completion or numerical performance.
- All four forward-contract hashes were recomputed from the exact contract
  objects included in the public JSON and matched the recorded hashes.
- Checkpoint and tokenizer hashes/fingerprints were cross-checked across the
  four records. Their underlying binary files were **not rehashed** in this
  publication audit. These entries identify recorded artifacts; they do not
  supply the artifacts or demonstrate an independent reproduction.
- The selected extraction records do not verify immutable upstream model
  revisions: their `tokenizer_commit_hash` fields are empty. Those fields are
  preserved as recorded, and no missing revision is inferred here.

## Shared local-extraction settings

| Item | Recorded value |
|---|---|
| Source model | `state-spaces/mamba3-siso-187m` |
| Tokenizer | `meta-llama/Llama-3.1-8B` |
| Clean-DAPT checkpoint step / scheduler total steps | 25,338 / 25,338 |
| Extraction precision | BF16 |
| Maximum local input length | 512 tokens |
| Maximum content length before the terminal marker | 511 tokens |
| Extraction batch size | 128 |
| Section marker / token ID | `<SEC_SUM>` / 128269 |
| Chunk marker / token ID | `<CHUNK_SUM>` / 128270 |
| Single-summary readout | Terminal-marker hidden state |
| Two-span readout | Mean hidden state of each nonempty contiguous content span |
| Short-input policy | Store nonempty spans only |
| Matched extraction | Both readouts share one backbone forward pass per batch |

The checkpoint state-file SHA256 recorded in all four records is:

```text
79dff12274d168c7fcabcd4aeac13e8d2e3c585bf9bd27a5b8e3ccfdc3054a70
```

The tokenizer-backend SHA256 recorded in all four records is:

```text
c04adb81887edd62258f2468f090cb50cb0a9a3c817548207e63d6d7dcf40c5f
```

The marker-specific tokenizer fingerprints are:

| Local input | Recorded fingerprint |
|---|---|
| Section | `101c84d440ce15938c30936f31b571c1e17d21d9e8756fe8220f4764e68e9031` |
| Chunk | `23e7ea8b1938783cf0710c35e41d71b3c0676417aab987cebbc88a1347b3587d` |

These are the recorded tokenizer fingerprints, not model-weight hashes or
patient/document fingerprints. The tokenizer-backend hash is shared across
both input kinds.

## Experiment identifiers and forward contracts

| Extraction | Recorded joint run ID |
|---|---|
| Training/validation | `12685a53c41f078c1a629d02429b0618904e286cef560874a69d565e99cfe5aa` |
| Test | `56d0ff0dfa20cdcec41c05c1ff484a4b58c8fd51e9e6d7dc7ffd0ace27cf2260` |

Run IDs are experiment-level labels. Their internal generation payloads are
not included, so these IDs cannot be regenerated from this public subset.

| Extraction | Forward-contract SHA256 |
|---|---|
| Training/validation section | `ffc4a6089ac3adc73d77fc119209fb85ea6bf76d8c8cca36b9139f36f99ccbe4` |
| Training/validation chunk | `a031510312243805d2c771ceedbc50aa3db5fb80d84b29c34804656434154330` |
| Test section | `cc9933b491117f3227cad55006ec3e552052c5818d707612fa0703d04d606d22` |
| Test chunk | `8af7401795e8a956f322eed2101ef6b71783202e236cce0828905e6c1ed3a28e` |

The test contracts retain the historical protocol string
`mamba3_joint_s1_s2_len512_train_val`. Their recorded `test_accessed: true`
and the outer `split: test` identify the test extraction. The historical
string and empty tokenizer revision fields are unchanged so that the
original contract hashes remain verifiable.

To verify the included contract hashes, run from the repository root:

```python
import hashlib
import json
from pathlib import Path

records = json.loads(Path("docs/EXPERIMENT_PROVENANCE.json").read_text())
for record in records["extraction_records"]:
    canonical = json.dumps(
        record["forward_contract"], sort_keys=True,
        separators=(",", ":"), ensure_ascii=True,
    ).encode("utf-8")
    assert hashlib.sha256(canonical).hexdigest() == record["forward_contract_sha256"]
print("All four recorded forward contracts match.")
```

This checks the internal consistency of the published metadata. It does not
check private cache contents, prove dataset membership, or reproduce scores.

## Original final-head selection lock

| Setting | Preserved value |
|---|---|
| Selected arm | D: section and chunk two-span means |
| Section / chunk embedding prefix | `section_span2` / `chunk_span2` |
| Supervised-training seeds | 17, 23, 42 |
| Head / loss / sampling | MLP / BCE / natural |
| Epochs / batch size / learning rate | 5 / 64 / 0.001 |
| Auxiliary-loss weight | 0.0 |
| Evaluation splits recorded in the lock | Validation and test |

These settings describe the original final-system recipe, not the later
5/10/20-epoch or two-learning-rate follow-ups. Reproduction commands are
provided separately in [PAPER_REPRODUCTION_COMMANDS.md](PAPER_REPRODUCTION_COMMANDS.md).

## Publication boundary

The public files contain selected settings and experiment-level identifiers.
They omit patient/episode/document identifiers, clinical text, token arrays,
per-example labels and predictions, internal filesystem paths, source-dataset
manifest digests, and operational repair histories. Checkpoint weights and
embedding caches are not included. Access to protected data remains subject
to the source data providers' requirements; see
[DATA_AND_ARTIFACTS.md](../DATA_AND_ARTIFACTS.md).

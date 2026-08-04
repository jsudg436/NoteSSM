#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"
if [[ -n "${NOTESM_ENV_FILE:-}" && -f "${NOTESM_ENV_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${NOTESM_ENV_FILE}"
elif [[ -f "${REPO_ROOT}/env.runpod.sh" ]]; then
  # shellcheck disable=SC1091
  source "${REPO_ROOT}/env.runpod.sh"
fi

mkdir -p logs

NOTESM_DIR="results/frozen/dapt_mamba3_joint_D_final_valtest_3seed_aux00_20260803/artifact"
LONGFORMER_DIR="results/downstream_head_longformer_mean_3seed/longformer_mean_full_len4096_sharded_mlp_natural_bce"
CLINICALMAMBA_DIR="results/downstream_head_clinicalmamba_3seed/clinicalmamba_last_full_len4096_sharded_mlp_natural_bce"
MANIFEST="results/lampr_full_v0_2/benchmark_manifest_no_text.parquet"
OUTPUT_DIR="results/notesm_multispan_external_patient_bootstrap_frozen"

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "ERROR: missing required input: $path" >&2
    exit 1
  fi
  echo "OK: $path"
}

for seed in 17 23 42; do
  require_file "${NOTESM_DIR}/notesm_v2_hybrid_tuned_predictions_seed${seed}.csv"
  require_file "${LONGFORMER_DIR}/notesm_v1_tuned_predictions_seed${seed}.csv"
  require_file "${CLINICALMAMBA_DIR}/notesm_v1_tuned_predictions_seed${seed}.csv"
done
require_file "$MANIFEST"

python scripts/freeze_notesm_multispan_patient_bootstrap.py \
  --project-root "${REPO_ROOT}" \
  --notesm-dir "$NOTESM_DIR" \
  --longformer-dir "$LONGFORMER_DIR" \
  --clinicalmamba-dir "$CLINICALMAMBA_DIR" \
  --manifest "$MANIFEST" \
  --output-dir "$OUTPUT_DIR" \
  --seeds 17 23 42 \
  --n-bootstrap 1000 \
  --bootstrap-seed 20260803 \
  --bootstrap-batch-size 20 \
  --expected-test-n 29460 \
  --expected-test-patients 13454 \
  2>&1 | tee logs/notesm_multispan_patient_bootstrap.log

echo
echo "===== Primary AUPRC confidence intervals ====="
python - <<'PY'
from pathlib import Path
import pandas as pd

path = Path("results/notesm_multispan_external_patient_bootstrap_frozen/patient_cluster_paired_bootstrap_ci.csv")
frame = pd.read_csv(path)
print(
    frame.loc[
        frame["metric"].eq("AUPRC"),
        [
            "left_display_name",
            "right_display_name",
            "observed_delta_left_minus_right",
            "ci_low",
            "ci_high",
            "ci_excludes_zero",
            "bootstrap_probability_left_favorable",
        ],
    ].to_string(index=False)
)
PY

echo
echo "Frozen artifacts: ${REPO_ROOT}/${OUTPUT_DIR}"

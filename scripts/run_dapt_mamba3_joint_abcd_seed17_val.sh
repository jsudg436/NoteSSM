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

JOINT_DIR="${JOINT_DIR:-results/section_embedding_cache/dapt1epoch_mamba3_siso187_section_chunk_joint_s1_s2_full_len512_train_val}"
NOTE_DIR="${NOTE_DIR:-results/embedding_cache/dapt1epoch_mamba3_siso187_note_full_len2048}"
NOTE_CACHE="${NOTE_CACHE:-dapt1epoch_mamba3_siso187_note_full_len2048}"
OUT_ROOT="${OUT_ROOT:-results/downstream_notesm_v2_mamba3_dapt_joint_abcd_seed17_val}"
STATUS="${STATUS:-logs/dapt_mamba3_joint_abcd_seed17_val.status}"
SEED=17
AUX_LOSS_WEIGHT="${AUX_LOSS_WEIGHT:-0.0}"

write_status() {
  local arm="$1"
  local state="$2"
  local detail="${3:-}"
  {
    echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "arm=${arm}"
    echo "state=${state}"
    echo "detail=${detail}"
    echo "joint_dir=${JOINT_DIR}"
    echo "note_dir=${NOTE_DIR}"
    echo "note_cache=${NOTE_CACHE}"
    echo "out_root=${OUT_ROOT}"
    echo "seed=${SEED}"
    echo "aux_loss_weight=${AUX_LOSS_WEIGHT}"
  } > "${STATUS}"
}

require_joint_inputs() {
  python scripts/validate_mamba3_joint_s1_s2_artifacts.py \
    --joint-cache-dir "${JOINT_DIR}" \
    --output "${JOINT_DIR}/joint_s1_s2_artifact_validation_summary.csv" \
    > logs/dapt_mamba3_joint_abcd_artifact_validation.log 2>&1
  for prefix in section section_span2 chunk chunk_span2; do
    if ! compgen -G "${JOINT_DIR}/${prefix}_embeddings_shard_*.npy" > /dev/null; then
      echo "Missing ${prefix} embeddings under ${JOINT_DIR}" >&2
      exit 1
    fi
    if ! compgen -G "${JOINT_DIR}/${prefix}_embedding_metadata_shard_*.parquet" > /dev/null; then
      echo "Missing ${prefix} metadata under ${JOINT_DIR}" >&2
      exit 1
    fi
  done
  test -f "${NOTE_DIR}/${NOTE_CACHE}.npy"
  test -f "${NOTE_DIR}/${NOTE_CACHE}_metadata.parquet"
}

run_arm() {
  local arm="$1"
  local section_prefix="$2"
  local chunk_prefix="$3"
  local out_dir="${OUT_ROOT}/arm_${arm}_${section_prefix}_${chunk_prefix}"
  local log_path="logs/dapt_mamba3_joint_abcd_seed17_arm_${arm}.log"
  local metrics="${out_dir}/notesm_v2_hybrid_cached_embedding_metrics_seed${SEED}.csv"
  local preds="${out_dir}/notesm_v2_hybrid_cached_predictions_seed${SEED}.csv"

  if [ -f "${metrics}" ] && [ -f "${preds}" ]; then
    write_status "${arm}" "SKIPPED" "already complete section=${section_prefix} chunk=${chunk_prefix}"
    echo "Skipping arm ${arm}; already complete."
    return
  fi

  write_status "${arm}" "RUNNING_TRAIN" "section=${section_prefix} chunk=${chunk_prefix}"
  echo "Running arm ${arm}: section=${section_prefix}, chunk=${chunk_prefix}"

  CUDA_VISIBLE_DEVICES=0 python scripts/train_downstream.py \
    --benchmark-root results/lampr_full_v0_2 \
    --model notesm_v2_hybrid \
    --section-embedding-dir "${JOINT_DIR}" \
    --chunk-embedding-dir "${JOINT_DIR}" \
    --section-embedding-prefix "${section_prefix}" \
    --chunk-embedding-prefix "${chunk_prefix}" \
    --embedding-cache-dir "${NOTE_DIR}" \
    --cache-name "${NOTE_CACHE}" \
    --output-dir "${out_dir}" \
    --hybrid-mode three_view_concat \
    --head mlp \
    --sampling natural \
    --loss bce \
    --seed "${SEED}" \
    --epochs 5 \
    --batch-size 64 \
    --lr 0.001 \
    --device cuda \
    --aux-loss-weight "${AUX_LOSS_WEIGHT}" \
    --eval-splits val \
    --skip-test \
    --architecture-screening-only \
    --run-training \
    > "${log_path}" 2>&1

  write_status "${arm}" "DONE" "section=${section_prefix} chunk=${chunk_prefix}"
}

trap 'write_status "${CURRENT_ARM:-unknown}" "FAILED" "line=${LINENO}"' ERR

require_joint_inputs

CURRENT_ARM=A
run_arm A section chunk

CURRENT_ARM=B
run_arm B section_span2 chunk

CURRENT_ARM=C
run_arm C section chunk_span2

CURRENT_ARM=D
run_arm D section_span2 chunk_span2

write_status summary RUNNING "collecting validation-only A/B/C/D metrics"
python - "${OUT_ROOT}" <<'PY'
import sys
from pathlib import Path

import pandas as pd

out_root = Path(sys.argv[1])
specs = [
    ("A", "section", "chunk"),
    ("B", "section_span2", "chunk"),
    ("C", "section", "chunk_span2"),
    ("D", "section_span2", "chunk_span2"),
]
rows = []
for arm, section_prefix, chunk_prefix in specs:
    out_dir = out_root / f"arm_{arm}_{section_prefix}_{chunk_prefix}"
    metrics_path = out_dir / "notesm_v2_hybrid_cached_embedding_metrics_seed17.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    metrics = pd.read_csv(metrics_path)
    if set(metrics["split"].astype(str)) != {"val"}:
        raise ValueError(f"{metrics_path} must contain validation-only rows, got {sorted(set(metrics['split'].astype(str)))}")
    row = metrics.iloc[0].to_dict()
    row.update({"arm": arm, "section_embedding_prefix": section_prefix, "chunk_embedding_prefix": chunk_prefix})
    rows.append(row)
summary = pd.DataFrame(rows)
summary = summary.sort_values("AUPRC", ascending=False)
summary.to_csv(out_root / "joint_abcd_seed17_validation_summary.csv", index=False)
print(summary.to_string(index=False))
winner = summary.iloc[0]
(out_root / "joint_abcd_seed17_winner.txt").write_text(
    f"arm={winner['arm']}\n"
    f"section_embedding_prefix={winner['section_embedding_prefix']}\n"
    f"chunk_embedding_prefix={winner['chunk_embedding_prefix']}\n"
    f"selection_metric=validation_AUPRC\n"
    f"validation_AUPRC={winner['AUPRC']}\n",
    encoding="utf-8",
)
PY

WINNER_ARM="$(grep '^arm=' "${OUT_ROOT}/joint_abcd_seed17_winner.txt" | cut -d= -f2)"
WINNER_SECTION="$(grep '^section_embedding_prefix=' "${OUT_ROOT}/joint_abcd_seed17_winner.txt" | cut -d= -f2)"
WINNER_CHUNK="$(grep '^chunk_embedding_prefix=' "${OUT_ROOT}/joint_abcd_seed17_winner.txt" | cut -d= -f2)"
WINNER_OUT="${OUT_ROOT}/arm_${WINNER_ARM}_${WINNER_SECTION}_${WINNER_CHUNK}"

write_status threshold RUNNING "tuning F1 threshold for validation-AUPRC winner arm=${WINNER_ARM}"
python scripts/tune_validation_threshold.py \
  --downstream-dir "${WINNER_OUT}" \
  --model notesm_v2_hybrid \
  --seed "${SEED}" \
  --objective f1 \
  --output "${OUT_ROOT}/joint_abcd_seed17_winner_threshold_summary.csv" \
  > logs/dapt_mamba3_joint_abcd_seed17_winner_threshold.log 2>&1

write_status all DONE "A/B/C/D seed17 validation-only completed"
echo "DONE DAPT Mamba3 joint A/B/C/D seed17 validation-only"

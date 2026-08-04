#!/usr/bin/env bash
set -euo pipefail

BENCHMARK_ROOT="${BENCHMARK_ROOT:-results/lampr_full_v0_2}"
SECTION_EMBEDDING_DIR="${SECTION_EMBEDDING_DIR:-results/section_embedding_cache/dapt1epoch_mamba2_section_full}"
NOTE_EMBEDDING_DIR="${NOTE_EMBEDDING_DIR:-results/embedding_cache/dapt1epoch_mamba2_last_full_len4096}"
CACHE_NAME="${CACHE_NAME:-dapt1epoch_mamba2_last_full_len4096}"
if [[ -z "${OUTPUT_ROOT_TEMPLATE:-}" ]]; then
  OUTPUT_ROOT_TEMPLATE='results/downstream_notesm_v2_v2_1_val_screen_seed{seed}_aux01'
fi
if [[ -z "${SUMMARY_ROOT_TEMPLATE:-}" ]]; then
  SUMMARY_ROOT_TEMPLATE='results/notesm_v2_v2_1_val_screen_seed{seed}_aux01'
fi
AGGREGATE_ROOT="${AGGREGATE_ROOT:-results/notesm_v2_v2_1_val_screen_3seed_concat_aux01}"
AUX_CHECK_ROOT="${AUX_CHECK_ROOT:-results/notesm_v2_v2_1_aux01_3seed_val_check}"
SEEDS="${SEEDS:-23 42}"
AGGREGATE_SEEDS="${AGGREGATE_SEEDS:-17 23 42}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-0.001}"
DEVICE="${DEVICE:-cuda}"

mkdir -p logs
LOG_FILE="logs/notesm_v2_v2_1_three_view_concat_aux01_seed23_42_val_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "log_file=${LOG_FILE}"
echo "benchmark_root=${BENCHMARK_ROOT}"
echo "section_embedding_dir=${SECTION_EMBEDDING_DIR}"
echo "note_embedding_dir=${NOTE_EMBEDDING_DIR}"
echo "cache_name=${CACHE_NAME}"
echo "seeds=${SEEDS}"
echo "aggregate_seeds=${AGGREGATE_SEEDS}"
echo "batch_size=${BATCH_SIZE}"
echo "epochs=${EPOCHS}"
echo "lr=${LR}"
echo "device=${DEVICE}"
echo "aux_loss_weight=0.1"

for required in "${BENCHMARK_ROOT}" "${SECTION_EMBEDDING_DIR}" "${NOTE_EMBEDDING_DIR}"; do
  if [[ ! -e "${required}" ]]; then
    echo "ERROR: missing required artifact: ${required}" >&2
    exit 1
  fi
done

for seed in ${SEEDS}; do
  output_root="$(printf '%s' "${OUTPUT_ROOT_TEMPLATE}" | sed "s/{seed}/${seed}/g")"
  summary_root="$(printf '%s' "${SUMMARY_ROOT_TEMPLATE}" | sed "s/{seed}/${seed}/g")"
  mkdir -p "${output_root}" "${summary_root}"
  echo "Running three_view_concat aux=0.1 validation-only screen for seed=${seed}"
  python scripts/train_downstream.py \
    --benchmark-root "${BENCHMARK_ROOT}" \
    --model notesm_v2_hybrid \
    --section-embedding-dir "${SECTION_EMBEDDING_DIR}" \
    --embedding-cache-dir "${NOTE_EMBEDDING_DIR}" \
    --cache-name "${CACHE_NAME}" \
    --output-dir "${output_root}/three_view_concat" \
    --hybrid-mode three_view_concat \
    --head mlp \
    --sampling natural \
    --loss bce \
    --seed "${seed}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --device "${DEVICE}" \
    --aux-loss-weight 0.1 \
    --eval-splits val \
    --architecture-screening-only \
    --run-training

  python scripts/summarize_notesm_v2_v2_1_val_screen.py \
    --screen-root "${output_root}" \
    --output-dir "${summary_root}" \
    --seed "${seed}" \
    --modes three_view_concat
done

python scripts/aggregate_notesm_v2_v2_1_val_screen_seeds.py \
  --summary-root-template "${SUMMARY_ROOT_TEMPLATE}" \
  --seeds ${AGGREGATE_SEEDS} \
  --setting three_view_concat \
  --output-dir "${AGGREGATE_ROOT}"

python scripts/summarize_notesm_v2_v2_1_aux01_3seed.py \
  --aux01-aggregate "${AGGREGATE_ROOT}/notesm_v2_v2_1_val_aggregate.csv" \
  --aux00-locked-summary results/notesm_v2_v2_1_val_screen_3seed_concat_locked/notesm_v2_v2_1_three_view_concat_val_locked_summary.csv \
  --output-dir "${AUX_CHECK_ROOT}"

echo "Aux=0.1 seed23/42 validation-only repeat complete. Review ${AUX_CHECK_ROOT}."

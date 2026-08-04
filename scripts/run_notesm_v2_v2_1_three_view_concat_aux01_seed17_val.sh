#!/usr/bin/env bash
set -euo pipefail

BENCHMARK_ROOT="${BENCHMARK_ROOT:-results/lampr_full_v0_2}"
SECTION_EMBEDDING_DIR="${SECTION_EMBEDDING_DIR:-results/section_embedding_cache/dapt1epoch_mamba2_section_full}"
NOTE_EMBEDDING_DIR="${NOTE_EMBEDDING_DIR:-results/embedding_cache/dapt1epoch_mamba2_last_full_len4096}"
CACHE_NAME="${CACHE_NAME:-dapt1epoch_mamba2_last_full_len4096}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results/downstream_notesm_v2_v2_1_val_screen_seed17_aux01}"
SUMMARY_ROOT="${SUMMARY_ROOT:-results/notesm_v2_v2_1_val_screen_seed17_aux01}"
AUX_CHECK_ROOT="${AUX_CHECK_ROOT:-results/notesm_v2_v2_1_aux01_seed17_val_check}"
SEED="${SEED:-17}"
BATCH_SIZE="${BATCH_SIZE:-64}"
EPOCHS="${EPOCHS:-5}"
LR="${LR:-0.001}"
DEVICE="${DEVICE:-cuda}"

mkdir -p logs "${OUTPUT_ROOT}" "${SUMMARY_ROOT}" "${AUX_CHECK_ROOT}"
LOG_FILE="logs/notesm_v2_v2_1_three_view_concat_aux01_seed17_val_$(date -u +%Y%m%dT%H%M%SZ).log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "log_file=${LOG_FILE}"
echo "benchmark_root=${BENCHMARK_ROOT}"
echo "section_embedding_dir=${SECTION_EMBEDDING_DIR}"
echo "note_embedding_dir=${NOTE_EMBEDDING_DIR}"
echo "cache_name=${CACHE_NAME}"
echo "output_root=${OUTPUT_ROOT}"
echo "summary_root=${SUMMARY_ROOT}"
echo "aux_check_root=${AUX_CHECK_ROOT}"
echo "seed=${SEED}"
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

python scripts/train_downstream.py \
  --benchmark-root "${BENCHMARK_ROOT}" \
  --model notesm_v2_hybrid \
  --section-embedding-dir "${SECTION_EMBEDDING_DIR}" \
  --embedding-cache-dir "${NOTE_EMBEDDING_DIR}" \
  --cache-name "${CACHE_NAME}" \
  --output-dir "${OUTPUT_ROOT}/three_view_concat" \
  --hybrid-mode three_view_concat \
  --head mlp \
  --sampling natural \
  --loss bce \
  --seed "${SEED}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --device "${DEVICE}" \
  --aux-loss-weight 0.1 \
  --eval-splits val \
  --architecture-screening-only \
  --run-training

python scripts/summarize_notesm_v2_v2_1_val_screen.py \
  --screen-root "${OUTPUT_ROOT}" \
  --output-dir "${SUMMARY_ROOT}" \
  --seed "${SEED}" \
  --modes three_view_concat

python scripts/summarize_notesm_v2_v2_1_aux01_seed17.py \
  --aux01-summary "${SUMMARY_ROOT}/notesm_v2_v2_1_val_main_table.csv" \
  --aux00-seed17-summary results/notesm_v2_v2_1_val_screen_seed17/notesm_v2_v2_1_val_main_table.csv \
  --aux00-locked-summary results/notesm_v2_v2_1_val_screen_3seed_concat_locked/notesm_v2_v2_1_three_view_concat_val_locked_summary.csv \
  --output-dir "${AUX_CHECK_ROOT}"

echo "Aux=0.1 seed17 validation-only check complete. Review ${AUX_CHECK_ROOT}."

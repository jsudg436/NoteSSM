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

RUN_TESTS="${RUN_TESTS:-1}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_DETERMINISM="${RUN_DETERMINISM:-1}"
RUN_BENCHMARK="${RUN_BENCHMARK:-1}"
RUN_FULL="${RUN_FULL:-1}"
RUN_VALIDATE="${RUN_VALIDATE:-1}"
RUN_PARITY="${RUN_PARITY:-1}"

CKPT="checkpoints/dapt_mamba3_siso187_lampr_v0_2_len2048_1epoch_lr1e4_min2_clean/checkpoint-step-25338"
SOURCE_CACHE="results/notesm_v2_section_cache/lampr_v0_2_full"
S1_CACHE="${S1_CACHE:-}"
FINAL_OUT="results/section_embedding_cache/dapt1epoch_mamba3_siso187_section_chunk_joint_s1_s2_full_len512_train_val"
SMOKE_OUT="results/section_embedding_cache/dapt1epoch_mamba3_siso187_section_chunk_joint_s1_s2_smoke_len512_train_val"
BENCH_OUT="results/section_embedding_cache/dapt1epoch_mamba3_siso187_section_chunk_joint_s1_s2_benchmark_len512_train_val"
STATUS="logs/dapt_mamba3_span2_len512_train_val.status"

write_status() {
  local stage="$1"
  local state="$2"
  local detail="${3:-}"
  {
    echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "stage=${stage}"
    echo "state=${state}"
    echo "detail=${detail}"
    echo "checkpoint=${CKPT}"
    echo "source_cache=${SOURCE_CACHE}"
    echo "s1_cache=${S1_CACHE}"
    echo "final_out=${FINAL_OUT}"
  } > "${STATUS}"
}

best_batch_size() {
  local csv_path="$1"
  python - "$csv_path" <<'PY'
import sys
import pandas as pd

path = sys.argv[1]
df = pd.read_csv(path)
ok = df[df["status"].eq("ok") & df["rows_per_second"].notna()].copy()
if ok.empty:
    print(64)
else:
    print(int(ok.sort_values("rows_per_second", ascending=False).iloc[0]["batch_size"]))
PY
}

move_stale_output_if_needed() {
  local out_dir="$1"
  local label="$2"
  if [ ! -d "${out_dir}" ]; then
    return
  fi
  python - "${out_dir}" "${label}" <<'PY'
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

out_dir = Path(sys.argv[1])
label = sys.argv[2]
completion_paths = sorted(out_dir.glob("*_embedding_shard_*.complete.json"))
data_paths = sorted(out_dir.glob("*_embeddings_shard_*.npy")) + sorted(out_dir.glob("*_embedding_metadata_shard_*.parquet"))
if not completion_paths and not data_paths:
    raise SystemExit(0)

stale = []
joint_completion_paths = sorted(out_dir.glob("*_joint_embedding_shard_*.complete.json"))
non_joint_completion_paths = [path for path in completion_paths if "_joint_embedding_shard_" not in path.name]
if data_paths and not joint_completion_paths:
    stale.append(("partial_data_without_joint_completion", None))
if non_joint_completion_paths:
    stale.append(("legacy_non_joint_completion", "present"))
for path in completion_paths:
    try:
        schema = json.loads(path.read_text(encoding="utf-8")).get("schema_version")
    except Exception:
        schema = None
    if schema != 5:
        stale.append((path.name, schema))

if not stale:
    raise SystemExit(0)

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
target_root = Path("results/_stale_mamba3_span2_schema_mismatch")
target_root.mkdir(parents=True, exist_ok=True)
target = target_root / f"{out_dir.name}_{label}_schema_mismatch_{stamp}"
if target.exists():
    raise SystemExit(f"Refusing to overwrite stale target: {target}")
shutil.move(str(out_dir), str(target))
print(f"Moved stale {label} output to {target}")
print("First mismatched completions:", stale[:5])
PY
}

trap 'write_status "${CURRENT_STAGE:-unknown}" "FAILED" "line=${LINENO}"' ERR

CURRENT_STAGE="preflight_stale_schema"
write_status "${CURRENT_STAGE}" "RUNNING" "checking non-schema-v5 or partial resume outputs"
move_stale_output_if_needed "${SMOKE_OUT}" "smoke"
move_stale_output_if_needed "${FINAL_OUT}" "full"
write_status "${CURRENT_STAGE}" "DONE" "non-schema-v5 or partial outputs moved if present"

if [ "${RUN_TESTS}" = "1" ]; then
  CURRENT_STAGE="tests"
  write_status "${CURRENT_STAGE}" "RUNNING" "pytest extractor and downstream mixed-cache tests"
  python -m pytest -q \
    tests/test_extract_mamba3_joint_s1_span2.py \
    tests/test_downstream_training.py \
    > logs/dapt_mamba3_span2_tests.log 2>&1
  write_status "${CURRENT_STAGE}" "DONE" "pytest passed"
fi

if [ "${RUN_SMOKE}" = "1" ]; then
  for KIND in section chunk; do
    CURRENT_STAGE="smoke_${KIND}"
    write_status "${CURRENT_STAGE}" "RUNNING" "32-example ${KIND} smoke"
    CUDA_VISIBLE_DEVICES=0 python scripts/extract_mamba3_joint_s1_span2.py \
      --section-cache-dir "${SOURCE_CACHE}" \
      --s1-cache-dir "${S1_CACHE}" \
      --output-dir "${SMOKE_OUT}" \
      --input-kind "${KIND}" \
      --split train_val \
      --limit-examples 32 \
      --model-checkpoint "${CKPT}" \
      --max-length 512 \
      --batch-size 16 \
      --shard-size 32 \
      --write-s1 \
      --no-s1-audit \
      --resume \
      --device cuda \
      > "logs/dapt_mamba3_span2_${KIND}_smoke.log" 2>&1
    write_status "${CURRENT_STAGE}" "DONE" "32-example ${KIND} smoke passed"
  done
fi

if [ "${RUN_DETERMINISM}" = "1" ]; then
  DET_ROOT="results/section_embedding_cache/dapt1epoch_mamba3_siso187_section_chunk_joint_s1_s2_determinism_$(date -u +%Y%m%dT%H%M%SZ)"
  for KIND in section chunk; do
    CURRENT_STAGE="determinism_${KIND}"
    write_status "${CURRENT_STAGE}" "RUNNING" "fresh-process determinism ${KIND}"
    for RUN_ID in a b; do
      CUDA_VISIBLE_DEVICES=0 python scripts/extract_mamba3_joint_s1_span2.py \
        --section-cache-dir "${SOURCE_CACHE}" \
        --s1-cache-dir "${S1_CACHE}" \
        --output-dir "${DET_ROOT}/${KIND}_${RUN_ID}" \
        --input-kind "${KIND}" \
        --split train_val \
        --limit-examples 32 \
        --model-checkpoint "${CKPT}" \
        --max-length 512 \
        --batch-size 16 \
        --shard-size 32 \
        --write-s1 \
        --no-s1-audit \
        --device cuda \
        > "logs/dapt_mamba3_span2_${KIND}_determinism_${RUN_ID}.log" 2>&1
    done
    for PREFIX in "${KIND}" "${KIND}_span2"; do
      python - "${DET_ROOT}/${KIND}_a" "${DET_ROOT}/${KIND}_b" "${PREFIX}" \
      > "logs/dapt_mamba3_span2_${PREFIX}_determinism_compare.log" 2>&1 <<'PY'
import sys
import importlib.util
from pathlib import Path

left = Path(sys.argv[1])
right = Path(sys.argv[2])
prefix = sys.argv[3]

script = Path("scripts/extract_mamba3_joint_s1_span2.py").resolve()
spec = importlib.util.spec_from_file_location("extract_mamba3_joint_s1_span2_compare", script)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)

result = module.compare_sharded_embedding_dirs(left, right, prefix, max_abs_tol=1e-6)
print(
    f"{prefix} fresh-process determinism passed "
    f"shards={result['n_shards']} rows={result['n_rows']} max_abs={result['max_abs']:.6g}"
)
PY
    done
    write_status "${CURRENT_STAGE}" "DONE" "fresh-process determinism ${KIND} passed"
  done
fi

if [ "${RUN_BENCHMARK}" = "1" ]; then
  for KIND in section chunk; do
    CURRENT_STAGE="benchmark_${KIND}"
    write_status "${CURRENT_STAGE}" "RUNNING" "benchmarking ${KIND} batch sizes 32 64 128"
    CUDA_VISIBLE_DEVICES=0 python scripts/extract_mamba3_joint_s1_span2.py \
      --section-cache-dir "${SOURCE_CACHE}" \
      --s1-cache-dir "${S1_CACHE}" \
      --output-dir "${BENCH_OUT}" \
      --input-kind "${KIND}" \
      --split train_val \
      --limit-rows 10000 \
      --model-checkpoint "${CKPT}" \
      --max-length 512 \
      --batch-size 64 \
      --shard-size 2048 \
      --benchmark-batch-sizes 32 64 128 \
      --benchmark-warmup-rows 512 \
      --benchmark-repeats 3 \
      --benchmark-only \
      --write-s1 \
      --no-s1-audit \
      --device cuda \
      > "logs/dapt_mamba3_span2_${KIND}_benchmark.log" 2>&1
    write_status "${CURRENT_STAGE}" "DONE" "benchmark ${KIND} completed"
  done
fi

SECTION_BATCH=64
CHUNK_BATCH=64
if [ -f "${BENCH_OUT}/section_batch_benchmark.csv" ]; then
  SECTION_BATCH="$(best_batch_size "${BENCH_OUT}/section_batch_benchmark.csv")"
fi
if [ -f "${BENCH_OUT}/chunk_batch_benchmark.csv" ]; then
  CHUNK_BATCH="$(best_batch_size "${BENCH_OUT}/chunk_batch_benchmark.csv")"
fi

if [ "${RUN_FULL}" = "1" ]; then
  CURRENT_STAGE="full_section"
  write_status "${CURRENT_STAGE}" "RUNNING" "full train_val section joint S1/S2 extraction batch=${SECTION_BATCH}"
  CUDA_VISIBLE_DEVICES=0 python scripts/extract_mamba3_joint_s1_span2.py \
    --section-cache-dir "${SOURCE_CACHE}" \
    --s1-cache-dir "${S1_CACHE}" \
    --output-dir "${FINAL_OUT}" \
    --input-kind section \
    --split train_val \
    --model-checkpoint "${CKPT}" \
    --max-length 512 \
    --batch-size "${SECTION_BATCH}" \
    --shard-size 2048 \
    --write-s1 \
    --no-s1-audit \
    --resume \
    --device cuda \
    > logs/dapt_mamba3_span2_section_full.log 2>&1
  write_status "${CURRENT_STAGE}" "DONE" "full section joint S1/S2 extraction completed"

  CURRENT_STAGE="full_chunk"
  write_status "${CURRENT_STAGE}" "RUNNING" "full train_val chunk joint S1/S2 extraction batch=${CHUNK_BATCH}"
  CUDA_VISIBLE_DEVICES=0 python scripts/extract_mamba3_joint_s1_span2.py \
    --section-cache-dir "${SOURCE_CACHE}" \
    --s1-cache-dir "${S1_CACHE}" \
    --output-dir "${FINAL_OUT}" \
    --input-kind chunk \
    --split train_val \
    --model-checkpoint "${CKPT}" \
    --max-length 512 \
    --batch-size "${CHUNK_BATCH}" \
    --shard-size 2048 \
    --write-s1 \
    --no-s1-audit \
    --resume \
    --device cuda \
    > logs/dapt_mamba3_span2_chunk_full.log 2>&1
  write_status "${CURRENT_STAGE}" "DONE" "full chunk joint S1/S2 extraction completed"
fi

if [ "${RUN_VALIDATE}" = "1" ]; then
  CURRENT_STAGE="joint_artifact_validation"
  write_status "${CURRENT_STAGE}" "RUNNING" "validating joint S1/S2 completions and hashes"
  python scripts/validate_mamba3_joint_s1_s2_artifacts.py \
    --joint-cache-dir "${FINAL_OUT}" \
    --output "${FINAL_OUT}/joint_s1_s2_artifact_validation_summary.csv" \
    > logs/dapt_mamba3_joint_artifact_validation.log 2>&1
  write_status "${CURRENT_STAGE}" "DONE" "joint S1/S2 artifact validation completed"
fi

if [ "${RUN_PARITY}" = "1" ]; then
  CURRENT_STAGE="old_s1_parity"
  write_status "${CURRENT_STAGE}" "RUNNING" "diagnostic old clean S1 vs joint S1 parity"
  python scripts/diagnose_mamba3_joint_s1_parity.py \
    --old-s1-cache-dir results/section_embedding_cache/dapt1epoch_mamba3_siso187_section_chunk_full_len512 \
    --joint-cache-dir "${FINAL_OUT}" \
    --output-dir "${FINAL_OUT}" \
    > logs/dapt_mamba3_joint_old_s1_parity.log 2>&1
  write_status "${CURRENT_STAGE}" "DONE" "diagnostic old clean S1 vs joint S1 parity completed"
fi

write_status "all" "DONE" "Joint S1/S2 train_val extraction workflow completed; section_batch=${SECTION_BATCH}; chunk_batch=${CHUNK_BATCH}"
echo "DONE dapt Mamba3 joint S1/S2 span_mean_2 len512 train_val workflow"

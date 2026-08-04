#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

TRAINVAL_JOINT="${TRAINVAL_JOINT:-results/section_embedding_cache/dapt1epoch_mamba3_siso187_section_chunk_joint_s1_s2_full_len512_train_val}"
TEST_JOINT="${TEST_JOINT:-results/section_embedding_cache/dapt1epoch_mamba3_siso187_section_chunk_joint_s1_s2_test_len512}"
NOTE_DIR="${NOTE_DIR:-results/embedding_cache/dapt1epoch_mamba3_siso187_note_full_len2048}"
NOTE_CACHE="${NOTE_CACHE:-dapt1epoch_mamba3_siso187_note_full_len2048}"
OUT="${OUT:-results/downstream_notesm_v2_mamba3_dapt_joint_final_test/D_section_span2_chunk_span2_val_test}"

python scripts/validate_mamba3_joint_s1_s2_artifacts.py \
  --joint-cache-dir "$TRAINVAL_JOINT" \
  --output "$TRAINVAL_JOINT/joint_s1_s2_artifact_validation_summary.csv"

python scripts/validate_mamba3_joint_s1_s2_artifacts.py \
  --joint-cache-dir "$TEST_JOINT" \
  --allow-test-accessed \
  --output "$TEST_JOINT/joint_s1_s2_artifact_validation_summary.csv"

for seed in 17 23 42; do
  metrics="$OUT/notesm_v2_hybrid_threshold_tuned_metrics_seed${seed}.csv"
  preds="$OUT/notesm_v2_hybrid_tuned_predictions_seed${seed}.csv"
  if [[ -f "$metrics" && -f "$preds" ]]; then
    echo "Skipping seed=${seed}; final outputs already exist."
    continue
  fi

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python scripts/train_downstream.py \
    --benchmark-root results/lampr_full_v0_2 \
    --model notesm_v2_hybrid \
    --section-embedding-dir "$TRAINVAL_JOINT" \
    --chunk-embedding-dir "$TRAINVAL_JOINT" \
    --section-embedding-extra-dir "$TEST_JOINT" \
    --chunk-embedding-extra-dir "$TEST_JOINT" \
    --section-embedding-prefix section_span2 \
    --chunk-embedding-prefix chunk_span2 \
    --embedding-cache-dir "$NOTE_DIR" \
    --cache-name "$NOTE_CACHE" \
    --output-dir "$OUT" \
    --hybrid-mode three_view_concat \
    --head mlp \
    --sampling natural \
    --loss bce \
    --seed "$seed" \
    --epochs 5 \
    --batch-size 64 \
    --lr 0.001 \
    --device cuda \
    --aux-loss-weight 0.0 \
    --eval-splits val,test \
    --run-training \
    > "logs/dapt_mamba3_joint_D_final_valtest_seed${seed}.log" 2>&1

  python scripts/tune_validation_threshold.py \
    --downstream-dir "$OUT" \
    --model notesm_v2_hybrid \
    --seed "$seed" \
    --objective f1 \
    >> "logs/dapt_mamba3_joint_D_final_valtest_seed${seed}.log" 2>&1
done

OUT="$OUT" python - <<'PY'
import os
from pathlib import Path
import pandas as pd

out = Path(os.environ["OUT"])
rows = []
for seed in [17, 23, 42]:
    frame = pd.read_csv(out / f"notesm_v2_hybrid_threshold_tuned_metrics_seed{seed}.csv")
    frame["seed"] = seed
    rows.append(frame)
all_df = pd.concat(rows, ignore_index=True)
all_df.to_csv(out / "D_final_valtest_threshold_tuned_metrics_all_seeds.csv", index=False)
metrics = ["Accuracy", "Precision", "Recall", "F1", "AUROC", "AUPRC", "Brier", "ECE"]
summary_rows = []
for split, group in all_df.groupby("split"):
    row = {"split": split, "n_seeds": group["seed"].nunique()}
    for metric in metrics:
        row[f"{metric}_mean"] = group[metric].mean()
        row[f"{metric}_std"] = group[metric].std(ddof=1)
    summary_rows.append(row)
pd.DataFrame(summary_rows).to_csv(out / "D_final_valtest_3seed_aggregate.csv", index=False)
print(pd.DataFrame(summary_rows).to_string(index=False))
PY

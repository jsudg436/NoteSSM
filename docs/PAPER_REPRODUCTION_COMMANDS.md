# Paper Reproduction Commands

These commands document the locked NoteSSM paper setting. They assume the
LAMP-R benchmark, train-only DAPT token shards, and protected source data are
available locally. Protected clinical data, checkpoints, embedding caches, and
prediction files are not included in this repository.

## 1. Clean Mamba3 DAPT

```bash
python scripts/train_dapt_from_shards.py \
  --shard-dir results/dapt_token_shards/lampr_v0_2_mamba3_siso187_len2048 \
  --model-id state-spaces/mamba3-siso-187m \
  --tokenizer-id meta-llama/Llama-3.1-8B \
  --output-dir checkpoints/dapt_mamba3_siso187_lampr_v0_2_len2048_1epoch_lr1e4_min2_clean \
  --scheduler-total-steps 25338 \
  --per-device-batch-size 2 \
  --gradient-accumulation-steps 8 \
  --learning-rate 1e-4 \
  --weight-decay 0.1 \
  --warmup-ratio 0.03 \
  --max-grad-norm 0.5 \
  --bf16 \
  --gradient-checkpointing \
  --min-train-block-tokens 2 \
  --save-steps 5000 \
  --logging-steps 50 \
  --checkpoint-format state_dict \
  --allow-full-epoch \
  --confirm-full-dapt I_UNDERSTAND_FULL_DAPT
```

Full-epoch resume is intentionally disabled unless a dataloader cursor is
implemented. Restart full DAPT from scratch for paper-grade reproduction.

## 2. Note-Level Embeddings

```bash
CKPT=checkpoints/dapt_mamba3_siso187_lampr_v0_2_len2048_1epoch_lr1e4_min2_clean/checkpoint-step-25338

python scripts/extract_embeddings.py \
  --benchmark-root results/lampr_full_v0_2 \
  --split all \
  --output-dir results/embedding_cache/dapt1epoch_mamba3_siso187_note_full_len2048 \
  --cache-name dapt1epoch_mamba3_siso187_note_full_len2048 \
  --run-backbone \
  --backbone mamba3_siso_187m \
  --model-checkpoint "$CKPT" \
  --input-kind note \
  --pooling last_token \
  --max-length 2048 \
  --batch-size 16 \
  --device cuda
```

## 3. Train/Validation Joint S1/S2 Local Embeddings

```bash
bash scripts/run_dapt_mamba3_span2_len512_train_val.sh
```

This writes a joint cache with S1 prefixes `section`/`chunk` and S2 prefixes
`section_span2`/`chunk_span2`.

## 4. Locked Test Joint S1/S2 Local Embeddings

Only run this after the validation-selected model is locked.

```bash
CKPT=checkpoints/dapt_mamba3_siso187_lampr_v0_2_len2048_1epoch_lr1e4_min2_clean/checkpoint-step-25338
TEST_JOINT=results/section_embedding_cache/dapt1epoch_mamba3_siso187_section_chunk_joint_s1_s2_test_len512

python scripts/extract_mamba3_joint_s1_span2.py \
  --section-cache-dir results/notesm_v2_section_cache/lampr_v0_2_full \
  --output-dir "$TEST_JOINT" \
  --input-kind section \
  --split test \
  --allow-test-split \
  --model-checkpoint "$CKPT" \
  --max-length 512 \
  --batch-size 128 \
  --shard-size 2048 \
  --write-s1 \
  --no-s1-audit \
  --resume \
  --device cuda

python scripts/extract_mamba3_joint_s1_span2.py \
  --section-cache-dir results/notesm_v2_section_cache/lampr_v0_2_full \
  --output-dir "$TEST_JOINT" \
  --input-kind chunk \
  --split test \
  --allow-test-split \
  --model-checkpoint "$CKPT" \
  --max-length 512 \
  --batch-size 128 \
  --shard-size 2048 \
  --write-s1 \
  --no-s1-audit \
  --resume \
  --device cuda
```

## 5. Final D S2/S2 Val/Test Three-Seed Evaluation

```bash
bash scripts/run_dapt_mamba3_joint_D_final_valtest_3seed.sh
```

The locked final arm uses:

- section prefix: `section_span2`
- chunk prefix: `chunk_span2`
- note cache: clean Mamba3 note-level max length 2,048
- local unit max length: 512
- hybrid mode: `three_view_concat`
- loss/sampling/head: natural BCE with MLP head
- aux loss weight: `0.0`

## 6. Patient-Cluster Bootstrap

```bash
bash scripts/run_notesm_multispan_patient_bootstrap.sh
```

The bootstrap samples patients with replacement and applies the same patient
cluster resample across models and matched seeds within each replicate.

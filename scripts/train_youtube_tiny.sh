#!/usr/bin/env bash
set -euo pipefail

# Tiny smoke-test training script (around ~100 samples)
# Example:
# DATA_ROOT=playground/data \
# TRAIN_DATA=playground/data/processed/train_sft_100.json \
# EVAL_DATA=playground/data/processed/test_sft_100.json \
# MODEL_ID=llava-1.5-7b \
# RUN_ID=youtubefakevlm-tinytest \
# bash scripts/train_youtube_tiny.sh

NUM_GPUS=${NUM_GPUS:-1}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

DISTRIBUTED_ARGS="
    --nnodes=1 \
    --nproc_per_node ${NUM_GPUS} \
    --rdzv_backend c10d \
    --rdzv_endpoint localhost:0
"

MODEL_ID=${MODEL_ID:-llava-1.5-7b}
DATA_ROOT=${DATA_ROOT:-playground/data}
TRAIN_DATA=${TRAIN_DATA:-playground/data/train.json}
EVAL_DATA=${EVAL_DATA:-playground/data/test.json}

RUN_ID=${RUN_ID:-${MODEL_ID}-youtubefakevlm-tinytest}
DS_STAGE=${DS_STAGE:-zero2}

# Keep it short and cheap for pipeline sanity check
MAX_STEPS=${MAX_STEPS:-30}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}
GRAD_ACCUM=${GRAD_ACCUM:-2}
LR=${LR:-2e-5}
MODEL_MAX_LEN=${MODEL_MAX_LEN:-1024}

# LoRA defaults follow train_youtube.sh
USE_LORA=${USE_LORA:-True}
LORA_R=${LORA_R:-16}
LORA_ALPHA=${LORA_ALPHA:-16}
Q_LORA=${Q_LORA:-False}

TRAIN_VISION_ENCODER=${TRAIN_VISION_ENCODER:-False}
USE_VISION_LORA=${USE_VISION_LORA:-False}
TRAIN_VISION_PROJECTOR=${TRAIN_VISION_PROJECTOR:-True}

REPORT_TO=${REPORT_TO:-none}

torchrun $DISTRIBUTED_ARGS train.py \
    --model_id "$MODEL_ID" \
    --data_path "$TRAIN_DATA" \
    --eval_data_path "$EVAL_DATA" \
    --image_folder "$DATA_ROOT" \
    --output_dir "./checkpoints/${RUN_ID}" \
    --report_to "$REPORT_TO" \
    --run_name "$RUN_ID" \
    --deepspeed "./ds_configs/${DS_STAGE}.json" \
    --bf16 True \
    --max_steps "$MAX_STEPS" \
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps "$MAX_STEPS" \
    --save_total_limit 1 \
    --learning_rate "$LR" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length "$MODEL_MAX_LEN" \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --train_vision_encoder "$TRAIN_VISION_ENCODER" \
    --use_vision_lora "$USE_VISION_LORA" \
    --train_vision_projector "$TRAIN_VISION_PROJECTOR" \
    --use_lora "$USE_LORA" \
    --q_lora "$Q_LORA" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA"

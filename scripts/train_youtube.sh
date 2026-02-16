#!/usr/bin/env bash
set -euo pipefail

# Example:
# DATA_ROOT=playground/data \
# TRAIN_DATA=playground/data/processed/train_sft.json \
# EVAL_DATA=playground/data/processed/test_sft.json \
# MODEL_ID=llava-1.5-7b \
# RUN_ID=youtubefakevlm-lora \
# bash scripts/train_youtube.sh

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
TRAIN_DATA=${TRAIN_DATA:-playground/data/processed/train_sft.json}
EVAL_DATA=${EVAL_DATA:-playground/data/processed/test_sft.json}

RUN_ID=${RUN_ID:-${MODEL_ID}-youtubefakevlm}
DS_STAGE=${DS_STAGE:-zero2}

NUM_EPOCHS=${NUM_EPOCHS:-3}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-8}
LR=${LR:-2e-5}
MODEL_MAX_LEN=${MODEL_MAX_LEN:-1024}

# Paper-like defaults for domain adaptation with LoRA rank=16
USE_LORA=${USE_LORA:-True}
LORA_R=${LORA_R:-16}
LORA_ALPHA=${LORA_ALPHA:-16}
Q_LORA=${Q_LORA:-False}

TRAIN_VISION_ENCODER=${TRAIN_VISION_ENCODER:-False}
USE_VISION_LORA=${USE_VISION_LORA:-False}
TRAIN_VISION_PROJECTOR=${TRAIN_VISION_PROJECTOR:-True}

torchrun $DISTRIBUTED_ARGS train.py \
    --model_id "$MODEL_ID" \
    --data_path "$TRAIN_DATA" \
    --eval_data_path "$EVAL_DATA" \
    --image_folder "$DATA_ROOT" \
    --output_dir "./checkpoints/${RUN_ID}" \
    --report_to wandb \
    --run_name "$RUN_ID" \
    --deepspeed "./ds_configs/${DS_STAGE}.json" \
    --bf16 True \
    --num_train_epochs "$NUM_EPOCHS" \
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --per_device_eval_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --eval_strategy "no" \
    --save_strategy "epoch" \
    --save_total_limit 1 \
    --learning_rate "$LR" \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length "$MODEL_MAX_LEN" \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --train_vision_encoder "$TRAIN_VISION_ENCODER" \
    --use_vision_lora "$USE_VISION_LORA" \
    --train_vision_projector "$TRAIN_VISION_PROJECTOR" \
    --use_lora "$USE_LORA" \
    --q_lora "$Q_LORA" \
    --lora_r "$LORA_R" \
    --lora_alpha "$LORA_ALPHA"

@echo off
REM RTX 4090 Single GPU Optimized Training Script for FakeVLM (Windows)
REM This script is optimized for 24GB VRAM using QLoRA fine-tuning WITHOUT DeepSpeed

set NUM_GPUS=1
set CUDA_VISIBLE_DEVICES=0

REM Model configuration
set MODEL_ID=llava-1.5-7b

REM Data paths
set TRAIN_DATA_PATH=playground\data\train.json
set EVAL_DATA_PATH=playground\data\test.json
set IMAGE_FOLDER=playground\data
set NUM_FRAMES=8

REM Training strategy - QLoRA for memory efficiency
set TRAIN_VISION_ENCODER=True
set USE_VISION_LORA=True
set TRAIN_VISION_PROJECTOR=True

set USE_LORA=True
set Q_LORA=True
set LORA_R=8
set LORA_ALPHA=16

REM Run configuration
set RUN_ID=%MODEL_ID%-fakevlm-rtx4090

REM Training hyperparameters - optimized for 24GB VRAM
set PER_DEVICE_BATCH_SIZE=1
set GRAD_ACCUM=32
set NUM_EPOCHS=2

set LR=2e-5
set MODEL_MAX_LEN=1024

REM Activate conda environment
call conda activate fakevlm

REM Launch training (WITHOUT DeepSpeed)
python train.py ^
    --model_id %MODEL_ID% ^
    --data_path %TRAIN_DATA_PATH% ^
    --eval_data_path %EVAL_DATA_PATH% ^
    --image_folder %IMAGE_FOLDER% ^
    --num_frames %NUM_FRAMES% ^
    --output_dir ./checkpoints/%RUN_ID% ^
    --report_to wandb ^
    --run_name %RUN_ID% ^
    --bf16 True ^
    --num_train_epochs %NUM_EPOCHS% ^
    --per_device_train_batch_size %PER_DEVICE_BATCH_SIZE% ^
    --per_device_eval_batch_size %PER_DEVICE_BATCH_SIZE% ^
    --gradient_accumulation_steps %GRAD_ACCUM% ^
    --eval_strategy "no" ^
    --save_strategy "epoch" ^
    --save_total_limit 1 ^
    --learning_rate %LR% ^
    --weight_decay 0. ^
    --warmup_ratio 0.03 ^
    --lr_scheduler_type "cosine" ^
    --logging_steps 1 ^
    --tf32 True ^
    --model_max_length %MODEL_MAX_LEN% ^
    --gradient_checkpointing True ^
    --dataloader_num_workers 4 ^
    --train_vision_encoder %TRAIN_VISION_ENCODER% ^
    --use_vision_lora %USE_VISION_LORA% ^
    --train_vision_projector %TRAIN_VISION_PROJECTOR% ^
    --use_lora %USE_LORA% ^
    --q_lora %Q_LORA% ^
    --lora_r %LORA_R% ^
    --lora_alpha %LORA_ALPHA%

echo.
echo ========================================
echo Training completed!
echo Checkpoints saved to ./checkpoints/%RUN_ID%
echo ========================================
pause

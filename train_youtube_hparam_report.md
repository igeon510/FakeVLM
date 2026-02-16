# Youtube Final Training Configuration Report

## Goal

Fine-tune Youtube domain data (~6,000 images) starting from:

- Base model: `llava-hf/llava-1.5-7b-hf`
- Initial adapter: `igeon510/llava-1.5-7b-qlora`

## Code Changes

### 1) Adapter continuation support in training code

- File: `train.py`
- Change:
  - Added support for `--lora_weight_path` to actually load existing LoRA adapters.
  - When `use_lora=True` and `lora_weight_path` is set, training now uses:
    - `PeftModel.from_pretrained(..., is_trainable=True)`
  - This enables continuing training from an existing QLoRA adapter, instead of always creating a fresh LoRA adapter.

### 2) Finalized Youtube training script

- File: `scripts/train_youtube.sh`
- Change:
  - Set defaults for final domain adaptation training.
  - Added adapter defaults and exposed overridable runtime env vars.

## Final Default Hyperparameters

- `MODEL_ID=llava-1.5-7b`
- `LORA_WEIGHT_PATH=igeon510/llava-1.5-7b-qlora`
- `Q_LORA=True`
- `USE_LORA=True`
- `NUM_EPOCHS=4`
- `PER_DEVICE_BATCH_SIZE=2`
- `GRAD_ACCUM=8`
- `LR=5e-5`
- `WARMUP_RATIO=0.05`
- `WEIGHT_DECAY=0.0`
- `MODEL_MAX_LEN=1024`
- `DS_STAGE=zero2`
- `TRAIN_VISION_ENCODER=False`
- `USE_VISION_LORA=False`
- `TRAIN_VISION_PROJECTOR=False`
- `EVAL_STRATEGY=epoch`
- `SAVE_STRATEGY=epoch`
- `SAVE_TOTAL_LIMIT=2`

## Why These Settings

- `Q_LORA=True`: lower memory usage for 7B model fine-tuning.
- `LORA_WEIGHT_PATH` set: starts from existing fake-image detection adapter knowledge.
- `LR=5e-5`: moderate LR for adapter continuation (faster than very low LR, less unstable than aggressive LR).
- `NUM_EPOCHS=4`: enough exposure for ~6k samples without extreme overfitting by default.
- `batch=2, accum=8`: stable effective batch with manageable GPU memory.
- freeze vision modules: improves stability and reduces compute/memory; focuses adaptation on language-side decision/explanation behavior.
- `eval/save by epoch`: clean checkpointing and easier comparison per epoch.

## Effective Batch Size

Effective batch size is:
`PER_DEVICE_BATCH_SIZE x GRAD_ACCUM x NUM_GPUS`

Examples:

- 1 GPU: `2 x 8 x 1 = 16`
- 2 GPU: `2 x 8 x 2 = 32`
- 4 GPU: `2 x 8 x 4 = 64`

## Recommended Run Command

```bash
CUDA_VISIBLE_DEVICES=0 \
NUM_GPUS=1 \
DATA_ROOT=playground/data \
TRAIN_DATA=playground/data/processed/train.json \
EVAL_DATA=playground/data/processed/test.json \
RUN_ID=youtube-qlora-final \
bash scripts/train_youtube.sh
```

## Notes

- If your base LLaVA checkpoint is local, set `MODEL_LOCAL_PATH=/path/to/llava-1.5-7b-hf`.
- If no evaluation split is available, set `EVAL_STRATEGY=no` and leave `EVAL_DATA` unset.
- If OOM happens, reduce `PER_DEVICE_BATCH_SIZE` to 1, then increase `GRAD_ACCUM`.

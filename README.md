## environment

## YoutubeFakeClue Workflow (Custom Dataset)

If your custom dataset looks like:

```text
playground
└── data
    ├── fake
    ├── real
    ├── train.json
    └── test.json
```

you can prepare/train/evaluate with the scripts below.

### 1. Fine-tune for Youtube domain

```bash
DATA_ROOT=playground/data \
TRAIN_DATA=playground/data/processed/train.json \
EVAL_DATA=playground/data/processed/test.json \
RUN_ID=youtubefakevlm-lora \
bash scripts/train_youtube.sh
```

### 2. Compare Our Model vs FakeVLM vs OpenAI API

```bash
OPENAI_API_KEY=YOUR_KEY \
python scripts/eval_multimodel.py \
  --test-json playground/data/processed/test_eval.json \
  --image-root playground/data \
  --our-model-path checkpoints/youtubefakevlm-lora \
  --fakevlm-model-path lingcco/fakeVLM \
  --openai-model chatgpt-5.2 \
  --output-dir results/youtube_benchmark
```

#!/usr/bin/env bash
set -euo pipefail
INPUT_FILE="../predictions_Lang2Act_top3.jsonl"
MODEL="Qwen/Qwen2.5-72B-Instruct"
BATCH=5
WORKERS=2
API_BASE="https://api.siliconflow.cn/v1/"

ROOT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

# 启动 Python 评估程序
python "$ROOT_DIR/src/eval.py" \
    --input_file "$INPUT_FILE" \
    --llm_judge_model "$MODEL" \
    --batch_size "$BATCH" \
    --num_workers "$WORKERS" \
    --api_base_url "$API_BASE" \
    --resume
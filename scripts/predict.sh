#!/bin/bash
set -e
export CUDA_VISIBLE_DEVICES=0

python src/predict.py \
  --dataset mmlb \
  --model xiongyq/Lang2Act-7B \
  --prompt_mode Lang2Act 
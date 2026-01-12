#!/bin/bash

set -x
export PYTHONUNBUFFERED=1
export MKL_SERVICE_FORCE_INTEL=1

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct 

TRAIN_DATA_PATH=actionrl_train.jsonl
VALIDATION_DATA_PATH=actionrl_val.jsonl

export DEBUG_REWARD=0

python3 -m verl.trainer.main \
    config=examples/actionrl.yaml \
    data.train_files=${TRAIN_DATA_PATH} \
    data.val_files=${VALIDATION_DATA_PATH} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.clip_ratio_low=0.2 \
    worker.actor.clip_ratio_high=0.28 \
    algorithm.disable_kl=True \
    algorithm.online_filtering=True \



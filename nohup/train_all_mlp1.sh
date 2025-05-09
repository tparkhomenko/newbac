#!/bin/bash

mkdir -p nohup

MODEL_CONFIGS=("256_512_256_DO03" "128_64_16_DO01" "64_16_DO01")

echo "Starting MLP1 (Skin/Not Skin) training with all configurations"
for model_config in "${MODEL_CONFIGS[@]}"; do
    log_file="nohup/log_mlp1_${model_config}.out"
    echo "Starting MLP1 with model=${model_config}"
    nohup python training/train_mlp.py --dataset_config mlp1_balanced --model_config ${model_config} > ${log_file} 2>&1 &
    echo "Job started, log file: ${log_file}"
done

echo "All MLP1 training jobs have been started. Check nohup logs for progress." 
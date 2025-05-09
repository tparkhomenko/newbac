#!/bin/bash

mkdir -p nohup

MODEL_CONFIGS=("256_512_256_DO03" "128_64_16_DO01" "64_16_DO01")
DATASET_CONFIGS=("mlp2_augmented" "mlp2_original")

echo "Starting MLP2 (Lesion Type) training with all configurations"
for dataset_config in "${DATASET_CONFIGS[@]}"; do
    for model_config in "${MODEL_CONFIGS[@]}"; do
        log_file="nohup/log_mlp2_${dataset_config}_${model_config}.out"
        echo "Starting MLP2 with dataset=${dataset_config}, model=${model_config}"
        nohup python training/train_mlp.py --dataset_config ${dataset_config} --model_config ${model_config} > ${log_file} 2>&1 &
        echo "Job started, log file: ${log_file}"
    done
done

echo "All MLP2 training jobs have been started. Check nohup logs for progress." 
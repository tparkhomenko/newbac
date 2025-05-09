#!/bin/bash

mkdir -p nohup

MODEL_CONFIGS=("256_512_256_DO03" "128_64_16_DO01" "64_16_DO01")

WANDB_PROJECT="skin-lesion-mlp-experiments-18-models"

echo "Starting sequential MLP1 (Skin/Not Skin) training for all configurations"
for model_config in "${MODEL_CONFIGS[@]}"; do
    log_file="nohup/log_mlp1_${model_config}.out"
    echo "Running MLP1 with model=${model_config} (logging to $log_file)"
    python training/train_mlp.py --dataset_config mlp1_balanced --model_config ${model_config} --wandb_project ${WANDB_PROJECT} 2>&1 | tee ${log_file}
    echo "Finished MLP1 with model=${model_config}"
done

echo "All sequential MLP1 training jobs have completed. Check nohup logs for details." 
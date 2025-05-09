#!/bin/bash

mkdir -p nohup

WANDB_PROJECT="skin-lesion-mlp-experiments-18-models"

# Run the previous best architecture first
log_file="nohup/log_mlp2_augmented_512_256_DO03.out"
echo "Running MLP2 with dataset=mlp2_augmented, model=512_256_DO03 (logging to $log_file)"
python training/train_mlp.py --dataset_config mlp2_augmented --model_config 512_256_DO03 --wandb_project ${WANDB_PROJECT} 2>&1 | tee ${log_file}
echo "Finished MLP2 with dataset=mlp2_augmented, model=512_256_DO03"

MODEL_CONFIGS=("256_512_256_DO03" "128_64_16_DO01" "64_16_DO01")
DATASET_CONFIGS=("mlp2_augmented" "mlp2_original")

for dataset_config in "${DATASET_CONFIGS[@]}"; do
    for model_config in "${MODEL_CONFIGS[@]}"; do
        log_file="nohup/log_mlp2_${dataset_config}_${model_config}.out"
        echo "Running MLP2 with dataset=${dataset_config}, model=${model_config} (logging to $log_file)"
        python training/train_mlp.py --dataset_config ${dataset_config} --model_config ${model_config} --wandb_project ${WANDB_PROJECT} 2>&1 | tee ${log_file}
        echo "Finished MLP2 with dataset=${dataset_config}, model=${model_config}"
    done
done

echo "All sequential MLP2 training jobs have completed. Check nohup logs for details."
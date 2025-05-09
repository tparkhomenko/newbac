#!/bin/bash

# Create log directory if it doesn't exist
mkdir -p nohup

echo "Starting training for all model and dataset configurations..."

# Array of model architectures
MODEL_CONFIGS=("256_512_256_DO03" "128_64_16_DO01" "64_16_DO01")

# MLP1: Skin/Not Skin Classification
echo "Starting MLP1 (Skin/Not Skin) training with all configurations"
for model_config in "${MODEL_CONFIGS[@]}"; do
    log_file="nohup/log_mlp1_${model_config}.out"
    echo "Starting MLP1 with dataset=mlp1_balanced, model=${model_config}"
    nohup python training/train_mlp.py --dataset_config mlp1_balanced --model_config ${model_config} > ${log_file} 2>&1 &
    echo "Job started, log file: ${log_file}"
done

# Wait 1 minute before starting MLP2 jobs
echo "Waiting 1 minute before starting MLP2 jobs..."
sleep 60

# MLP2: Lesion Type Classification
echo "Starting MLP2 (Lesion Type) training with all configurations"
# Augmented dataset version
for model_config in "${MODEL_CONFIGS[@]}"; do
    log_file="nohup/log_mlp2_augmented_${model_config}.out"
    echo "Starting MLP2 with dataset=mlp2_augmented, model=${model_config}"
    nohup python training/train_mlp.py --dataset_config mlp2_augmented --model_config ${model_config} > ${log_file} 2>&1 &
    echo "Job started, log file: ${log_file}"
done
# Original dataset version
for model_config in "${MODEL_CONFIGS[@]}"; do
    log_file="nohup/log_mlp2_original_${model_config}.out"
    echo "Starting MLP2 with dataset=mlp2_original, model=${model_config}"
    nohup python training/train_mlp.py --dataset_config mlp2_original --model_config ${model_config} > ${log_file} 2>&1 &
    echo "Job started, log file: ${log_file}"
done

# Wait 1 minute before starting MLP3 jobs
echo "Waiting 1 minute before starting MLP3 jobs..."
sleep 60

# MLP3: Benign/Malignant Classification
echo "Starting MLP3 (Benign/Malignant) training with all configurations"
# Augmented dataset version with 2000 samples per class
for model_config in "${MODEL_CONFIGS[@]}"; do
    log_file="nohup/log_mlp3_augmented_${model_config}.out"
    echo "Starting MLP3 with dataset=mlp3_augmented, model=${model_config}"
    nohup python training/train_mlp.py --dataset_config mlp3_augmented --model_config ${model_config} > ${log_file} 2>&1 &
    echo "Job started, log file: ${log_file}"
done
# Original dataset version
for model_config in "${MODEL_CONFIGS[@]}"; do
    log_file="nohup/log_mlp3_original_${model_config}.out"
    echo "Starting MLP3 with dataset=mlp3_original, model=${model_config}"
    nohup python training/train_mlp.py --dataset_config mlp3_original --model_config ${model_config} > ${log_file} 2>&1 &
    echo "Job started, log file: ${log_file}"
done
# Full augmented dataset version
for model_config in "${MODEL_CONFIGS[@]}"; do
    log_file="nohup/log_mlp3_augmented_full_${model_config}.out"
    echo "Starting MLP3 with dataset=mlp3_augmented_full, model=${model_config}"
    nohup python training/train_mlp.py --dataset_config mlp3_augmented_full --model_config ${model_config} > ${log_file} 2>&1 &
    echo "Job started, log file: ${log_file}"
done

echo "All training jobs have been started. Check nohup logs for progress."
echo "Use 'ps aux | grep python' to see running processes."
echo "Use 'watch nvidia-smi' to monitor GPU usage." 
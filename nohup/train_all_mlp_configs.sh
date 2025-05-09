#!/bin/bash

# Create log directory if it doesn't exist
mkdir -p nohup

echo "Starting training for all MLP configurations..."

# MLP1: Skin/Not Skin Classifier
echo "Starting MLP1 (Skin/Not Skin) training..."
# nohup python training/train_skin_not_skin.py --model_config 256_512_256_DO03 > nohup/log_mlp1_DO03.out 2>&1 &
# echo "Started MLP1 with 256_512_256_DO03 configuration"

nohup python training/train_skin_not_skin.py --model_config 128_64_16_DO01 > nohup/log_mlp1_DO01.out 2>&1 &
echo "Started MLP1 with 128_64_16_DO01 configuration"

nohup python training/train_skin_not_skin.py --model_config 64_16_DO01 > nohup/log_mlp1_DO16.out 2>&1 &
echo "Started MLP1 with 64_16_DO01 configuration"

# Wait a bit to stagger GPU usage
echo "Waiting 60 seconds before starting MLP2 training..."
sleep 60

# MLP2: Lesion Type Classifier
echo "Starting MLP2 (Lesion Type) training..."
# nohup python training/train_lesion_type.py --model_config 256_512_256_DO03 > nohup/log_mlp2_DO03.out 2>&1 &
# echo "Started MLP2 with 256_512_256_DO03 configuration"

nohup python training/train_lesion_type.py --model_config 128_64_16_DO01 > nohup/log_mlp2_DO01.out 2>&1 &
echo "Started MLP2 with 128_64_16_DO01 configuration"

nohup python training/train_lesion_type.py --model_config 64_16_DO01 > nohup/log_mlp2_DO16.out 2>&1 &
echo "Started MLP2 with 64_16_DO01 configuration"

# Wait a bit to stagger GPU usage
echo "Waiting 60 seconds before starting MLP3 training..."
sleep 60

# MLP3: Benign/Malignant Classifier
echo "Starting MLP3 (Benign/Malignant) training..."
# nohup python training/train_benign_malignant.py --model_config 256_512_256_DO03 --config balanced > nohup/log_mlp3_DO03.out 2>&1 &
# echo "Started MLP3 with 256_512_256_DO03 configuration"

nohup python training/train_benign_malignant.py --model_config 128_64_16_DO01 --config balanced > nohup/log_mlp3_DO01.out 2>&1 &
echo "Started MLP3 with 128_64_16_DO01 configuration"

nohup python training/train_benign_malignant.py --model_config 64_16_DO01 --config balanced > nohup/log_mlp3_DO16.out 2>&1 &
echo "Started MLP3 with 64_16_DO01 configuration"

echo "All training jobs have been started. Check nohup logs for progress."
echo "Use 'ps aux | grep python' to see running processes."
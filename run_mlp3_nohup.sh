#!/bin/bash
# Run MLP3 training with nohup to keep it running after the terminal is closed

# Ensure we're in the project directory
cd "$(dirname "$0")"

# Start training with nohup to make it keep running even if terminal closes
# Direct stdout and stderr to log files
nohup ./run_all_mlp3.sh > mlp3_stdout.log 2> mlp3_stderr.log &

# Get the process ID
PROCESS_ID=$!

echo "MLP3 training started with process ID: $PROCESS_ID"
echo "You can safely close this terminal."
echo ""
echo "To check progress later:"
echo "  - View logs: tail -f mlp3_training_sequence.log"
echo "  - Check if process is running: ps -p $PROCESS_ID"
echo "  - Kill the process if needed: kill -9 $PROCESS_ID"
echo ""
echo "Models will be saved to:"
echo " - saved_models/benign_malignant/benign_malignant_balanced_best.pth"
echo " - saved_models/benign_malignant/benign_malignant_original_best.pth"
echo " - saved_models/benign_malignant/benign_malignant_augmented_best.pth" 
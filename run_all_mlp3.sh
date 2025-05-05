#!/bin/bash
# Script to run all three MLP3 configurations sequentially with error handling
# and memory cleanup between runs.

LOG_FILE="mlp3_training_sequence.log"

# Function to log messages with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to clean up GPU memory between runs
cleanup() {
    log "Cleaning up GPU memory..."
    # Kill any potentially hanging Python processes from previous run
    for pid in $(ps -ef | grep python | grep train_benign_malignant | awk '{print $2}'); do
        log "Killing process $pid"
        kill -9 $pid 2>/dev/null
    done
    
    # Wait a moment
    sleep 10
    
    # Clear CUDA cache
    log "Waiting for GPU memory to be released..."
    sleep 30
    log "Cleanup complete."
}

# Function to run a single configuration
run_config() {
    CONFIG_NAME=$1
    log "=========================================================="
    log "Starting training for configuration: $CONFIG_NAME"
    log "=========================================================="
    
    python -m training.train_benign_malignant --config "$CONFIG_NAME"
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        log "Configuration $CONFIG_NAME completed successfully."
    else
        log "Configuration $CONFIG_NAME failed with exit code $EXIT_CODE."
    fi
    
    # Clean up regardless of success/failure
    cleanup
    
    return $EXIT_CODE
}

# Main execution
log "Starting sequential MLP3 training for all configurations"
log "Logging to $LOG_FILE"

# Initial cleanup to ensure clean start
cleanup

# Run each configuration in sequence
CONFIGS=("balanced" "original" "augmented")
COMPLETED=0
FAILED=0

for CONFIG in "${CONFIGS[@]}"; do
    run_config "$CONFIG"
    if [ $? -eq 0 ]; then
        COMPLETED=$((COMPLETED+1))
    else
        FAILED=$((FAILED+1))
    fi
    
    # Add some extra cleanup time between runs
    log "Waiting 60 seconds before starting next configuration..."
    sleep 60
done

# Print summary
log "=========================================================="
log "Training completed."
log "Configurations completed successfully: $COMPLETED"
log "Configurations failed: $FAILED"
log "=========================================================="

log "All training configurations have been processed."
log "Check wandb for detailed metrics and the following folders for models:"
log " - balanced: saved_models/benign_malignant/benign_malignant_balanced_best.pth"
log " - original: saved_models/benign_malignant/benign_malignant_original_best.pth"
log " - augmented: saved_models/benign_malignant/benign_malignant_augmented_best.pth" 
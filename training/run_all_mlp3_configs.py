#!/usr/bin/env python3
"""
Script to run all MLP3 (benign/malignant classifier) configurations sequentially.
"""

import os
import sys
import argparse
import subprocess
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mlp3_experiments.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_experiment(config_name):
    """Run a single experiment with the specified configuration."""
    start_time = time.time()
    logger.info(f"Starting experiment with config: {config_name}")
    
    try:
        # Run the training script with the specified config
        cmd = f"python -m training.train_benign_malignant --config {config_name}"
        logger.info(f"Running command: {cmd}")
        
        process = subprocess.Popen(
            cmd, 
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True
        )
        
        # Stream the output
        for line in process.stdout:
            print(line.strip())
        
        process.wait()
        
        if process.returncode == 0:
            logger.info(f"Experiment {config_name} completed successfully")
        else:
            logger.error(f"Experiment {config_name} failed with return code {process.returncode}")
    
    except Exception as e:
        logger.error(f"Error running experiment {config_name}: {str(e)}")
    
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    logger.info(f"Experiment {config_name} took {int(hours)}h {int(minutes)}m {int(seconds)}s")

def run_all_experiments():
    """Run all three MLP3 experiments sequentially."""
    configs = ["original", "augmented", "balanced"]
    
    total_start_time = time.time()
    logger.info(f"Starting all MLP3 experiments at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Will run the following configurations: {', '.join(configs)}")
    
    for config in configs:
        run_experiment(config)
        logger.info(f"Completed experiment: {config}")
        logger.info("=" * 80)
    
    total_elapsed_time = time.time() - total_start_time
    hours, remainder = divmod(total_elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info("=" * 80)
    logger.info(f"All experiments completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    logger.info("Summary of experiments:")
    for config in configs:
        logger.info(f"- {config}: See saved_models/benign_malignant/benign_malignant_{config}_best.pth")
    
    logger.info("Check wandb for detailed logs and metrics")

def run_single_experiment(config_name):
    """Run a single specified experiment."""
    if config_name not in ["original", "augmented", "balanced"]:
        logger.error(f"Invalid configuration: {config_name}")
        logger.info("Valid configurations are: original, augmented, balanced")
        return
    
    logger.info(f"Running single experiment: {config_name}")
    run_experiment(config_name)
    logger.info(f"Experiment {config_name} completed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MLP3 experiments')
    parser.add_argument('--config', type=str, default=None,
                        choices=['original', 'augmented', 'balanced', 'all'],
                        help='Configuration to run (or "all" for all configs)')
    args = parser.parse_args()
    
    if args.config == 'all' or args.config is None:
        run_all_experiments()
    else:
        run_single_experiment(args.config) 
#!/bin/bash

# Define the project directory
PROJECT_DIR="/home/iwe67/PycharmProjects/DrugFlow"

# Navigate to the directory
cd "$PROJECT_DIR" || exit

# Define the log file name with a timestamp
LOG_FILE="train_$(date +%Y%m%d_%H%M%S).log"

echo "Starting training. Output being logged to $LOG_FILE"

# Execute the command
# We use 'exec' to replace the shell process with the python process
env PYTHONNOUSERSITE=1 \
    PYTHONPATH=. \
    WANDB_MODE=offline \
    WANDB_DIR="$PROJECT_DIR/runs/wandb" \
    /home/iwe67/miniforge3/envs/drugflow/bin/python src/train.py \
    --config configs/training/energy_force_qm9_curated.yml \
    > "$LOG_FILE" 2>&1
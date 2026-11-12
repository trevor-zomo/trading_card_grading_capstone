#!/bin/bash
# Grid search over the training settings (needs GNU parallel).
#   usage: master_train_gridsearch.sh [base_path] [number of runs to train at the same time]
set -euo pipefail

# Path to the data
base_path="${1:-/home/data/data_for_model_3}"
jobs="${2:-2}"

# Drop rates
droprates=(0.3 0.5 0.7)

# Starting learning rates for the fine-tuning stage
lr_finetunes=(0.00001 0.00003 0.0001)

# Spread of the training targets around the true grade (0 = plain one-hot targets)
label_sigmas=(0 1 1.5)

# How many layers, counted from the last of the ResNet50, are fine-tuned (175 is all of them)
unfreeze_after_list=(40 80 175)

# Export variables to be used by parallel
export base_path

# Define the function that will be run in parallel
run_experiment() {
  # Run the training with the current combination of parameters
  card-grading-train \
    --base_path "$base_path" \
    --droprate "$1" \
    --lr_finetune "$2" \
    --label_sigma "$3" \
    --unfreeze_after "$4"
}

# Export the function so it can be used by parallel
export -f run_experiment

# Use GNU parallel to run the experiments in parallel
parallel -j "$jobs" run_experiment ::: "${droprates[@]}" ::: "${lr_finetunes[@]}" ::: "${label_sigmas[@]}" ::: "${unfreeze_after_list[@]}"

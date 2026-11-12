#!/bin/bash
# testing out a bunch of different dropout rates
set -euo pipefail

base_path="${1:-/home/data/data_for_model_3}"

for droprate in 0 0.012 0.025 0.05 0.1 0.2; do
  card-grading-train --n_epochs 50 --base_path "$base_path" --droprate "$droprate"
done

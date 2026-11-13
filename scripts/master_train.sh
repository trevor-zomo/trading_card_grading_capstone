#!/bin/bash
# testing out a bunch of different dropout rates
set -euo pipefail

base_path="${1:-/home/data/data_for_model_3}"

for droprate in 0.2 0.35 0.5 0.65; do
  card-grading-train --base_path "$base_path" --droprate "$droprate"
done

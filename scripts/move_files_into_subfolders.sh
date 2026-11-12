#!/bin/bash
# Move the cropped images in a folder into psa1 ... psa10 subfolders, by the grade in their file name.
set -euo pipefail
shopt -s nullglob

directory="${1:?usage: move_files_into_subfolders.sh <directory>}"

mkdir -p "$directory"/psa{1..10}

for file in "$directory"/card*_PSA*_cropped.jpg; do
    # half grades (e.g. _PSA8_5_) go into the folder of the whole grade
    grade=$(basename "$file" | sed -n 's/.*_PSA\([0-9][0-9]*\).*/\1/p')
    if [[ -z "$grade" || ! -d "$directory/psa${grade}" ]]; then
        echo "Skipping file without a PSA grade from 1 to 10: $file" >&2
        continue
    fi
    mv "$file" "$directory/psa${grade}/"
done

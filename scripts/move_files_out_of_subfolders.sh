#!/bin/bash
# Move the images out of the psa1 ... psa10 subfolders of the current directory, into the current directory.
set -euo pipefail
shopt -s nullglob

for folder in psa{1..10}; do
    if [[ -d "$folder" ]]; then
        echo "Moving files from $folder to parent directory"
        for file in "$folder"/*; do
            mv "$file" .
        done
    fi
done

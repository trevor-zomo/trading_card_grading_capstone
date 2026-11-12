#!/bin/bash
# Scrape all ten PSA grades in parallel, resuming each grade at the given card number.
# Stop the scrapers with ctrl-c.
set -euo pipefail

files_dir="${1:-$HOME/Desktop/collection_files}"

# card number to resume from, for PSA grades 1 to 10
start_cards=(3104 5352 7992 16288 18053 17917 40680 47142 94426 101241)

trap 'kill 0' INT TERM

for grade in {1..10}; do
  card-grading-scrape --grade "$grade" --card_num "${start_cards[grade-1]}" \
    --files_dir "$files_dir" --photo_dir "$files_dir/images_psa$grade" &
done
wait

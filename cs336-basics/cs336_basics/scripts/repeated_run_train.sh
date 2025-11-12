#!/bin/bash
set -euo pipefail

if [ "$#" -lt 1 ]; then
	echo "Usage: $0 BASE_DIR [NUM_RUNS]"
	exit 1
fi

BASE_DIR="$1"
NUM_RUNS="${2:-100}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_default_owt_on_h100.sh"

if [ ! -f "$TRAIN_SCRIPT" ]; then
	echo "Train script not found at $TRAIN_SCRIPT" >&2
	exit 1
fi

if [ ! -x "$TRAIN_SCRIPT" ]; then
	# Try to make it executable; ignore failure (script may be sourced by an interpreter)
	chmod +x "$TRAIN_SCRIPT" || true
fi

for i in $(seq 1 "$NUM_RUNS"); do
	echo "=== Run $i/$NUM_RUNS ==="
	# Use an if-check so a single run failing does not stop the entire loop
	if ! "$TRAIN_SCRIPT" "$BASE_DIR"; then
		echo "Run $i failed; continuing to next run" >&2
	fi
done

echo "All $NUM_RUNS runs complete."


#!/usr/bin/env bash
set -u

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ENDPOINT="${ENDPOINT:-http://localhost:5001/predict}"

images=(
  "$SCRIPT_DIR/screen_1.jpg"
  "$SCRIPT_DIR/screen_2.jpg"
  "$SCRIPT_DIR/real_1.jpeg"
  "$SCRIPT_DIR/real_2.jpeg"
)

echo "Running tests against: $ENDPOINT"

for img in "${images[@]}"; do
  if [ -f "$img" ]; then
    echo "Posting $img"
    curl -s -S -X POST -F "file=@$img" "$ENDPOINT" || true
    echo ""
  else
    echo "Skipping missing file: $img"
  fi
done


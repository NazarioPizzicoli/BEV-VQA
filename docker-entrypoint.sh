#!/usr/bin/env bash
set -e

# Set default directory paths if not already provided
export BEV_VQA_BEV_DIR="${BEV_VQA_BEV_DIR:-/data/dataset/bev_features}"
export BEV_VQA_DATASETS_DIR="${BEV_VQA_DATASETS_DIR:-/data/vqa_datasets/unified}"

# Display aesthetic & informative ASCII banner (unless QUIET=1 is set)
if [ -z "$QUIET" ] && [ -f "/app/scripts/banner.py" ]; then
    python /app/scripts/banner.py 2>/dev/null || true
fi

# If first argument starts with a dash (-), forward as arguments to interactive inference
if [ $# -gt 0 ] && [ "${1:0:1}" = "-" ]; then
    set -- python scripts/inference_interactive.py "$@"
fi

exec "$@"

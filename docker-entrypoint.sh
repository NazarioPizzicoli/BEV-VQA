#!/usr/bin/env bash
set -e

# Set default directory paths if not already provided
export BEV_VQA_BEV_DIR="${BEV_VQA_BEV_DIR:-/data/dataset/bev_features}"
export BEV_VQA_DATASETS_DIR="${BEV_VQA_DATASETS_DIR:-/data/vqa_datasets/unified}"

# Display aesthetic & informative ASCII banner (unless QUIET=1 is set)
if [ -z "$QUIET" ] && [ -f "/app/scripts/banner.py" ]; then
    python /app/scripts/banner.py 2>/dev/null || true
fi

exec "$@"

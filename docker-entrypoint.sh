#!/usr/bin/env bash
set -e

echo "=================================================="
echo "          BEV-VLM Docker Environment              "
echo "=================================================="

# Check Python environment & PyTorch CUDA status
python -c "
import torch
print(f'PyTorch {torch.__version__} | CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'Active GPU: {torch.cuda.get_device_name(0)} (Capability {torch.cuda.get_device_capability(0)})')
" 2>/dev/null || true

# Set default directory paths if not already provided
export BEV_VQA_BEV_DIR="${BEV_VQA_BEV_DIR:-/data/bev_features}"
export BEV_VQA_DATASETS_DIR="${BEV_VQA_DATASETS_DIR:-/data/vqa_datasets}"

echo "Working directory: $(pwd)"
echo "BEV Features Dir:  $BEV_VQA_BEV_DIR"
echo "Datasets Dir:      $BEV_VQA_DATASETS_DIR"
echo "=================================================="

exec "$@"

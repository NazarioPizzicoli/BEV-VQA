# BEV-VQA

Modular **BEV → VLM → VQA** pipeline for autonomous driving scene understanding.

## Architecture

BeLLA-style approach: pre-extracted BEV features from GaussianCar → Deeper Conv projector → Qwen2.5-3B-Instruct + LoRA → generative VQA.

## Project Structure

```
BEV-VQA/
├── configs/                    # Hydra configuration
│   ├── config.yaml             # Main config (model, training, eval)
│   └── paths/veh.yaml          # Data paths
├── src/bev_vqa/                # Core library
│   ├── data/                   # Data loading & validation
│   │   ├── dataset.py          # BEVPretrainDataset, BEVQADataset
│   │   ├── collate.py          # Collate functions (train/eval)
│   │   └── sanity.py           # Block 0: Data sanity checks
│   ├── probes/                 # Diagnostic probes
│   │   └── feature_probe.py    # Block 1: CNN probe on BEV features
│   ├── models/                 # Model architectures
│   │   ├── projector.py        # DeeperConv / Q-Former projectors
│   │   └── vlm.py              # BEVVLM: full projector + LLM + LoRA
│   └── training/               # Training & evaluation
│       ├── trainer.py          # Training loop, eval functions
│       └── metrics.py          # Accuracy, NLG metrics, shuffle test
├── scripts/                    # Standalone runnable scripts
│   ├── run_block0_sanity.py    # Data validation (5 min)
│   ├── run_block1_probe.py     # Feature probe (30 min)
│   ├── run_block2_encoder.py   # Encoder probe [TODO]
│   ├── run_block3_textonly.py  # LLM baseline [TODO]
│   └── run_block4_vlm.py      # Full VLM training [TODO]
├── tests/                      # Unit tests
├── outputs/                    # Training outputs & reports
└── pyproject.toml              # Dependencies & build config
```

## Pipeline Blocks

Each block is independently testable. Run them in order:

| Block | Script | What it tests | Success Criterion |
|-------|--------|--------------|-------------------|
| **0** | `run_block0_sanity.py` | Data integrity | All checks PASS |
| **1** | `run_block1_probe.py` | BEV features contain info | Probe accuracy > text-only + 5pts |
| **2** | `run_block2_encoder.py` | Projector produces useful tokens | Linear probe accuracy > random |
| **3** | `run_block3_textonly.py` | LLM text-only baseline | ~55% accuracy (reference) |
| **4** | `run_block4_vlm.py` | Full BEV+LLM system | Accuracy > text-only, shuffle gap > 0 |

## Setup

```bash
cd BEV-VQA
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```

## Datasets

Uses 3 NuScenes-based unified datasets:
- **OmniDrive Descriptions** (28k) → Stage 1 pretrain
- **NuScenes-QA** (234k) → Stage 2 fine-tuning + accuracy eval
- **DriveLM** (341k) → Stage 2 fine-tuning + generative eval (BLEU/METEOR/ROUGE/CIDEr)

## Key Design Decisions

1. **LLM**: Qwen2.5-3B-Instruct (BeLLA used LLaMA-3B → 59.6% accuracy)
2. **Projector**: BeLLA-style Deeper Conv with 32 output tokens (NOT 196 like previous attempts)
3. **No F.normalize/scale**: Previous project's 0.45 scale was wrong — projector learns to match LLM embedding space through training
4. **2-stage training**: Stage 1 (alignment, projector only) → Stage 2 (VQA, projector + LoRA)

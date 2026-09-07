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

| Block | Script | What it tests | Success Criterion | Result | Status |
|-------|--------|--------------|-------------------|--------|--------|
| **0** | `run_block0_sanity.py` | Data integrity & BEV tensors | All checks PASS | 27,905 train / 5,984 val verified | ✅ PASSED |
| **1** | `run_block1_probe.py` | BEV features contain spatial info | Probe > text baseline (+5%) | **57.0%** (+7.0% over text 50%) | ✅ PASSED |
| **2** | `run_block2_projector.py` | Projector produces useful tokens | DeeperConv vs Q-Former | DeeperConv selected, pos_embed added | ✅ PASSED |
| **3** | `run_block3_llm_baseline.py` | LLM text-only baseline | Zero-shot reference | **19.6%** balanced zero-shot | ✅ PASSED |
| **4** | `run_stage1_pretrain.py` | BEV-Language Alignment (16k scenes) | Val Loss decrease, coherent text | Val Loss: **2.0854** ($\Delta = -0.66$) | ✅ PASSED |
| **5** | `run_stage2_vqa_finetune.py` | End-to-End VQA Finetuning (LoRA+Proj) | Accuracy > 19.6% baseline | **62.0%** peak / **54.7%** final (+35.1%) | ✅ PASSED |

## Final Benchmark Results (Block 5 vs Block 3 Baseline)

Evaluated on NuScenes-QA validation across all 5 balanced reasoning categories:

| Category | Text-Only Baseline (Block 3) | BEV-VLM Stage 2 (Block 5) | Visual & Adaptation Gain ($\Delta$) |
|:---|:---:|:---:|:---:|
| **Overall Accuracy** | **19.6%** | **54.7%** (Peak: **62.0%**) | **+35.1%** |
| `count` | 0.0% | **18.3%** | +18.3% |
| `status` | 6.0% | **56.7%** | +50.7% |
| `object` | 12.0% | **51.7%** | +39.7% |
| `exist` | 38.0% | **78.3%** | +40.3% |
| `comparison` | 42.0% | **68.3%** | +26.3% |

## DriveLM Generative Benchmark (Official NLG Metrics)

Evaluated across perception, prediction, and planning questions on DriveLM validation:

| Task / Category | Samples | BLEU-4 | ROUGE-L | CIDEr | METEOR |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Overall DriveLM** | **200** | **36.37%** | **64.37%** | **2.422** | **43.59%** |
| `perception` (objects & scene layout) | 67 | 34.33% | 67.74% | 2.528 | 49.01% |
| `prediction` (future vehicle behavior) | 67 | 51.11% | 58.83% | 2.237 | 37.32% |
| `planning` (ego vehicle decision-making) | 66 | 32.07% | 66.58% | 2.411 | 44.46% |

## Setup & Inference

```bash
cd BEV-VQA
# Creazione ambiente e installazione
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .

# 1. Inferenza Interattiva (Demo REPL da terminale)
python scripts/run_inference.py --interactive

# 2. Query Singola su una specifica scena NuScenes
python scripts/run_inference.py --token <sample_token> --question "Are there any cars ahead?"

# 3. Valutazione su Campione di Validazione
python scripts/run_inference.py --sample-val --num-samples 5

# 4. Valutazione Generativa DriveLM (Metriche NLG)
python scripts/evaluate_drivelm_nlg.py --num-samples 200
```

## Checkpoints

- **Stage 1 Pretrained Projector:** `checkpoints/stage1/stage1_projector_best.pt`
- **Stage 2 Best VLM Model:** `checkpoints/stage2/best_model/` (`projector.pt` + `lora_adapters/`)
- **Stage 2 Latest VLM Model:** `checkpoints/stage2/latest_model/`

## Key Design Decisions & Findings

1. **LLM**: `Qwen2.5-3B-Instruct` ($d_{llm}=2048$, 36 transformer layers).
2. **Projector**: `DeeperConvProjector` with 32 visual tokens and 2D spatial positional embeddings. Kept in `float32` during training to prevent AdamW gradient underflow / NaNs.
3. **Sequence Length**: Bounded to 220 tokens to guarantee peak VRAM remains under 8.3 GB on an 11 GB NVIDIA RTX 2080 Ti.
4. **Gradient Checkpointing**: Activated on the LLM base model during Stage 2 LoRA finetuning, maintaining stable memory headroom (>2.2 GB free).
5. **Two-Stage Training**: Stage 1 (BEV $\rightarrow$ Scene Description pretraining) aligns continuous BEV tokens with language space. Stage 2 (LoRA + Projector joint finetuning) achieves 62.0% VQA accuracy on NuScenes-QA + DriveLM.

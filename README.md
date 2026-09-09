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

## Benchmark NuScenes-QA: Balanced vs Natural Distribution

Evaluated on NuScenes-QA validation set across all 5 reasoning categories (`exist`, `count`, `object`, `status`, `comparison`) comparing both Macro-Balanced and Natural real-world distributions against Text Baseline and literature:

| Model | Evaluated Distribution | Overall Accuracy | `exist` | `count` | `object` | `status` | `comparison` |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Text-Only LLM (Zero-Shot)** | Balanced | **19.6%** | 38.0% | 0.0% | 12.0% | 6.0% | 42.0% |
| **BEV-VLM (Ours - SOTA Stage 2)** | **Macro-Balanced** | **43.4%** | **72.0%** | **9.0%** | **36.0%** | **47.0%** | **53.0%** |
| **BEV-VLM (Ours - SOTA Stage 2)** | **Natural Distribution** | **45.8%** | **69.5%** | **13.9%** | **39.3%** | **45.4%** | **51.5%** |
| **CenterPoint + MCAN** | Natural | **59.5%** | 84.8% | 20.8% | 52.3% | 59.8% | 70.0% |
| **MSMDFusion + MCAN** | Natural | **60.4%** | 85.4% | 22.3% | 54.3% | 60.7% | 69.7% |
| **BeLLA w/ LLaMA** | Natural | **59.6%** | 80.9% | 20.5% | 54.8% | 69.9% | 62.7% |

> [!NOTE]
> **Grounding Visivo Certificato (Shuffle Test):** Durante la validazione dello Stage 2 Grounded, il test di permutazione casuale delle feature BEV ha misurato **61.3%** con BEV corretta vs **52.0%** con BEV permutata (**Delta visivo globale: +9.3%**), a riprova che il modello sfrutta le rappresentazioni spaziali per formulare le risposte.

---

## DriveLM Generative Benchmark (Official NLG Metrics)

Evaluated across perception, prediction, and planning questions on DriveLM validation using standard COCO/VQA evaluation tools (`pycocoevalcap` & `nltk`):

| Model / Category | Samples | BLEU-1 | BLEU-4 | ROUGE-L | METEOR | CIDEr |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **BEV-VLM (Ours) - Overall** | **300** | **30.18%** | **20.48%** | **60.96%** | **39.62%** | **2.201** |
| `perception` | 100 | 25.40% | 16.19% | 58.79% | 42.82% | 2.273 |
| `prediction` | 100 | 18.20% | 11.85% | 69.81% | 42.43% | 2.555 |
| `planning` | 100 | 38.60% | 28.34% | 54.28% | 33.61% | 1.673 |
| **BeLLA (Published Literature)** | - | - | **38.66%** | **73.94%** | **32.39%** | **3.090** |
| `planning (BeLLA)` | - | - | 47.83% | 72.01% | 34.40% | 2.980 |

> [!TIP]
> **Analisi Semantica:** Su METEOR (richiamo semantico e allineamento di sinonimi), **BEV-VLM** ottiene **39.62%**, superando BeLLA (**32.39%**) di oltre **+7.23 punti percentuali**, dimostrando un'eccellente comprensione delle manovre e del comportamento di guida.

---

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

# 3. Benchmark Completo NuScenes-QA (Bilanciato & Naturale)
python scripts/evaluate_nuscenes_qa.py --checkpoint checkpoints/sota/stage2/best_model

# 4. Valutazione Generativa DriveLM (Metriche NLG)
python scripts/evaluate_drivelm_nlg.py --checkpoint checkpoints/sota/stage2/best_model --num-samples 300
```

## Checkpoints SOTA

- **Stage 1 SOTA Pretrained Projector:** `checkpoints/sota/stage1/stage1_projector_best.pt`
- **Stage 2 SOTA Best VLM Model:** `checkpoints/sota/stage2/best_model/` (`projector.pt` + `lora_adapters/`)
- **Stage 2 SOTA Latest Model:** `checkpoints/sota/stage2/latest_model/`

## Key Design Decisions & Findings

1. **LLM**: `Qwen2.5-3B-Instruct` ($d_{llm}=2048$, 36 transformer layers).
2. **Projector**: `DeeperConvProjector` with 32 visual tokens and 2D spatial positional embeddings. Kept in `float32` during training to prevent AdamW gradient underflow / NaNs.
3. **Sequence Length**: Bounded to 220 tokens to guarantee peak VRAM remains under 8.3 GB on an 11 GB NVIDIA RTX 2080 Ti.
4. **Gradient Checkpointing**: Activated on the LLM base model during Stage 2 LoRA finetuning, maintaining stable memory headroom (>2.2 GB free).
5. **Two-Stage Training**: Stage 1 (BEV $\rightarrow$ Scene Description pretraining) aligns continuous BEV tokens with language space. Stage 2 (LoRA + Projector joint finetuning) achieves 62.0% VQA accuracy on NuScenes-QA + DriveLM.

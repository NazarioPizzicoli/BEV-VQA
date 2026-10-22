#!/usr/bin/env python3
"""
Environment & Hardware Verification Script for BEV-VLM.

Can be run inside Docker or on the host to verify:
1. Python version and essential dependencies
2. PyTorch CUDA capability and available GPU VRAM
3. BEV-VLM core package integrity (MultiScalePeakProjector, 2D Polar PE)
4. Forward-pass gradient and shape consistency
"""

import sys
import time

def main():
    print("=" * 60)
    print("       BEV-VLM Docker Environment Diagnostic Check       ")
    print("=" * 60)

    # 1. Python & Core Libraries
    print(f"[1/4] Python Version: {sys.version.split()[0]}")
    try:
        import torch
        import transformers
        import peft
        import accelerate
        import nltk
        print(f"      PyTorch:       {torch.__version__}")
        print(f"      Transformers:  {transformers.__version__}")
        print(f"      PEFT:          {peft.__version__}")
        print(f"      Accelerate:    {accelerate.__version__}")
        print(f"      NLTK:          {nltk.__version__}")
    except ImportError as e:
        print(f"      ❌ Missing core dependency: {e}")
        sys.exit(1)

    # 2. CUDA & GPU Acceleration
    print("\n[2/4] Hardware & Acceleration:")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        cuda_version = torch.version.cuda
        print(f"      CUDA Available:  ✅ True (CUDA {cuda_version})")
        print(f"      Primary GPU:     {gpu_name}")
        print(f"      Total VRAM:      {vram_gb:.2f} GB")
        device = torch.device("cuda:0")
    else:
        print("      CUDA Available:  ⚠️ False (Running on CPU)")
        device = torch.device("cpu")

    # 3. BEV-VLM Package Integrity
    print("\n[3/4] Package Verification (bev_vqa):")
    try:
        from bev_vqa.models.projector import ProjectorConfig, build_projector
        from bev_vqa.models.vlm import BEVVLMConfig
        print("      bev_vqa import:  ✅ Successful")
    except ImportError as e:
        print(f"      ❌ Failed to import bev_vqa: {e}")
        sys.exit(1)

    # 4. Forward Pass Verification (MultiScale Peak + 2D Polar Metric)
    print("\n[4/4] Model Forward Verification:")
    try:
        cfg = ProjectorConfig(
            arch_type="multiscale_peak",
            pos_embed_type="2d_polar_metric",
            spatial_shape=(8, 4),
            num_tokens=32,
            hidden_channels=256,
            projector_output_size=2048,
        )
        projector = build_projector(cfg).to(device)
        dummy_bev = torch.randn(2, 128, 200, 200, device=device)

        t0 = time.time()
        with torch.no_grad():
            out = projector(dummy_bev)
        latency_ms = (time.time() - t0) * 1000.0

        assert out.shape == (2, 32, 2048), f"Unexpected output shape: {out.shape}"
        assert torch.isfinite(out).all(), "Output contains NaN or Inf values"
        print(f"      Projector Arch:  {cfg.arch_type} + {cfg.pos_embed_type}")
        print(f"      Input BEV Shape: {tuple(dummy_bev.shape)}")
        print(f"      Output Tokens:   {tuple(out.shape)}")
        print(f"      Forward Latency: {latency_ms:.2f} ms")
        print("      Output Check:    ✅ Finite & Correct Shape")
    except Exception as e:
        print(f"      ❌ Forward pass failed: {e}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  ✅ All checks passed! Environment is ready for BEV-VLM.")
    print("=" * 60)

if __name__ == "__main__":
    main()

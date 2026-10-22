#!/usr/bin/env python3
"""
BEV-VLM Interactive Inference & Visual Debugging Pipeline.

Supports:
1. Interactive REPL Mode (default):
   - Browse nuScenes samples or jump to a sample_token.
   - ANSI truecolor previews of camera surround mosaic and BEV feature grid.
   - Select from pre-indexed DriveLM / NuScenes-QA questions with Ground Truth.
   - Ask custom free-form natural language questions.
   - Real-time multimodal generation with BEV-VLM.
   - Automatic generation of high-res debug dashboard PNGs (150 DPI) in outputs/inference/.
   - Optional desktop GUI visualization via --open-gui (xdg-open).

2. Headless / Batch Mode:
   - CLI execution with --token, --random, --question, --question-idx.
   - Options: --no-viz, --terminal-viz, --output-image <path>, --open-gui.
"""

import argparse
from collections import defaultdict
import datetime
import json
import logging
import os
from bev_vqa.utils.io import get_default_paths

_PATHS = get_default_paths()

from bev_vqa.utils.seed import set_seed
from bev_vqa.constants import Colors
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import torch

# Ensure REPO_ROOT is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from bev_vqa.data.dataset import build_tokenizer, normalize_answer, BEV_TOKEN, VQA_SYSTEM_PROMPT
from bev_vqa.models.projector import ProjectorConfig
from bev_vqa.models.vlm import BEVVLM, BEVVLMConfig
from bev_vqa.utils.visualization import (
    build_multimodal_debug_figure,
    render_ansi_terminal_image,
    render_ansi_bev_grid,
    render_ansi_camera_mosaic,
    compute_bev_l2_heatmap,
    SURROUND_CAMERAS,
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("inference_interactive")

# ANSI Color codes for rich terminal styling



BLUE = "\033[94m"
MAGENTA = "\033[95m"
RED = "\033[91m"

DIM = "\033[2m"


DRIVELM_SYSTEM_PROMPT = (
    "You are an autonomous driving perception and planning assistant. "
    "Analyze the bird's-eye-view (BEV) features of the scene and provide accurate, "
    "concise, and clear answers regarding perception, prediction, and planning decisions."
)

DEFAULT_CHECKPOINT = "checkpoints/ablation_pos_embed/2d_polar_metric/drivelm/best_model"
DEFAULT_CAM_MAP_PATH = os.environ.get(
    "NUSCENES_CAM_MAP", "/home/nazario.pizzicoli/nuscenes/.cache_cam_map_v1.0-trainval.json"
)
# Robust paths across host and Docker container
_BEV_BASE = os.environ.get("BEV_VQA_BEV_DIR", os.environ.get("BEV_FEATURES_DIR", "/data/dataset/bev_features"))
_DATASETS_BASE = os.environ.get("BEV_VQA_DATASETS_DIR", "/data/vqa_datasets/unified")

DEFAULT_DRIVELM_JSON = os.environ.get(
    "DRIVELM_VAL_JSON",
    os.path.join(_DATASETS_BASE, "drivelm_val.json")
    if os.path.exists(os.path.join(_DATASETS_BASE, "drivelm_val.json"))
    else "/media/nazario.pizzicoli/Datos/vqa_datasets/unified/drivelm_val.json",
)
DEFAULT_NUSCENES_QA_JSON = os.environ.get(
    "NUSCENES_QA_VAL_JSON",
    os.path.join(_DATASETS_BASE, "nuscenes_qa_val.json")
    if os.path.exists(os.path.join(_DATASETS_BASE, "nuscenes_qa_val.json"))
    else "/media/nazario.pizzicoli/Datos/vqa_datasets/unified/nuscenes_qa_val.json",
)
DEFAULT_BEV_DIRS = [
    d for d in [
        os.environ.get("BEV_VQA_BEV_DIR"),
        os.environ.get("BEV_FEATURES_DIR"),
        os.path.join(_BEV_BASE, "val"),
        os.path.join(_BEV_BASE, "train"),
        "/data/dataset/bev_features/val",
        "/data/dataset/bev_features/train",
        "/data/dataset_veh/bev_features_veh/val",
        "/data/dataset_veh/bev_features_veh/train",
        "/media/nazario.pizzicoli/Datos/dataset_veh/bev_features_veh/val",
        "/media/nazario.pizzicoli/Datos/dataset_veh/bev_features_veh/train",
        "/media/nazario.pizzicoli/Datos/dataset/bev_features/val",
        "/media/nazario.pizzicoli/Datos/dataset/bev_features/train",
    ] if d is not None and os.path.exists(d)
]


class BEVInferencePipeline:
    """Manages data loading, indexing, BEV-VLM inference, and visual debugging."""

    def __init__(
        self,
        checkpoint_dir: str = DEFAULT_CHECKPOINT,
        llm_path: str = "Qwen/Qwen2.5-3B-Instruct",
        arch_type: Optional[str] = None,
        pos_embed_type: Optional[str] = None,
        num_tokens: int = 32,
        spatial_shape: Optional[Tuple[int, int]] = None,
        use_dual_pooling: Optional[bool] = None,
        use_l2_pooling: Optional[bool] = None,
        l2_pooling_gamma: float = 1.0,
        bev_dirs: Optional[List[str]] = None,
        cam_map_path: str = DEFAULT_CAM_MAP_PATH,
        drivelm_json: str = DEFAULT_DRIVELM_JSON,
        nuscenes_qa_json: str = DEFAULT_NUSCENES_QA_JSON,
        device: Optional[str] = None,
        mock: bool = False,
    ):
        self.checkpoint_dir = checkpoint_dir
        self.llm_path = llm_path
        self.arch_type = arch_type
        self.pos_embed_type = pos_embed_type
        self.num_tokens = num_tokens
        self.spatial_shape = tuple(spatial_shape) if spatial_shape is not None else None
        self.use_dual_pooling = use_dual_pooling
        self.use_l2_pooling = use_l2_pooling
        self.l2_pooling_gamma = l2_pooling_gamma
        self.cam_map_path = cam_map_path
        self.drivelm_json = drivelm_json
        self.nuscenes_qa_json = nuscenes_qa_json
        self.mock = mock

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        if bev_dirs is None:
            self.bev_dirs = [Path(p) for p in DEFAULT_BEV_DIRS]
        else:
            self.bev_dirs = [Path(p) for p in bev_dirs]

        # 1. Index available BEV feature files
        self.available_bevs: Dict[str, Path] = {}
        for b_dir in self.bev_dirs:
            if b_dir.exists():
                for pt_file in b_dir.glob("*.pt"):
                    if pt_file.stem not in self.available_bevs:
                        self.available_bevs[pt_file.stem] = pt_file

        # 2. Load NuScenes camera map cache
        self.cam_map: Dict[str, Dict[str, str]] = {}
        if Path(self.cam_map_path).exists():
            try:
                with open(self.cam_map_path, "r") as f:
                    self.cam_map = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load camera map cache from {self.cam_map_path}: {e}")

        # 3. Pre-index questions from DriveLM and NuScenes-QA by sample_token
        self.questions_by_token: Dict[str, List[Dict]] = defaultdict(list)
        self._index_questions()

        # 4. Initialize model and tokenizer (unless running in mock mode)
        self.tokenizer = None
        self.model = None
        if not self.mock:
            self._init_model()

    def _index_questions(self):
        """Indexes questions from DriveLM and NuScenes-QA by sample_token."""
        # DriveLM
        if Path(self.drivelm_json).exists():
            try:
                with open(self.drivelm_json, "r") as f:
                    drivelm_data = json.load(f).get("questions", [])
                for item in drivelm_data:
                    tok = item.get("sample_token")
                    if tok:
                        ttype = item.get("template_type", "planning")
                        self.questions_by_token[tok].append({
                            "source": "DriveLM",
                            "category": f"DriveLM - {ttype.capitalize()}",
                            "question": item.get("question", ""),
                            "answer": item.get("answer", item.get("templated_answer", "")),
                            "templated_answer": item.get("templated_answer", item.get("answer", "")),
                        })
            except Exception as e:
                logger.warning(f"Error reading DriveLM val JSON {self.drivelm_json}: {e}")

        # NuScenes-QA
        if Path(self.nuscenes_qa_json).exists():
            try:
                with open(self.nuscenes_qa_json, "r") as f:
                    nusc_data = json.load(f).get("questions", [])
                for item in nusc_data:
                    tok = item.get("sample_token")
                    if tok:
                        ttype = item.get("template_type", "exist")
                        self.questions_by_token[tok].append({
                            "source": "NuScenes-QA",
                            "category": f"NuScenes-QA - {ttype.capitalize()}",
                            "question": item.get("question", ""),
                            "answer": item.get("answer", item.get("templated_answer", "")),
                            "templated_answer": item.get("templated_answer", item.get("answer", "")),
                        })
            except Exception as e:
                logger.warning(f"Error reading NuScenes-QA val JSON {self.nuscenes_qa_json}: {e}")

    def _init_model(self):
        """Initializes tokenizer and BEVVLM architecture, and loads weights."""
        print(f"{Colors.CYAN}Loading tokenizer from: {self.llm_path}...{Colors.RESET}")
        self.tokenizer = build_tokenizer(self.llm_path)

        # Detect LoRA parameters from checkpoint if available
        lora_r = 32
        lora_alpha = 64
        lora_cfg_path = Path(self.checkpoint_dir) / "lora_adapters" / "adapter_config.json"
        if lora_cfg_path.exists():
            try:
                with open(lora_cfg_path) as f:
                    lcfg = json.load(f)
                lora_r = lcfg.get("r", lora_r)
                lora_alpha = lcfg.get("lora_alpha", lora_alpha)
            except Exception:
                pass

        # Auto-detect projector configuration from checkpoint if not explicitly provided
        detected_spatial_shape = (8, 4)
        detected_dual_pooling = False
        detected_l2_pooling = False
        detected_arch_type = "multiscale_peak"
        detected_pos_embed_type = "2d_polar_metric"
        proj_weights_path = Path(self.checkpoint_dir) / "projector.pt"
        if proj_weights_path.exists():
            try:
                sd = torch.load(proj_weights_path, map_location="cpu")
                if "pool_shape_tensor" in sd:
                    detected_spatial_shape = tuple(sd["pool_shape_tensor"].tolist())
                if "lateral_scale" in sd:
                    detected_arch_type = "multiscale_peak"
                elif "sparse_embed" in sd:
                    detected_arch_type = "hybrid_sparse"
                elif "cross_attn.in_proj_weight" in sd:
                    detected_arch_type = "question_aware"
                elif "pool_proj.weight" in sd:
                    detected_arch_type = "dual_pooling"
                    detected_dual_pooling = True

                if "polar_pe" in sd:
                    detected_pos_embed_type = "2d_polar_metric"
                elif "cartesian_pe" in sd:
                    detected_pos_embed_type = "2d_cartesian_metric"
                elif "row_embed" in sd:
                    detected_pos_embed_type = "2d_separable_learned"
                elif "pos_embed" in sd:
                    detected_pos_embed_type = "1d_learned"

                if "l2_gamma" in sd or "l2_pooling" in str(self.checkpoint_dir).lower():
                    detected_l2_pooling = True
            except Exception as e:
                logger.warning(f"Could not inspect projector.pt for metadata: {e}")

        arch_type = self.arch_type if self.arch_type is not None else detected_arch_type
        pos_embed_type = self.pos_embed_type if self.pos_embed_type is not None else detected_pos_embed_type
        spatial_shape = self.spatial_shape if self.spatial_shape is not None else detected_spatial_shape
        use_dual_pooling = self.use_dual_pooling if self.use_dual_pooling is not None else detected_dual_pooling
        use_l2_pooling = self.use_l2_pooling if self.use_l2_pooling is not None else detected_l2_pooling
        num_tokens = (spatial_shape[0] * spatial_shape[1]) if spatial_shape is not None else self.num_tokens

        grid_desc = f"{spatial_shape[0]}x{spatial_shape[1]} ({num_tokens} tokens)" if spatial_shape else f"{num_tokens} tokens"
        print(f"{Colors.CYAN}Building BEVVLM ({arch_type}, {pos_embed_type}, Grid: {grid_desc}, LoRA r={lora_r})...{Colors.RESET}")
        cfg = BEVVLMConfig(
            llm_name_or_path=self.llm_path,
            projector_config=ProjectorConfig(
                arch_type=arch_type,
                pos_embed_type=pos_embed_type,
                num_tokens=num_tokens,
                spatial_shape=spatial_shape,
                use_dual_pooling=use_dual_pooling,
                use_l2_pooling=use_l2_pooling,
                l2_pooling_gamma=self.l2_pooling_gamma,
            ),
            lora_r=lora_r,
            lora_alpha=lora_alpha,
        )
        self.model = BEVVLM(cfg, self.tokenizer).to(self.device)
        self.model.projector.float()

        ckpt_path = Path(self.checkpoint_dir)
        if ckpt_path.exists():
            print(f"{Colors.GREEN}Loading checkpoint weights from: {self.checkpoint_dir}...{Colors.RESET}")
            self.model.load_checkpoint(str(ckpt_path))
        else:
            print(f"{Colors.YELLOW}WARNING: Checkpoint {self.checkpoint_dir} not found. Running with uninitialized weights!{Colors.RESET}")

        self.model.eval()
        print(f"{Colors.BOLD}{Colors.GREEN}✓ BEV-VLM Engine initialized on {self.device}{Colors.RESET}\n")

    def get_camera_paths(self, sample_token: str) -> Dict[str, str]:
        """Returns camera paths dictionary for a sample token."""
        return self.cam_map.get(sample_token, {})

    def get_questions_for_sample(self, sample_token: str) -> List[Dict]:
        """Returns pre-indexed questions for a sample token."""
        return self.questions_by_token.get(sample_token, [])

    def find_bev_path(self, sample_token: str) -> Optional[Path]:
        """Finds BEV feature file path on disk."""
        if sample_token in self.available_bevs:
            return self.available_bevs[sample_token]
        for b_dir in self.bev_dirs:
            p = b_dir / f"{sample_token}.pt"
            if p.exists():
                self.available_bevs[sample_token] = p
                return p
        return None

    def load_bev_tensor(self, sample_token: str) -> Optional[torch.Tensor]:
        """Loads BEV feature tensor [1, 128, 200, 200]."""
        p = self.find_bev_path(sample_token)
        if p is None or not p.exists():
            return None
        data = torch.load(p, map_location="cpu", weights_only=True)
        feat = data["features_fused"]
        if feat.dim() == 3:
            feat = feat.unsqueeze(0)
        return feat.to(self.device)

    def answer_question(
        self,
        sample_token: str,
        question: str,
        category: Optional[str] = None,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
    ) -> Dict:
        """
        Executes multimodal generation for a given sample token and question.
        """
        if self.mock:
            # Deterministic mock generation for testing/CI
            q_lower = question.lower()
            if "pedestrian" in q_lower:
                ans = "No, there are no pedestrians in front of the vehicle."
            elif "car" in q_lower or "vehicle" in q_lower:
                ans = "There are vehicles parked on the right side."
            elif "action" in q_lower or "drive" in q_lower:
                ans = "The ego vehicle should continue straight and maintain speed."
            else:
                ans = "Clear autonomous driving conditions observed."
            return {
                "success": True,
                "sample_token": sample_token,
                "question": question,
                "raw_answer": ans,
                "normalized_answer": normalize_answer(ans),
                "latency_ms": 15.0,
                "vram_allocated_gb": 0.0,
            }

        bev = self.load_bev_tensor(sample_token)
        if bev is None:
            return {
                "success": False,
                "error": f"BEV tensor not found for token: {sample_token}",
            }

        # Choose appropriate system prompt based on checkpoint or category
        if "nuscenes" in self.checkpoint_dir.lower() or (category and "nuscenes" in category.lower()):
            sys_prompt = VQA_SYSTEM_PROMPT
        else:
            sys_prompt = DRIVELM_SYSTEM_PROMPT

        bev_placeholder = BEV_TOKEN * self.num_tokens
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"{bev_placeholder}\n{question}"},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)

        autocast_ctx = (
            torch.amp.autocast("cuda", dtype=torch.float16)
            if self.device.type == "cuda"
            else torch.amp.autocast("cpu", dtype=torch.bfloat16)
        )

        t0 = time.time()
        with torch.no_grad(), autocast_ctx:
            gen_ids = self.model.generate(
                bev=bev,
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature if temperature > 0.0 else None,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        elapsed_ms = (time.time() - t0) * 1000.0

        raw_answer = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()
        norm_answer = normalize_answer(raw_answer)

        vram_gb = (
            round(torch.cuda.memory_allocated(self.device) / 1e9, 2)
            if torch.cuda.is_available()
            else 0.0
        )

        return {
            "success": True,
            "sample_token": sample_token,
            "question": question,
            "raw_answer": raw_answer,
            "normalized_answer": norm_answer,
            "latency_ms": round(elapsed_ms, 1),
            "vram_allocated_gb": vram_gb,
        }

    def render_visual_previews(self, sample_token: str, bev_tensor: Optional[torch.Tensor] = None):
        """Prints ANSI truecolor previews of BEV heatmap and camera surround mosaic."""
        if bev_tensor is None:
            bev_tensor = self.load_bev_tensor(sample_token)

        cam_paths = self.get_camera_paths(sample_token)

        print(f"\n{Colors.BOLD}{Colors.CYAN}--- [TERMINAL PREVIEW: 6 SURROUND CAMERAS] ---{Colors.RESET}")
        try:
            mosaic_ansi = render_ansi_camera_mosaic(cam_paths, width=72)
            print(mosaic_ansi)
        except Exception as e:
            print(f"{Colors.YELLOW}[Camera mosaic preview unavailable: {e}]{Colors.RESET}")

        if bev_tensor is not None:
            print(f"\n{Colors.BOLD}{Colors.CYAN}--- [TERMINAL PREVIEW: BEV FEATURE HEATMAP (L2 NORM)] ---{Colors.RESET}")
            try:
                bev_ansi = render_ansi_bev_grid(bev_tensor, width=44)
                print(bev_ansi)
            except Exception as e:
                print(f"{Colors.YELLOW}[BEV ANSI grid preview unavailable: {e}]{Colors.RESET}")
        print()


def run_headless_inference(
    pipeline: BEVInferencePipeline,
    token: Optional[str] = None,
    is_random: bool = False,
    question: Optional[str] = None,
    question_idx: Optional[int] = None,
    output_image: Optional[str] = None,
    no_viz: bool = False,
    terminal_viz: bool = False,
    open_gui: bool = False,
    max_new_tokens: int = 64,
) -> Dict:
    """Executes headless / batch inference."""
    # Resolve token
    if is_random or token is None:
        rich_tokens = [t for t in pipeline.available_bevs.keys() if t in pipeline.questions_by_token]
        if (question is None or question_idx is not None) and rich_tokens:
            token = random.choice(rich_tokens)
        else:
            valid_tokens = list(pipeline.available_bevs.keys())
            if not valid_tokens:
                raise RuntimeError("No BEV feature files found to sample from.")
            token = random.choice(valid_tokens)

    # Resolve question and ground truth
    sample_questions = pipeline.get_questions_for_sample(token)
    selected_gt = None
    selected_cat = None

    if question:
        # User supplied custom question string
        selected_q = question
        for sq in sample_questions:
            if sq["question"].strip().lower() == question.strip().lower():
                selected_gt = sq.get("answer", sq.get("templated_answer"))
                selected_cat = sq.get("category")
                break
    elif question_idx is not None:
        if 1 <= question_idx <= len(sample_questions):
            sq = sample_questions[question_idx - 1]
            selected_q = sq["question"]
            selected_gt = sq.get("answer", sq.get("templated_answer"))
            selected_cat = sq.get("category")
        else:
            logger.warning(
                f"Question index {question_idx} out of range for sample {token} (has {len(sample_questions)} questions)."
            )
            if sample_questions:
                sq = sample_questions[0]
                selected_q = sq["question"]
                selected_gt = sq.get("answer", sq.get("templated_answer"))
                selected_cat = sq.get("category")
            else:
                selected_q = "Describe the autonomous driving scene around the ego vehicle."
    elif sample_questions:
        # Default to first question if present
        sq = sample_questions[0]
        selected_q = sq["question"]
        selected_gt = sq.get("answer", sq.get("templated_answer"))
        selected_cat = sq.get("category")
    else:
        selected_q = "Describe the autonomous driving scene around the ego vehicle."

    # Load BEV tensor
    if pipeline.mock:
        bev = torch.randn(1, 128, 200, 200)
    else:
        bev = pipeline.load_bev_tensor(token)
        if bev is None:
            raise FileNotFoundError(f"BEV feature tensor not found for token: {token}")

    # Terminal preview if requested
    if terminal_viz:
        pipeline.render_visual_previews(token, bev_tensor=bev)

    # Forward generation
    res = pipeline.answer_question(
        sample_token=token,
        question=selected_q,
        category=selected_cat,
        max_new_tokens=max_new_tokens,
    )

    if not res["success"]:
        print(f"{RED}Inference failed: {res.get('error')}{Colors.RESET}")
        return res

    pred = res["raw_answer"]
    norm_pred = res["normalized_answer"]
    norm_gt = normalize_answer(selected_gt) if selected_gt else None
    is_match = (norm_gt is not None) and ((norm_pred == norm_gt) or (norm_gt in norm_pred.split()) or (norm_gt in norm_pred))

    print(f"\n{Colors.BOLD}{Colors.CYAN}=== BEV-VLM INFERENCE RESULT ==={Colors.RESET}")
    print(f"  {Colors.BOLD}Sample Token:{Colors.RESET}     {token}")
    if selected_cat:
        print(f"  {Colors.BOLD}Category:{Colors.RESET}         {selected_cat}")
    print(f"  {Colors.BOLD}Question:{Colors.RESET}         {selected_q}")
    if selected_gt:
        print(f"  {Colors.BOLD}Ground Truth:{Colors.RESET}     {Colors.GREEN}{selected_gt}{Colors.RESET}")
    print(f"  {Colors.BOLD}Prediction:{Colors.RESET}       {Colors.BOLD}{BLUE}{pred}{Colors.RESET}")
    if selected_gt:
        status_str = f"{Colors.GREEN}✓ MATCH{Colors.RESET}" if is_match else f"{Colors.YELLOW}✗ MISMATCH{Colors.RESET}"
        print(f"  {Colors.BOLD}Evaluation:{Colors.RESET}       {status_str}")
    print(f"  {Colors.BOLD}Latency:{Colors.RESET}          {res['latency_ms']} ms")
    if res.get("vram_allocated_gb", 0) > 0:
        print(f"  {Colors.BOLD}VRAM Used:{Colors.RESET}        {res['vram_allocated_gb']} GB")

    # Generate visual figure unless disabled
    saved_figure_path = None
    if not no_viz:
        cam_paths = pipeline.get_camera_paths(token)
        saved_figure_path = build_multimodal_debug_figure(
            sample_token=token,
            cam_paths=cam_paths,
            bev_tensor=bev,
            question=selected_q,
            prediction=pred,
            ground_truth=selected_gt,
            category=selected_cat,
            output_path=output_image,
        )
        print(f"\n{Colors.BOLD}{Colors.GREEN}✓ Visual debug dashboard saved to:{Colors.RESET} {saved_figure_path}")

        if open_gui and saved_figure_path and Path(saved_figure_path).exists():
            try:
                subprocess.Popen(["xdg-open", str(saved_figure_path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                print(f"  [Opened in system image viewer]")
            except Exception as e:
                logger.warning(f"Failed to open GUI viewer: {e}")

    res["saved_figure"] = saved_figure_path
    res["ground_truth"] = selected_gt
    return res


def run_interactive_session(pipeline: BEVInferencePipeline, open_gui: bool = False, max_new_tokens: int = 64):
    """Interactive REPL session in the terminal."""
    print(f"\n{Colors.BOLD}{MAGENTA}======================================================================{Colors.RESET}")
    print(f"{Colors.BOLD}{MAGENTA}     BEV-VLM INTERACTIVE INFERENCE & VISUAL DEBUGGING PIPELINE        {Colors.RESET}")
    print(f"{Colors.BOLD}{MAGENTA}======================================================================{Colors.RESET}")
    print(f"  Available BEV scenes indexed:       {len(pipeline.available_bevs):,}")
    print(f"  NuScenes camera map scenes:         {len(pipeline.cam_map):,}")
    print(f"  DriveLM / NuScenes-QA Q&A scenes:   {len(pipeline.questions_by_token):,}")
    print(f"  Active checkpoint:                  {pipeline.checkpoint_dir}")
    print(f"  Device:                             {pipeline.device}")
    print(f"{MAGENTA}----------------------------------------------------------------------{Colors.RESET}\n")

    # Prefer scenes that have both BEV and official questions
    rich_tokens = [t for t in pipeline.available_bevs.keys() if t in pipeline.questions_by_token]
    all_tokens = list(pipeline.available_bevs.keys())

    if not all_tokens:
        print(f"{RED}No BEV feature files found. Please check --bev-dirs paths.{Colors.RESET}")
        return

    active_token = None
    while True:
        if active_token is None:
            try:
                token_input = input(
                    f"{Colors.BOLD}{Colors.CYAN}Enter sample_token or press [Enter] for random sample ('q' to quit): {Colors.RESET}"
                ).strip()
            except (KeyboardInterrupt, EOFError):
                print("\nExiting interactive session.")
                break

            if token_input.lower() in ["q", "quit", "exit"]:
                print("Session finished.")
                break

            if token_input:
                target_token = token_input
                if target_token not in pipeline.available_bevs and pipeline.find_bev_path(target_token) is None:
                    print(f"{Colors.YELLOW}Warning: Token '{target_token}' not found in indexed BEV feature files. Try another.{Colors.RESET}\n")
                    continue
            else:
                # Pick random token
                pool = rich_tokens if rich_tokens else all_tokens
                target_token = random.choice(pool)
        else:
            target_token = active_token
            active_token = None

        # Load BEV tensor
        bev = pipeline.load_bev_tensor(target_token) if not pipeline.mock else torch.randn(1, 128, 200, 200)

        # Print scene metadata
        cams = pipeline.get_camera_paths(target_token)
        cam_count = sum(1 for c in SURROUND_CAMERAS if c in cams and Path(cams[c]).exists())
        qs = pipeline.get_questions_for_sample(target_token)

        print(f"\n{Colors.BOLD}{Colors.GREEN}Active Scene Sample Token:{Colors.RESET} {Colors.BOLD}{target_token}{Colors.RESET}")
        print(f"  Cameras found: {cam_count}/6  |  Pre-indexed Q&A items: {len(qs)}")

        # Visual preview in terminal
        pipeline.render_visual_previews(target_token, bev_tensor=bev)

        # Inner loop for asking questions on this scene
        while True:
            print(f"{Colors.BOLD}{Colors.CYAN}Available questions for this scene:{Colors.RESET}")
            if qs:
                for idx, q_item in enumerate(qs, 1):
                    src_tag = f"({q_item['category']})"
                    gt_tag = f"{Colors.GREEN}[GT: {q_item['answer']}]{Colors.RESET}"
                    print(f"  {Colors.BOLD}[{idx}]{Colors.RESET} {src_tag} {q_item['question']} {gt_tag}")
            else:
                print(f"  {DIM}(No pre-indexed questions found for this token in DriveLM / NuScenes-QA){Colors.RESET}")

            print(f"  {Colors.BOLD}[C]{Colors.RESET} Custom question")
            print(f"  {Colors.BOLD}[N]{Colors.RESET} Next random sample")
            print(f"  {Colors.BOLD}[T]{Colors.RESET} Specific sample token")
            print(f"  {Colors.BOLD}[Q]{Colors.RESET} Quit")

            try:
                choice = input(f"\n{Colors.BOLD}Select an option [1-{len(qs)} / C / N / T / Q]: {Colors.RESET}").strip()
            except (KeyboardInterrupt, EOFError):
                return

            if not choice:
                continue

            if choice.lower() in ["q", "quit"]:
                print("Session terminated.")
                return

            if choice.lower() in ["n", "next"]:
                pool = rich_tokens if rich_tokens else all_tokens
                active_token = random.choice(pool)
                print()
                break  # Jump directly to next scene

            if choice.lower() in ["t", "token"]:
                try:
                    new_tok = input(f"{Colors.BOLD}{Colors.CYAN}Enter sample_token: {Colors.RESET}").strip()
                except (KeyboardInterrupt, EOFError):
                    return
                if new_tok:
                    if new_tok in pipeline.available_bevs or pipeline.find_bev_path(new_tok) is not None:
                        active_token = new_tok
                        print()
                        break
                    else:
                        print(f"{Colors.YELLOW}Token '{new_tok}' not found. Staying on current scene.{Colors.RESET}\n")
                        continue
                continue

            selected_q = None
            selected_gt = None
            selected_cat = None

            if choice.lower() == "c":
                try:
                    custom_q = input(f"{Colors.BOLD}{BLUE}Enter your custom question: {Colors.RESET}").strip()
                except (KeyboardInterrupt, EOFError):
                    return
                if not custom_q:
                    continue
                selected_q = custom_q
                selected_cat = "Custom Query"
            elif choice.isdigit():
                q_idx = int(choice)
                if 1 <= q_idx <= len(qs):
                    item = qs[q_idx - 1]
                    selected_q = item["question"]
                    selected_gt = item.get("answer", item.get("templated_answer"))
                    selected_cat = item.get("category")
                else:
                    print(f"{Colors.YELLOW}Invalid option number.{Colors.RESET}\n")
                    continue
            else:
                print(f"{Colors.YELLOW}Invalid selection.{Colors.RESET}\n")
                continue

            # Run inference
            print(f"\n{Colors.CYAN}⚡ Running BEV-VLM Multimodal Generation...{Colors.RESET}")
            res = pipeline.answer_question(
                sample_token=target_token,
                question=selected_q,
                category=selected_cat,
                max_new_tokens=max_new_tokens,
            )

            if not res["success"]:
                print(f"{RED}Generation error: {res.get('error')}{Colors.RESET}\n")
                continue

            pred = res["raw_answer"]
            print(f"\n{Colors.BOLD}{BLUE}BEV-VLM Prediction:{Colors.RESET}  {Colors.BOLD}{pred}{Colors.RESET}")
            if selected_gt:
                print(f"{Colors.BOLD}{Colors.GREEN}Ground Truth:       {Colors.RESET}  {selected_gt}")
                norm_pred = res["normalized_answer"]
                norm_gt = normalize_answer(selected_gt)
                is_match = (norm_pred == norm_gt) or (norm_gt in norm_pred.split()) or (norm_gt in norm_pred)
                match_str = f"{Colors.GREEN}✓ MATCH{Colors.RESET}" if is_match else f"{Colors.YELLOW}✗ MISMATCH{Colors.RESET}"
                print(f"{Colors.BOLD}Evaluation:         {Colors.RESET}  {match_str}")
            print(f"{DIM}Latency: {res['latency_ms']} ms | VRAM: {res.get('vram_allocated_gb', 0)} GB{Colors.RESET}\n")

            # Save Visual Debug Dashboard PNG
            out_img = build_multimodal_debug_figure(
                sample_token=target_token,
                cam_paths=pipeline.get_camera_paths(target_token),
                bev_tensor=bev,
                question=selected_q,
                prediction=pred,
                ground_truth=selected_gt,
                category=selected_cat,
            )
            print(f"{Colors.BOLD}{Colors.GREEN}✓ Dashboard PNG saved to:{Colors.RESET} {out_img}")

            if open_gui and Path(out_img).exists():
                try:
                    subprocess.Popen(["xdg-open", str(out_img)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    print(f"  [Dashboard opened in desktop GUI viewer]")
                except Exception as e:
                    logger.warning(f"Could not open GUI: {e}")

            print(f"{MAGENTA}{'-'*70}{Colors.RESET}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="BEV-VLM Interactive Inference & Visual Debugging Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Checkpoint & Model Options
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT, help="Path to checkpoint directory (containing projector.pt and lora_adapters)")
    parser.add_argument("--llm-path", type=str, default="Qwen/Qwen2.5-3B-Instruct", help="LLM backbone huggingface identifier or path")
    parser.add_argument("--arch-type", type=str, default=None, help="Projector architecture type (auto-detected if omitted)")
    parser.add_argument("--pos-embed-type", type=str, default=None, help="Positional encoding type (auto-detected if omitted)")
    parser.add_argument("--num-tokens", type=int, default=32, help="Number of visual tokens produced by projector")
    parser.add_argument("--spatial-shape", type=int, nargs=2, default=None, metavar=("H", "W"), help="Explicit spatial pooling grid (H, W), e.g. 8 4 (auto-detected if omitted)")
    parser.add_argument("--use-dual-pooling", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable Dual Pooling (auto-detected from checkpoint if omitted)")
    parser.add_argument("--use-l2-pooling", action=argparse.BooleanOptionalAction, default=None, help="Enable/disable L2-norm salience pooling (auto-detected from checkpoint if omitted)")
    parser.add_argument("--l2-pooling-gamma", type=float, default=1.0, help="L2 pooling salience gamma/temperature")
    parser.add_argument("--device", type=str, default=None, help="Device to run on ('cuda', 'cpu')")
    parser.add_argument("--max-new-tokens", type=int, default=64, help="Maximum generated tokens")

    # Dataset & Feature Paths
    parser.add_argument("--cam-map-path", type=str, default=DEFAULT_CAM_MAP_PATH, help="NuScenes camera map JSON cache")
    parser.add_argument("--drivelm-json", type=str, default=DEFAULT_DRIVELM_JSON, help="DriveLM val unified JSON path")
    parser.add_argument("--nuscenes-qa-json", type=str, default=DEFAULT_NUSCENES_QA_JSON, help="NuScenes-QA val unified JSON path")
    parser.add_argument("--bev-dirs", nargs="+", default=DEFAULT_BEV_DIRS, help="Directories containing BEV feature .pt files")

    # Headless / Batch Mode Arguments
    parser.add_argument("--token", type=str, default=None, help="Sample token for single-sample headless inference")
    parser.add_argument("--random", action="store_true", help="Pick a random sample for single-sample headless inference")
    parser.add_argument("--question", type=str, default=None, help="Specific question text in headless mode")
    parser.add_argument("--question-idx", type=int, default=None, help="1-based index of question from sample's pre-indexed Q&A")
    parser.add_argument("--output-image", type=str, default=None, help="Destination file path for saved PNG visual dashboard")
    parser.add_argument("--no-viz", action="store_true", help="Disable generating and saving PNG visual dashboard")
    parser.add_argument("--terminal-viz", action="store_true", help="Display ANSI truecolor BEV grid and camera mosaic in terminal")
    parser.add_argument("--open-gui", action="store_true", help="Launch desktop image viewer via xdg-open on saved PNG dashboard")

    # Interactive & Mock flags
    parser.add_argument("--interactive", action="store_true", help="Launch interactive REPL mode explicitly")
    parser.add_argument("--mock", action="store_true", help="Use mock model generation (instant testing without loading weights)")

    return parser.parse_args()


def main():
    args = parse_args()

    pipeline = BEVInferencePipeline(
        checkpoint_dir=args.checkpoint,
        llm_path=args.llm_path,
        arch_type=args.arch_type,
        pos_embed_type=args.pos_embed_type,
        num_tokens=args.num_tokens,
        spatial_shape=args.spatial_shape,
        use_dual_pooling=args.use_dual_pooling,
        use_l2_pooling=args.use_l2_pooling,
        l2_pooling_gamma=args.l2_pooling_gamma,
        bev_dirs=args.bev_dirs,
        cam_map_path=args.cam_map_path,
        drivelm_json=args.drivelm_json,
        nuscenes_qa_json=args.nuscenes_qa_json,
        device=args.device,
        mock=args.mock,
    )

    is_headless = (
        (args.token is not None) or
        args.random or
        (args.question is not None) or
        (args.question_idx is not None)
    )

    if is_headless and not args.interactive:
        run_headless_inference(
            pipeline=pipeline,
            token=args.token,
            is_random=args.random,
            question=args.question,
            question_idx=args.question_idx,
            output_image=args.output_image,
            no_viz=args.no_viz,
            terminal_viz=args.terminal_viz,
            open_gui=args.open_gui,
            max_new_tokens=args.max_new_tokens,
        )
    else:
        run_interactive_session(
            pipeline=pipeline,
            open_gui=args.open_gui,
            max_new_tokens=args.max_new_tokens,
        )


if __name__ == "__main__":
    main()

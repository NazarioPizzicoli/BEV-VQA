"""DriveLM generative NLG evaluation module."""
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import nltk
import torch
from tqdm import tqdm

from bev_vqa.constants import BEV_TOKEN, DRIVELM_SYSTEM_PROMPT, Colors
from bev_vqa.training.metrics import compute_nlg_metrics
from bev_vqa.utils.io import load_bev_features

logger = logging.getLogger(__name__)


def ensure_nltk_resources():
    """Downloads minimal NLTK resources if not already present."""
    for res in ["wordnet", "punkt", "omw-1.4"]:
        try:
            nltk.download(res, quiet=True)
        except Exception:
            pass


def load_drivelm_data(json_path: str, bev_dirs: List[Path], max_samples: int = 200, seed: int = 42) -> List[Dict]:
    """Loads and samples questions from DriveLM validation with existing BEV.
    
    Args:
        json_path: Path to the JSON file containing the questions.
        bev_dirs: List of directories containing BEV features.
        max_samples: Maximum number of samples to return.
        seed: Random seed for sampling.
        
    Returns:
        List of sampled valid question dictionaries.
    """
    with open(json_path) as f:
        data = json.load(f).get("questions", [])

    available_tokens = set()
    for b_dir in bev_dirs:
        if b_dir.exists():
            for p in b_dir.glob("*.pt"):
                available_tokens.add(p.stem)

    valid_questions = [q for q in data if q.get("sample_token") in available_tokens]
    logger.info(f"DriveLM Val: {len(valid_questions):,} valid questions with BEV out of {len(data):,} total.")

    by_type = defaultdict(list)
    for q in valid_questions:
        by_type[q.get("template_type", "other")].append(q)

    random.seed(seed)
    target_types = ["perception", "prediction", "planning", "behavior"]
    n_per_type = max_samples // len(target_types)

    sampled = []
    for t in target_types:
        pool = by_type.get(t, [])
        sampled.extend(random.sample(pool, min(n_per_type, len(pool))))

    if len(sampled) < max_samples:
        remaining = [q for q in valid_questions if q not in sampled]
        needed = max_samples - len(sampled)
        sampled.extend(random.sample(remaining, min(needed, len(remaining))))

    random.shuffle(sampled)
    return sampled


def run_drivelm_evaluation(
    model,
    tokenizer,
    eval_items: List[Dict],
    bev_dirs: List[Path],
    device: torch.device,
    max_new_tokens: int = 96,
    num_tokens: int = 32,
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], List[Dict]]:
    """Runs the core generative evaluation loop for DriveLM.
    
    Args:
        model: The BEV-VLM model.
        tokenizer: The tokenizer.
        eval_items: List of question items to evaluate.
        bev_dirs: List of directories containing BEV features.
        device: Torch device.
        max_new_tokens: Maximum number of new tokens to generate.
        num_tokens: Number of BEV tokens to insert in the prompt.
        
    Returns:
        Tuple of overall metrics, category metrics, and sample previews.
    """
    model.eval()

    token_to_path = {}
    for item in eval_items:
        tok = item["sample_token"]
        for b_dir in bev_dirs:
            p = b_dir / f"{tok}.pt"
            if p.exists():
                token_to_path[tok] = p
                break

    references = []
    hypotheses = []
    cat_refs = defaultdict(list)
    cat_hyps = defaultdict(list)
    sample_previews = []
    all_predictions = []

    with torch.no_grad():
        for item in tqdm(eval_items, desc="  DriveLM NLG Eval"):
            token = item["sample_token"]
            q = item["question"]
            gt = item["answer"]
            cat = item.get("template_type", "unknown")

            bev_path = token_to_path.get(token)
            if bev_path is None:
                continue

            bev = load_bev_features(token, bev_path.parent, device)
            if bev is None:
                continue

            bev_placeholder = BEV_TOKEN * num_tokens
            messages = [
                {"role": "system", "content": DRIVELM_SYSTEM_PROMPT},
                {"role": "user", "content": f"{bev_placeholder}\n{q}"}
            ]
            prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            tokens = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)

            gen_ids = model.generate(
                bev=bev,
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )

            pred_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

            references.append(gt)
            hypotheses.append(pred_text)
            cat_refs[cat].append(gt)
            cat_hyps[cat].append(pred_text)

            pred_item = {
                "sample_token": token,
                "category": cat,
                "question": q,
                "ground_truth": gt,
                "predicted": pred_text,
            }
            all_predictions.append(pred_item)

            if len(sample_previews) < 6:
                sample_previews.append(pred_item)

            del bev, gen_ids

    overall_metrics = compute_nlg_metrics(references, hypotheses)
    
    category_metrics = {}
    for cat in ["perception", "prediction", "planning", "behavior"]:
        if cat in cat_refs and len(cat_refs[cat]) > 0:
            category_metrics[cat] = compute_nlg_metrics(cat_refs[cat], cat_hyps[cat])

    return overall_metrics, category_metrics, sample_previews, all_predictions


def print_drivelm_results(overall_metrics: Dict[str, float], category_metrics: Dict[str, Dict[str, float]], sample_previews: List[Dict]):
    """Prints formatted DriveLM NLG evaluation results.
    
    Args:
        overall_metrics: Overall computed metrics.
        category_metrics: Metrics computed per category.
        sample_previews: List of sample predictions for preview.
    """
    print(f"\n{Colors.BOLD}{Colors.GREEN}======================================================================{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}  DRIVELM GENERATIVE BENCHMARK RESULTS (NLG METRICS){Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}======================================================================{Colors.RESET}")
    print(f"  BLEU-1:               {Colors.BOLD}{overall_metrics['bleu_1']:.2f}%{Colors.RESET}")
    print(f"  BLEU-2:               {Colors.BOLD}{overall_metrics['bleu_2']:.2f}%{Colors.RESET}")
    print(f"  BLEU-3:               {Colors.BOLD}{overall_metrics['bleu_3']:.2f}%{Colors.RESET}")
    print(f"  BLEU-4:               {Colors.BOLD}{Colors.GREEN}{overall_metrics['bleu_4']:.2f}%{Colors.RESET}")
    print(f"  ROUGE-L:              {Colors.BOLD}{Colors.GREEN}{overall_metrics['rouge_l']:.2f}%{Colors.RESET}")
    print(f"  CIDEr:                {Colors.BOLD}{Colors.GREEN}{overall_metrics['cider']:.3f}{Colors.RESET}")
    print(f"  METEOR:               {Colors.BOLD}{Colors.GREEN}{overall_metrics['meteor']:.2f}%{Colors.RESET}")

    print(f"\n  DriveLM Category Detail:")
    print(f"  {'Category':<15} | {'BLEU-4':<10} | {'ROUGE-L':<10} | {'CIDEr':<10} | {'METEOR':<10}")
    print(f"  {'-'*65}")
    for cat, m in category_metrics.items():
        print(f"  {cat:<15} | {m['bleu_4']:>8.2f}% | {m['rouge_l']:>8.2f}% | {m['cider']:>8.3f}  | {m['meteor']:>8.2f}%")

    print(f"\n{Colors.BOLD}{Colors.CYAN}----------------------------------------------------------------------{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}  DRIVELM GENERATIVE EXAMPLES{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}----------------------------------------------------------------------{Colors.RESET}")
    for idx, prev in enumerate(sample_previews, 1):
        print(f"\n{Colors.BOLD}Example #{idx} [{prev['category'].upper()}]:{Colors.RESET}")
        print(f"  Question:      {prev['question']}")
        print(f"  Ground Truth: {Colors.BOLD}{prev['ground_truth']}{Colors.RESET}")
        print(f"  BEV-VLM:      {Colors.GREEN}{prev['predicted']}{Colors.RESET}")


def evaluate_drivelm_nlg(args) -> Dict:
    """Executes the full DriveLM NLG evaluation pipeline given CLI args."""
    import time
    from pathlib import Path
    import json
    from bev_vqa.data.dataset import build_tokenizer
    from bev_vqa.models.vlm import BEVVLM, BEVVLMConfig
    from bev_vqa.models.projector import ProjectorConfig
    from bev_vqa.utils.seed import set_seed

    ensure_nltk_resources()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bev_base = Path(args.bev_dir)
    bev_dirs = [bev_base / "val", bev_base / "train"]

    spatial_shape = tuple(args.spatial_shape) if getattr(args, "spatial_shape", None) is not None else None
    num_tokens = (spatial_shape[0] * spatial_shape[1]) if spatial_shape is not None else args.num_tokens

    tokenizer = build_tokenizer(args.llm_path)
    cfg = BEVVLMConfig(
        llm_name_or_path=args.llm_path,
        projector_config=ProjectorConfig(
            arch_type=args.arch_type,
            num_tokens=num_tokens,
            spatial_shape=spatial_shape,
            use_dual_pooling=getattr(args, "use_dual_pooling", False),
            use_l2_pooling=getattr(args, "use_l2_pooling", False),
            l2_pooling_gamma=getattr(args, "l2_pooling_gamma", 1.0),
            l2_pooling_learnable=getattr(args, "l2_pooling_learnable", False),
            pos_embed_type=getattr(args, "pos_embed_type", "1d_learned"),
        ),
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )
    model = BEVVLM(cfg, tokenizer).to(device)
    model.projector.float()
    model.load_checkpoint(args.checkpoint)

    eval_items = load_drivelm_data(args.drivelm_val_json, bev_dirs, max_samples=args.num_samples, seed=args.seed)

    t0 = time.time()
    overall, cat_metrics, previews, all_predictions = run_drivelm_evaluation(
        model, tokenizer, eval_items, bev_dirs, device, max_new_tokens=args.max_new_tokens, num_tokens=num_tokens
    )
    elapsed = time.time() - t0

    print_drivelm_results(overall, cat_metrics, previews)

    report = {
        "checkpoint": args.checkpoint,
        "num_evaluated": len(eval_items),
        "overall_metrics": overall,
        "category_metrics": cat_metrics,
        "sample_previews": previews,
        "predictions": all_predictions,
        "latency_seconds": round(elapsed, 2),
    }

    if getattr(args, "llm_judge", False):
        del model
        del tokenizer
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        from bev_vqa.evaluation.llm_judge import LLMJudge, LocalHFBackend, APIJudgeBackend, print_judge_summary
        judge_backend_name = getattr(args, "judge_backend", "local")
        judge_model = getattr(args, "judge_model", "Qwen/Qwen2.5-3B-Instruct")
        if judge_backend_name == "local":
            backend = LocalHFBackend(model_name_or_path=judge_model)
        else:
            backend = APIJudgeBackend(model_name=judge_model)
        judge = LLMJudge(backend=backend, benchmark="drivelm")
        judge_summary, _ = judge.evaluate_dataset(
            samples=all_predictions,
            max_samples=getattr(args, "judge_samples", 100),
        )
        print_judge_summary(judge_summary)
        report["llm_judge"] = judge_summary

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report generativo salvato in: {report_path}")
    return report

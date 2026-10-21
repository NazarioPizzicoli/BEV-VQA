"""NuScenes-QA evaluation module."""
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from bev_vqa.constants import BEV_TOKEN, VQA_SYSTEM_PROMPT, Colors
from bev_vqa.data.dataset import normalize_answer
from bev_vqa.utils.io import load_bev_features

logger = logging.getLogger(__name__)

PUBLISHED_BASELINES = {
    "Text-Only LLM (Zero-Shot)": {
        "overall": 19.6, "exist": 38.0, "count": 0.0, "object": 12.0, "status": 6.0, "comparison": 42.0
    },
    "CenterPoint + MCAN": {
        "overall": 61.2, "exist": 78.4, "count": 48.9, "object": 48.2, "status": 62.1, "comparison": 68.3
    },
    "BeLLA (State-of-the-Art)": {
        "overall": 66.8, "exist": 82.5, "count": 55.4, "object": 53.7, "status": 66.2, "comparison": 76.1
    }
}


def load_qa_data(json_path: str, bev_dir: Path) -> List[Dict]:
    """Loads and filters NuScenes-QA questions with available BEV.
    
    Args:
        json_path: Path to the JSON file containing the questions.
        bev_dir: Directory containing BEV features.
        
    Returns:
        List of valid question dictionaries.
    """
    with open(json_path) as f:
        data = json.load(f).get("questions", [])

    val_bev_dir = bev_dir / "val"
    train_bev_dir = bev_dir / "train"

    valid_questions = []
    for q in data:
        token = q.get("sample_token")
        if (val_bev_dir / f"{token}.pt").exists() or (train_bev_dir / f"{token}.pt").exists():
            valid_questions.append(q)

    logger.info(f"NuScenes-QA Val: {len(valid_questions):,} valid samples found with BEV out of {len(data):,} total.")
    return valid_questions


def run_nuscenes_evaluation(
    model,
    tokenizer,
    items: List[Dict],
    bev_dir: Path,
    device: torch.device,
    desc: str = "Evaluating",
    max_new_tokens: int = 10,
    num_tokens: int = 32,
    shuffled_tokens: Optional[List[str]] = None,
) -> Tuple[float, Dict[str, float], List[Dict]]:
    """Runs the core evaluation loop for NuScenes-QA.
    
    Args:
        model: The BEV-VLM model.
        tokenizer: The tokenizer.
        items: List of question items to evaluate.
        bev_dir: Directory containing BEV features.
        device: Torch device.
        desc: TQDM progress bar description.
        max_new_tokens: Maximum number of new tokens to generate.
        num_tokens: Number of BEV tokens to insert in the prompt.
        shuffled_tokens: Optional list of shuffled tokens for visual grounding delta test.
        
    Returns:
        A tuple of overall accuracy, dictionary of category accuracies, and list of result dictionaries.
    """
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    correct = 0
    total = 0
    cat_correct = defaultdict(int)
    cat_total = defaultdict(int)
    results = []

    with torch.no_grad():
        for idx, it in enumerate(tqdm(items, desc=desc, dynamic_ncols=True)):
            token = it["sample_token"]
            fetch_token = shuffled_tokens[idx] if shuffled_tokens is not None else token
            
            bev = load_bev_features(fetch_token, bev_dir, device)
            if bev is None:
                continue

            q = it["question"]
            gt = normalize_answer(it["answer"])
            cat = it.get("template_type", "unknown")

            bev_placeholder = BEV_TOKEN * num_tokens
            messages = [
                {"role": "system", "content": VQA_SYSTEM_PROMPT},
                {"role": "user", "content": f"{bev_placeholder}\n{q}"}
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            tokens = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(device)

            out = model.generate(
                bev=bev,
                input_ids=tokens["input_ids"],
                attention_mask=tokens["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            pred = normalize_answer(tokenizer.decode(out[0], skip_special_tokens=True).strip())
            is_match = (pred == gt) or (gt in pred.split())

            if is_match:
                correct += 1
                cat_correct[cat] += 1
            total += 1
            cat_total[cat] += 1

            results.append({
                "sample_token": token,
                "category": cat,
                "question": q,
                "ground_truth": gt,
                "prediction": pred,
                "correct": is_match,
            })
            del bev, out

    overall_acc = round((correct / max(total, 1)) * 100.0, 2)
    cat_accs = {}
    for c, tot in cat_total.items():
        cat_accs[c] = round((cat_correct[c] / tot) * 100.0, 2) if tot > 0 else 0.0

    return overall_acc, cat_accs, results


def print_nuscenes_results(bal_acc: float, bal_cat_accs: Dict[str, float], nat_acc: float, nat_cat_accs: Dict[str, float], shuffle_data: Dict):
    """Prints formatted NuScenes-QA evaluation results.
    
    Args:
        bal_acc: Balanced dataset overall accuracy.
        bal_cat_accs: Balanced dataset category accuracies.
        nat_acc: Natural dataset overall accuracy.
        nat_cat_accs: Natural dataset category accuracies.
        shuffle_data: Shuffle test results if run.
    """
    print(f"\n{Colors.BOLD}{Colors.GREEN}========================================================================================{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}  NUSCENES-QA BENCHMARK RESULTS{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.GREEN}========================================================================================{Colors.RESET}")
    print(f"  {Colors.BOLD}Balanced Overall Accuracy (Macro-Avg):{Colors.RESET} {Colors.BOLD}{bal_acc:.2f}%{Colors.RESET}")
    print(f"  {Colors.BOLD}Natural Overall Accuracy (True-Dist):{Colors.RESET}   {Colors.BOLD}{nat_acc:.2f}%{Colors.RESET}")
    
    target_cats = ["exist", "count", "object", "status", "comparison"]
    
    if shuffle_data:
        shuf_acc = shuffle_data.get("shuffled_bev_accuracy", 0.0)
        delta_grounding = shuffle_data.get("visual_grounding_delta", 0.0)
        print(f"  {Colors.BOLD}Shuffle Test Acc (Random Permuted BEV):{Colors.RESET}  {shuf_acc:.2f}%")
        print(f"  {Colors.BOLD}Visual Dependency Delta (Grounding):{Colors.RESET}      {Colors.GREEN if delta_grounding > 0 else Colors.YELLOW}{delta_grounding:+.2f}%{Colors.RESET}")

    print(f"\n  {Colors.BOLD}{'Category':<15} | {'Balanced':<12} | {'Natural':<12} | {'Text-Only LLM':<15} | {'BeLLA SOTA':<12}{Colors.RESET}")
    print(f"  {'-'*75}")
    for cat in target_cats:
        b_acc = bal_cat_accs.get(cat, 0.0)
        n_acc = nat_cat_accs.get(cat, 0.0)
        t_acc = PUBLISHED_BASELINES["Text-Only LLM (Zero-Shot)"].get(cat, 0.0)
        bella = PUBLISHED_BASELINES["BeLLA (State-of-the-Art)"].get(cat, 0.0)
        print(f"  {cat:<15} | {b_acc:>10.1f}% | {n_acc:>10.1f}% | {t_acc:>13.1f}% | {bella:>10.1f}%")
    print(f"  {'-'*75}")
    print(f"  {Colors.BOLD}{'OVERALL':<15} | {bal_acc:>10.1f}% | {nat_acc:>10.1f}% | {19.6:>13.1f}% | {66.8:>10.1f}%{Colors.RESET}")
    print(f"{Colors.GREEN}========================================================================================{Colors.RESET}\n")


def evaluate_nuscenes_qa(args) -> Dict:
    """Executes the full NuScenes-QA evaluation pipeline given CLI args."""
    from bev_vqa.data.dataset import build_tokenizer
    from bev_vqa.models.vlm import BEVVLM, BEVVLMConfig
    from bev_vqa.models.projector import ProjectorConfig
    from bev_vqa.utils.seed import set_seed

    set_seed(args.seed)
    bev_dir = Path(args.bev_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    model.load_checkpoint(args.checkpoint)

    all_questions = load_qa_data(args.nuscenes_val_json, bev_dir)

    # 1. Dataset Bilanciato
    by_cat = defaultdict(list)
    for q in all_questions:
        cat = q.get("template_type", "unknown")
        by_cat[cat].append(q)

    target_cats = ["exist", "count", "object", "status", "comparison"]
    balanced_items = []
    for cat in target_cats:
        pool = by_cat[cat]
        n_take = min(args.balanced_samples_per_cat, len(pool))
        balanced_items.extend(random.sample(pool, n_take))
    random.shuffle(balanced_items)

    logger.info(f"\n>>> 1. Valutazione su Distribuzione Bilanciata ({len(balanced_items)} campioni)...")
    bal_acc, bal_cat_accs, bal_samples = run_nuscenes_evaluation(
        model, tokenizer, balanced_items, bev_dir, device, desc="Balanced Eval", num_tokens=num_tokens
    )

    # 2. Shuffle Test
    shuffle_data = {}
    if getattr(args, "run_shuffle", False):
        tokens = [it["sample_token"] for it in balanced_items]
        shuffled_tokens = tokens.copy()
        random.seed(999)
        random.shuffle(shuffled_tokens)
        for i in range(len(tokens)):
            if tokens[i] == shuffled_tokens[i] and len(tokens) > 1:
                j = (i + 1) % len(tokens)
                shuffled_tokens[i], shuffled_tokens[j] = shuffled_tokens[j], shuffled_tokens[i]

        shuf_acc, shuf_cat_accs, _ = run_nuscenes_evaluation(
            model, tokenizer, balanced_items, bev_dir, device, desc="Shuffle Test", num_tokens=num_tokens, shuffled_tokens=shuffled_tokens
        )
        delta_grounding = round(bal_acc - shuf_acc, 2)
        shuffle_data = {
            "real_bev_accuracy": bal_acc,
            "shuffled_bev_accuracy": shuf_acc,
            "visual_grounding_delta": delta_grounding,
            "shuffled_category_accuracies": shuf_cat_accs,
        }

    # 3. Dataset Naturale
    logger.info(f"\n>>> 3. Valutazione su Distribuzione Naturale ({args.natural_samples} campioni)...")
    natural_items = random.sample(all_questions, min(args.natural_samples, len(all_questions)))
    nat_acc, nat_cat_accs, nat_samples = run_nuscenes_evaluation(
        model, tokenizer, natural_items, bev_dir, device, desc="Natural Eval", num_tokens=num_tokens
    )

    print_nuscenes_results(bal_acc, bal_cat_accs, nat_acc, nat_cat_accs, shuffle_data)

    report = {
        "checkpoint": args.checkpoint,
        "balanced_evaluation": {
            "total_samples": len(balanced_items),
            "overall_accuracy": bal_acc,
            "category_accuracies": bal_cat_accs,
            "sample_predictions": bal_samples,
        },
        "natural_evaluation": {
            "total_samples": len(natural_items),
            "overall_accuracy": nat_acc,
            "category_accuracies": nat_cat_accs,
            "sample_predictions": nat_samples,
        },
        "predictions": bal_samples,
        "shuffle_test": shuffle_data,
        "comparisons": PUBLISHED_BASELINES,
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
        judge = LLMJudge(backend=backend, benchmark="nuscenes_qa")
        judge_summary, _ = judge.evaluate_dataset(
            samples=bal_samples,
            max_samples=getattr(args, "judge_samples", 100),
        )
        print_judge_summary(judge_summary)
        report["llm_judge"] = judge_summary

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Report completo salvato in: {report_path}")
    return report

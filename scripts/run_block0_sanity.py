#!/usr/bin/env python3
"""Block 0: Data Sanity Checks.

Verifica l'integrità di tutti i dati prima di iniziare il training.
Esegui con: python scripts/run_block0_sanity.py

Checks:
  1. BEV features: shape, dtype, range, std, no all-zeros
  2. Dataset JSON: schema, required fields
  3. BEV coverage: every sample_token in JSON has a .pt file
  4. Class distribution: template_types and answers
  5. Duplicate check
"""
import json
import logging
import os
import sys
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("block0")

# ── colour helpers ──────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

def _pass(msg: str) -> dict:
    print(f"  {GREEN}✓ PASS{RESET}  {msg}")
    return {"status": "PASS", "detail": msg}

def _fail(msg: str) -> dict:
    print(f"  {RED}✗ FAIL{RESET}  {msg}")
    return {"status": "FAIL", "detail": msg}

def _warn(msg: str) -> dict:
    print(f"  {YELLOW}⚠ WARN{RESET}  {msg}")
    return {"status": "WARN", "detail": msg}


# ── 1. BEV feature checks ──────────────────────────────────────────
def check_bev_features(bev_dir: Path, split: str, n_samples: int = 50) -> dict:
    """Check shape, dtype, range, std of BEV features."""
    split_dir = bev_dir / split
    if not split_dir.exists():
        return _fail(f"Directory {split_dir} does not exist")

    files = sorted([f for f in os.listdir(split_dir) if f.endswith(".pt")])
    if len(files) == 0:
        return _fail(f"No .pt files in {split_dir}")

    sample_files = files[:n_samples]
    shapes, dtypes = set(), set()
    all_zero_count = 0
    mins, maxs, stds = [], [], []

    for fname in tqdm(sample_files, desc=f"  BEV {split}", leave=False):
        data = torch.load(split_dir / fname, map_location="cpu", weights_only=True)
        if "features_fused" not in data:
            return _fail(f"{fname}: missing 'features_fused' key. Keys: {list(data.keys())}")
        t = data["features_fused"].float()
        shapes.add(tuple(t.shape))
        dtypes.add(str(data["features_fused"].dtype))
        mins.append(t.min().item())
        maxs.append(t.max().item())
        stds.append(t.std().item())
        if t.abs().sum().item() == 0:
            all_zero_count += 1

    details = (
        f"{split}: {len(files)} files | shapes={shapes} | dtypes={dtypes} | "
        f"range=[{min(mins):.2f}, {max(maxs):.2f}] | std=[{min(stds):.3f}, {max(stds):.3f}] | "
        f"all_zeros={all_zero_count}/{len(sample_files)}"
    )
    if len(shapes) != 1:
        return _fail(f"Inconsistent shapes: {shapes}")
    if all_zero_count > 0:
        return _warn(details)
    return _pass(details)


# ── 2. Dataset JSON checks ──────────────────────────────────────────
def check_dataset_json(json_path: Path, key: str = "questions") -> dict:
    """Check JSON schema and required fields."""
    if not json_path.exists():
        return _fail(f"File not found: {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    if "info" not in data:
        return _fail(f"Missing 'info' key")
    if key not in data:
        return _fail(f"Missing '{key}' key")

    items = data[key]
    if len(items) == 0:
        return _fail(f"Empty '{key}' list")

    # Check required fields
    if key == "questions":
        required = {"sample_token", "question", "answer"}
    else:  # descriptions
        required = {"sample_token", "description"}

    sample = items[0]
    missing = required - set(sample.keys())
    if missing:
        return _fail(f"Missing fields in first item: {missing}")

    # Check for empty values
    empty_count = sum(
        1 for item in items
        if any(item.get(f, "") == "" for f in required)
    )

    details = f"{json_path.name}: {len(items)} items | empty_fields={empty_count}"
    if empty_count > 0:
        return _warn(details)
    return _pass(details)


# ── 3. BEV coverage check ──────────────────────────────────────────
def check_bev_coverage(json_path: Path, bev_dir: Path, key: str = "questions") -> dict:
    """Check every sample_token in JSON has a .pt file."""
    with open(json_path) as f:
        data = json.load(f)

    items = data[key]
    tokens = set(item["sample_token"] for item in items)

    # Check both train and val
    bev_tokens = set()
    for split in ["train", "val"]:
        split_dir = bev_dir / split
        if split_dir.exists():
            bev_tokens |= set(f[:-3] for f in os.listdir(split_dir) if f.endswith(".pt"))

    matched = tokens & bev_tokens
    missing = tokens - bev_tokens
    pct = 100 * len(matched) / max(len(tokens), 1)

    details = f"{json_path.name}: {len(matched)}/{len(tokens)} tokens matched ({pct:.1f}%)"
    if missing:
        details += f" | {len(missing)} missing"
        if pct < 95:
            return _fail(details)
        return _warn(details)
    return _pass(details)


# ── 4. Class distribution ──────────────────────────────────────────
def check_class_distribution(json_path: Path) -> dict:
    """Show distribution of template_types and answers."""
    with open(json_path) as f:
        data = json.load(f)

    items = data.get("questions", [])
    if not items:
        return _warn(f"{json_path.name}: no questions to analyze")

    types = Counter(q.get("template_type", "?") for q in items)
    answers = Counter(q.get("answer", "?") for q in items)

    type_str = " | ".join(f"{t}:{c}" for t, c in types.most_common(10))
    ans_str = " | ".join(f"{a}:{c}" for a, c in answers.most_common(10))

    details = f"{json_path.name}: types=[{type_str}] | top_answers=[{ans_str}]"
    return _pass(details)


# ── 5. Duplicate check ─────────────────────────────────────────────
def check_duplicates(json_path: Path, key: str = "questions") -> dict:
    """Check for duplicate (sample_token + question/description) pairs."""
    with open(json_path) as f:
        data = json.load(f)

    items = data[key]
    if key == "questions":
        keys = [(item["sample_token"], item["question"]) for item in items]
    else:
        keys = [(item["sample_token"],) for item in items]

    n_dupes = len(keys) - len(set(keys))
    details = f"{json_path.name}: {n_dupes} duplicates out of {len(keys)}"
    if n_dupes > len(keys) * 0.01:  # >1% duplicates
        return _warn(details)
    return _pass(details)


# ── Main ────────────────────────────────────────────────────────────
def main():
    print(f"\n{BOLD}{'='*70}{RESET}")
    print(f"{BOLD}  BEV-VQA Block 0: Data Sanity Checks{RESET}")
    print(f"{BOLD}{'='*70}{RESET}\n")

    bev_dir = Path("/media/nazario.pizzicoli/Datos/dataset_veh/bev_features_veh")
    unified_dir = Path("/media/nazario.pizzicoli/Datos/vqa_datasets/unified")

    report: dict = {"checks": [], "summary": {}}

    # ── BEV features ──
    print(f"{BOLD}[1/5] BEV Features{RESET}")
    for split in ["train", "val"]:
        r = check_bev_features(bev_dir, split)
        report["checks"].append({"name": f"bev_{split}", **r})

    # ── Dataset JSON ──
    print(f"\n{BOLD}[2/5] Dataset JSON Schema{RESET}")
    datasets = [
        ("nuscenes_qa_train.json", "questions"),
        ("nuscenes_qa_val.json", "questions"),
        ("drivelm_train.json", "questions"),
        ("drivelm_val.json", "questions"),
        ("omnidrive_descriptions_train.json", "descriptions"),
    ]
    for fname, key in datasets:
        r = check_dataset_json(unified_dir / fname, key=key)
        report["checks"].append({"name": f"schema_{fname}", **r})

    # ── BEV coverage ──
    print(f"\n{BOLD}[3/5] BEV Feature Coverage{RESET}")
    for fname, key in datasets:
        r = check_bev_coverage(unified_dir / fname, bev_dir, key=key)
        report["checks"].append({"name": f"coverage_{fname}", **r})

    # ── Class distribution ──
    print(f"\n{BOLD}[4/5] Class Distribution{RESET}")
    for fname, key in datasets:
        if key == "questions":
            r = check_class_distribution(unified_dir / fname)
            report["checks"].append({"name": f"dist_{fname}", **r})

    # ── Duplicates ──
    print(f"\n{BOLD}[5/5] Duplicate Check{RESET}")
    for fname, key in datasets:
        r = check_duplicates(unified_dir / fname, key=key)
        report["checks"].append({"name": f"dupes_{fname}", **r})

    # ── Summary ──
    n_pass = sum(1 for c in report["checks"] if c["status"] == "PASS")
    n_fail = sum(1 for c in report["checks"] if c["status"] == "FAIL")
    n_warn = sum(1 for c in report["checks"] if c["status"] == "WARN")
    total = len(report["checks"])

    report["summary"] = {
        "total": total, "pass": n_pass, "fail": n_fail, "warn": n_warn,
        "overall": "PASS" if n_fail == 0 else "FAIL",
    }

    print(f"\n{BOLD}{'='*70}{RESET}")
    color = GREEN if n_fail == 0 else RED
    print(f"  {color}{BOLD}OVERALL: {report['summary']['overall']}{RESET}")
    print(f"  {n_pass}/{total} PASS | {n_warn} WARN | {n_fail} FAIL")
    print(f"{BOLD}{'='*70}{RESET}\n")

    # Save report
    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "block0_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report saved to {report_path}\n")


if __name__ == "__main__":
    main()

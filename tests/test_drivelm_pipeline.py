"""
Comprehensive verification test suite for DriveLM pipeline fixes and enhancements.

Verifies:
a) BEVQADataset loads drivelm_val.json without len == 0.
b) Sample outputs have non- -100 labels (full answer token supervision, no truncation).
c) compute_nlg_metrics computes real BLEU-1..4, ROUGE-L, CIDEr, METEOR scores.
d) DriveLMCategoryBalancedSampler adheres to target category ratios.
e) scripts/train_sota_drivelm_v2.py --dry-run executes with exit code 0.
"""

from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

import torch

from bev_vqa.data.dataset import BEVQADataset, build_tokenizer
from bev_vqa.data.sampler import DriveLMCategoryBalancedSampler, DRIVELM_CATEGORY_RATIOS
from bev_vqa.training.metrics import compute_nlg_metrics


VAL_JSON_PATH = "/media/nazario.pizzicoli/Datos/vqa_datasets/unified/drivelm_val.json"
BEV_DIR_PATH = "/media/nazario.pizzicoli/Datos/dataset/bev_features"
LLM_PATH = "Qwen/Qwen2.5-3B-Instruct"


class TestDriveLMPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.has_dataset = Path(VAL_JSON_PATH).exists() and Path(BEV_DIR_PATH).exists()
        if cls.has_dataset:
            cls.tokenizer = build_tokenizer(LLM_PATH)

    def test_a_drivelm_val_dataset_load_non_empty(self):
        """a) BEVQADataset loads drivelm_val.json without len == 0."""
        if not self.has_dataset:
            self.skipTest(f"Dataset path {VAL_JSON_PATH} not found.")

        ds = BEVQADataset(
            json_path=VAL_JSON_PATH,
            bev_dir=BEV_DIR_PATH,
            tokenizer=self.tokenizer,
            split="val",
            max_length=384,
        )
        self.assertGreater(len(ds), 0, "Dataset length should not be 0!")
        self.assertEqual(len(ds), 36801, f"Expected 36801 items in drivelm_val, got {len(ds)}")

    def test_b_sample_labels_supervised_non_minus_100(self):
        """b) Sample outputs have non- -100 labels (full answer token supervision)."""
        if not self.has_dataset:
            self.skipTest(f"Dataset path {VAL_JSON_PATH} not found.")

        ds = BEVQADataset(
            json_path=VAL_JSON_PATH,
            bev_dir=BEV_DIR_PATH,
            tokenizer=self.tokenizer,
            split="val",
            max_length=384,
        )
        sample = ds[0]

        self.assertIn("input_ids", sample)
        self.assertIn("labels", sample)
        self.assertIn("bev", sample)

        labels = sample["labels"]
        non_masked_count = (labels != -100).sum().item()
        self.assertGreater(
            non_masked_count, 0,
            "Answer tokens in labels must not all be masked to -100 (truncation bug eliminated)."
        )

        # Verify BEV feature map is loaded (non-zero tensor)
        self.assertIsNotNone(sample["bev"])
        self.assertEqual(sample["bev"].shape, torch.Size([128, 200, 200]))

    def test_c_compute_nlg_metrics_real_scores(self):
        """c) compute_nlg_metrics computes real BLEU/CIDEr scores with pycocoevalcap & nltk."""
        predictions = [
            "the car should stop at the red light and wait",
            "pedestrian is crossing the street from left to right"
        ]
        references = [
            ["the car should stop at the red traffic light and wait", "vehicle stops at red light"],
            ["a pedestrian crosses the street from left to right"]
        ]

        metrics = compute_nlg_metrics(predictions=predictions, references=references)

        required_keys = ["bleu_1", "bleu_4", "rouge_l", "cider", "meteor", "BLEU-4", "CIDEr"]
        for k in required_keys:
            self.assertIn(k, metrics, f"Missing key '{k}' in NLG metrics output.")
            self.assertIsInstance(metrics[k], float)

        self.assertGreater(metrics["bleu_1"], 0.0, "BLEU-1 should be positive.")
        self.assertGreater(metrics["bleu_4"], 0.0, "BLEU-4 should be positive.")
        self.assertGreater(metrics["rouge_l"], 0.0, "ROUGE-L should be positive.")
        self.assertGreater(metrics["cider"], 0.0, "CIDEr should be positive.")
        self.assertGreater(metrics["meteor"], 0.0, "METEOR should be positive.")

        # Test graceful empty handling
        empty_metrics = compute_nlg_metrics([], [])
        self.assertEqual(empty_metrics["bleu_4"], 0.0)
        self.assertEqual(empty_metrics["cider"], 0.0)

    def test_d_drivelm_category_balanced_sampler(self):
        """d) DriveLMCategoryBalancedSampler adheres to target DriveLM proportions."""
        mock_data = (
            [{"template_type": "perception"}] * 1000 +
            [{"template_type": "prediction"}] * 100 +
            [{"template_type": "planning"}] * 50 +
            [{"template_type": "behavior"}] * 10
        )

        sampler = DriveLMCategoryBalancedSampler(mock_data, num_samples=10000, seed=42)
        indices = list(sampler)
        self.assertEqual(len(indices), 10000)

        counts = Counter(mock_data[i]["template_type"] for i in indices)
        total = len(indices)
        ratios = {k: v / total for k, v in counts.items()}

        self.assertAlmostEqual(ratios["perception"], 0.35, delta=0.03)
        self.assertAlmostEqual(ratios["prediction"], 0.30, delta=0.03)
        self.assertAlmostEqual(ratios["planning"], 0.30, delta=0.03)
        self.assertAlmostEqual(ratios["behavior"], 0.05, delta=0.02)

    def test_f_compute_nlg_metrics_empty_strings(self):
        """f) compute_nlg_metrics returns 0.0 for empty predictions / references without score inflation."""
        empty_scores = compute_nlg_metrics(predictions=["", "  "], references=["", ""])
        self.assertEqual(empty_scores["bleu_1"], 0.0, "Empty prediction against empty ref should yield 0.0 BLEU-1")
        self.assertEqual(empty_scores["rouge_l"], 0.0, "Empty prediction against empty ref should yield 0.0 ROUGE-L")
        self.assertEqual(empty_scores["meteor"], 0.0, "Empty prediction against empty ref should yield 0.0 METEOR")

        empty_pred_scores = compute_nlg_metrics(predictions=["", ""], references=["stop at red light", "turn left"])
        self.assertEqual(empty_pred_scores["bleu_1"], 0.0)
        self.assertEqual(empty_pred_scores["rouge_l"], 0.0)

    def test_g_sampler_dataset_wrapper_support(self):
        """g) DriveLMCategoryBalancedSampler works with custom objects having .data attribute."""
        class MockDatasetWrapper:
            def __init__(self, items):
                self.data = items

        mock_items = (
            [{"template_type": "perception"}] * 10 +
            [{"template_type": "prediction"}] * 10
        )
        wrapper = MockDatasetWrapper(mock_items)
        sampler = DriveLMCategoryBalancedSampler(wrapper, num_samples=20, seed=42)
        sampled = list(sampler)
        self.assertEqual(len(sampled), 20)

    def test_h_load_bev_fallback_shape_3d(self):
        """h) BEVQADataset._load_bev fallback consistently returns 3D tensor (128, 200, 200)."""
        if not self.has_dataset:
            self.skipTest(f"Dataset path {VAL_JSON_PATH} not found.")

        ds = BEVQADataset(
            json_path=VAL_JSON_PATH,
            bev_dir=BEV_DIR_PATH,
            tokenizer=self.tokenizer,
            split="val",
            max_length=384,
        )
        # Non-existent token fallback
        fallback_feat = ds._load_bev("non_existent_token_12345")
        self.assertIsNotNone(fallback_feat)
        self.assertEqual(fallback_feat.shape, torch.Size([128, 200, 200]))

    def test_e_dry_run_training_script_exit_zero(self):
        """e) Dry-run forward pass of scripts/train_sota_drivelm_v2.py --dry-run executes with exit code 0."""
        if not self.has_dataset:
            self.skipTest(f"Dataset path {VAL_JSON_PATH} not found.")

        test_output_dir = Path("checkpoints/test_pipeline_unit_dry_run")
        test_report = Path("outputs/test_pipeline_unit_dry_run.json")

        cmd = [
            sys.executable,
            "scripts/train_sota_drivelm_v2.py",
            "--dry-run",
            "--output-dir", str(test_output_dir),
            "--report-path", str(test_report),
        ]

        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            self.assertEqual(
                res.returncode, 0,
                f"Dry run failed with exit code {res.returncode}.\nSTDOUT:\n{res.stdout[-1000:]}\nSTDERR:\n{res.stderr[-1000:]}"
            )
            self.assertTrue(test_report.exists(), "Report JSON should exist after dry-run.")
            with open(test_report) as f:
                report_data = json.load(f)
            self.assertEqual(report_data["total_steps"], 2)
            self.assertIn("best_cider", report_data)
            self.assertIn("best_bleu_4", report_data)
        finally:
            if test_output_dir.exists():
                shutil.rmtree(test_output_dir, ignore_errors=True)
            if test_report.exists():
                test_report.unlink(missing_ok=True)


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDriveLMPipeline)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)


if __name__ == "__main__":
    run_tests()

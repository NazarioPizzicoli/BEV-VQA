"""
Automated Test Suite for BEV-VLM Interactive Inference & Visual Debugging Pipeline.

Verifies:
1. Camera path lookup and image loading from nuScenes cache.
2. BEV heatmap generation (L2 norm) and ANSI terminal truecolor rendering.
3. Multimodal debug figure creation, styling, and high-res PNG output.
4. End-to-end forward generation in headless mode (both mock and real model).
"""

from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

# Ensure src/ is on sys.path
TEST_DIR = Path(__file__).resolve().parent
REPO_ROOT = TEST_DIR.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bev_vqa.utils.visualization import (
    build_multimodal_debug_figure,
    compute_bev_l2_heatmap,
    render_ansi_terminal_image,
    render_ansi_bev_grid,
    render_ansi_camera_mosaic,
    SURROUND_CAMERAS,
)
from scripts.inference_interactive import (
    BEVInferencePipeline,
    run_headless_inference,
    DEFAULT_CHECKPOINT,
    DEFAULT_CAM_MAP_PATH,
    DEFAULT_DRIVELM_JSON,
    DEFAULT_NUSCENES_QA_JSON,
)


class TestCameraLookupAndLoading(unittest.TestCase):
    """Tests camera path indexing and image loading."""

    def test_camera_map_file_and_keys(self):
        """nuScenes camera map cache exists and has 6 standard camera channels."""
        cam_map_path = Path(DEFAULT_CAM_MAP_PATH)
        if not cam_map_path.exists():
            self.skipTest(f"Camera cache not found at {cam_map_path}")

        with open(cam_map_path, "r") as f:
            cam_map = json.load(f)

        self.assertGreater(len(cam_map), 0, "Camera map should not be empty")
        first_token = next(iter(cam_map))
        cams = cam_map[first_token]
        for c in SURROUND_CAMERAS:
            self.assertIn(c, cams, f"Expected camera '{c}' in nuScenes cache")

        # Verify that image path exists on disk
        sample_img = cams["CAM_FRONT"]
        self.assertTrue(Path(sample_img).exists(), f"Camera JPEG file should exist: {sample_img}")

    def test_image_loading_and_fallback(self):
        """PIL properly loads camera JPEG and handles missing image paths gracefully."""
        # 1. Create a temporary dummy image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            img = Image.new("RGB", (320, 180), color=(100, 150, 200))
            img.save(tmp_path)

            loaded = Image.open(tmp_path).convert("RGB")
            self.assertEqual(loaded.size, (320, 180))
            self.assertEqual(loaded.mode, "RGB")
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        # 2. Test missing path handled in render_ansi_terminal_image
        missing_result = render_ansi_terminal_image("non_existent_image.jpg", width=20)
        self.assertIn("not found", missing_result.lower())


class TestBEVHeatmapAndANSIRendering(unittest.TestCase):
    """Tests BEV L2 norm calculation and ANSI truecolor renderers."""

    def test_compute_bev_l2_heatmap_shapes(self):
        """compute_bev_l2_heatmap handles 4D, 3D, and 2D tensors consistently."""
        # 4D tensor [1, 128, 200, 200]
        t4 = torch.ones(1, 128, 200, 200)
        hm4 = compute_bev_l2_heatmap(t4)
        self.assertEqual(hm4.shape, (200, 200))
        expected_norm = np.sqrt(128.0)
        np.testing.assert_allclose(hm4[0, 0], expected_norm, rtol=1e-4)

        # 3D tensor [128, 200, 200]
        t3 = torch.randn(128, 200, 200)
        hm3 = compute_bev_l2_heatmap(t3)
        self.assertEqual(hm3.shape, (200, 200))
        self.assertTrue(np.all(hm3 >= 0.0), "L2 norm must be non-negative")

        # Numpy array input
        np_arr = np.ones((128, 50, 50), dtype=np.float32)
        hm_np = compute_bev_l2_heatmap(np_arr)
        self.assertEqual(hm_np.shape, (50, 50))

    def test_compute_bev_l2_heatmap_batched_multi(self):
        """compute_bev_l2_heatmap handles batched 4D tensors and numpy arrays where batch size B > 1."""
        t4_batched = torch.ones(3, 128, 60, 60)
        hm_batched = compute_bev_l2_heatmap(t4_batched)
        self.assertEqual(hm_batched.shape, (60, 60))
        expected_norm = np.sqrt(128.0)
        np.testing.assert_allclose(hm_batched[0, 0], expected_norm, rtol=1e-4)

        np_batched = np.ones((4, 128, 40, 40), dtype=np.float32)
        hm_np_batched = compute_bev_l2_heatmap(np_batched)
        self.assertEqual(hm_np_batched.shape, (40, 40))

    def test_render_ansi_bev_grid_inferno(self):
        """render_ansi_bev_grid supports inferno colormap and renders valid ANSI output."""
        bev = torch.randn(128, 100, 100)
        ansi_out = render_ansi_bev_grid(bev, width=32, colormap="inferno")
        self.assertIsInstance(ansi_out, str)
        self.assertIn("\u2580", ansi_out)
        self.assertIn("\033[38;2;", ansi_out)

    def test_render_ansi_terminal_image_half_block(self):
        """render_ansi_terminal_image outputs valid ANSI truecolor escape codes with half-blocks."""
        # Create a small RGB test array (4x4)
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        rgb[:2, :, 0] = 255  # Red on top
        rgb[2:, :, 2] = 255  # Blue on bottom

        ansi_out = render_ansi_terminal_image(rgb, width=4)
        self.assertIsInstance(ansi_out, str)
        self.assertIn("\u2580", ansi_out, "Should contain half-block character '▀'")
        self.assertIn("\033[38;2;", ansi_out, "Should contain 24-bit ANSI foreground code")
        self.assertIn("\033[48;2;", ansi_out, "Should contain 24-bit ANSI background code")
        self.assertIn("\033[0m", ansi_out, "Should contain ANSI reset code")

    def test_render_ansi_bev_grid(self):
        """render_ansi_bev_grid creates non-empty ANSI string with correct width."""
        bev = torch.randn(128, 200, 200)
        grid_ansi = render_ansi_bev_grid(bev, width=30)
        self.assertIsInstance(grid_ansi, str)
        self.assertGreater(len(grid_ansi), 100)
        lines = grid_ansi.strip().split("\n")
        # Each line should contain 30 half-blocks
        first_line_blocks = lines[0].count("\u2580")
        self.assertEqual(first_line_blocks, 30, f"Expected 30 characters per line, got {first_line_blocks}")

    def test_render_ansi_camera_mosaic(self):
        """render_ansi_camera_mosaic builds composite preview without crashing."""
        cam_paths = {c: f"dummy_{c}.jpg" for c in SURROUND_CAMERAS}
        mosaic_ansi = render_ansi_camera_mosaic(cam_paths, width=40)
        self.assertIsInstance(mosaic_ansi, str)
        self.assertIn("\u2580", mosaic_ansi)


class TestFigureCreation(unittest.TestCase):
    """Tests build_multimodal_debug_figure creation, layout, and output files."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="test_fig_")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_build_multimodal_debug_figure_creates_valid_png(self):
        """build_multimodal_debug_figure saves a valid high-res PNG with all components."""
        dest_png = Path(self.temp_dir) / "dashboard_test.png"
        dummy_bev = torch.randn(128, 200, 200)

        out_path = build_multimodal_debug_figure(
            sample_token="test_token_abc123",
            cam_paths={},
            bev_tensor=dummy_bev,
            question="What is the ego vehicle's planned trajectory?",
            prediction="The ego vehicle should continue straight at 30 km/h.",
            ground_truth="continue straight",
            category="DriveLM - Planning",
            output_path=dest_png,
        )

        self.assertTrue(Path(out_path).exists(), "Output PNG file should exist")
        self.assertEqual(Path(out_path).resolve(), dest_png.resolve())

        # Verify PNG is valid and has high resolution
        img = Image.open(out_path)
        self.assertGreater(img.size[0], 1800, "Figure width should be high-res (>1800px)")
        self.assertGreater(img.size[1], 1200, "Figure height should be high-res (>1200px)")

    def test_build_multimodal_debug_figure_default_path(self):
        """build_multimodal_debug_figure generates timestamped file in outputs/inference/ preserving full sample token."""
        dummy_bev = torch.randn(128, 200, 200)
        sample_token_test = "autoname_full_sample_token_123456789"
        out_path = build_multimodal_debug_figure(
            sample_token=sample_token_test,
            cam_paths={},
            bev_tensor=dummy_bev,
            question="Is the path clear?",
            prediction="Yes, path is clear.",
            ground_truth=None,
            category=None,
            output_path=None,
        )

        try:
            self.assertTrue(Path(out_path).exists())
            self.assertIn("outputs/inference", str(out_path))
            self.assertIn(sample_token_test, str(out_path), "Full sample token must be preserved in filename")
        finally:
            if Path(out_path).exists():
                Path(out_path).unlink()

    def test_build_multimodal_debug_figure_long_text_wrapping(self):
        """build_multimodal_debug_figure handles long questions and multi-sentence predictions without error."""
        long_q = "What are the important objects in the current scene, and what specific driving actions should the ego vehicle take in the next 3 to 5 seconds considering all surrounding road users?"
        long_pred = "The ego vehicle should observe the pedestrian on the right sidewalk walking towards the crosswalk. Therefore, the ego vehicle should decelerate smoothly, maintain lane centering, and yield right of way."
        long_gt = "Decelerate smoothly and yield to the pedestrian."

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            out_path = build_multimodal_debug_figure(
                sample_token="long_text_token",
                cam_paths={},
                bev_tensor=torch.randn(128, 200, 200),
                question=long_q,
                prediction=long_pred,
                ground_truth=long_gt,
                category="DriveLM - Complex Planning",
                colormap="inferno",
                output_path=tmp_path,
            )
            self.assertTrue(Path(out_path).exists())
            self.assertGreater(Path(out_path).stat().st_size, 50000)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


class TestInferencePipelineExecution(unittest.TestCase):
    """Tests BEVInferencePipeline end-to-end execution and CLI headless mode."""

    def test_pipeline_indexing_and_mock_generation(self):
        """Pipeline indexes questions and produces deterministic mock predictions."""
        pipeline = BEVInferencePipeline(mock=True)

        self.assertGreater(len(pipeline.available_bevs), 0, "Pipeline should find BEV files")
        sample_token = next(iter(pipeline.available_bevs.keys()))

        # Answer question
        res = pipeline.answer_question(
            sample_token=sample_token,
            question="What action should the vehicle take?",
            category="DriveLM - Planning",
        )

        self.assertTrue(res["success"])
        self.assertIn("action", res["question"])
        self.assertIsInstance(res["raw_answer"], str)
        self.assertGreater(len(res["raw_answer"]), 5)
        self.assertGreater(res["latency_ms"], 0.0)

    def test_run_headless_inference_function(self):
        """run_headless_inference executes cleanly with output PNG generation."""
        pipeline = BEVInferencePipeline(mock=True)
        sample_token = next(iter(pipeline.available_bevs.keys()))

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_png = tmp.name

        try:
            res = run_headless_inference(
                pipeline=pipeline,
                token=sample_token,
                question="Are there any cars ahead?",
                output_image=tmp_png,
                no_viz=False,
                terminal_viz=False,
            )

            self.assertTrue(res["success"])
            self.assertTrue(Path(tmp_png).exists())
            self.assertGreater(Path(tmp_png).stat().st_size, 50000)
        finally:
            if os.path.exists(tmp_png):
                os.unlink(tmp_png)

    def test_cli_headless_subprocess_exit_zero(self):
        """CLI execution `python scripts/inference_interactive.py --mock --random --no-viz` returns code 0."""
        cmd = [
            sys.executable,
            "scripts/inference_interactive.py",
            "--mock",
            "--random",
            "--no-viz",
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(
            proc.returncode,
            0,
            f"CLI failed with returncode {proc.returncode}.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
        self.assertIn("BEV-VLM INFERENCE RESULT", proc.stdout)

    def test_cli_headless_with_viz_and_terminal_preview(self):
        """CLI execution with --terminal-viz and --output-image produces visual output and exit code 0."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            test_img = tmp.name

        try:
            cmd = [
                sys.executable,
                "scripts/inference_interactive.py",
                "--mock",
                "--random",
                "--question", "Are there vehicles nearby?",
                "--terminal-viz",
                "--output-image", test_img,
            ]
            proc = subprocess.run(
                cmd,
                cwd=str(REPO_ROOT),
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(proc.returncode, 0, f"Error: {proc.stderr}")
            self.assertTrue(Path(test_img).exists())
            self.assertGreater(Path(test_img).stat().st_size, 50000)
            self.assertIn("TERMINAL PREVIEW", proc.stdout)
        finally:
            if os.path.exists(test_img):
                os.unlink(test_img)

    def test_cli_headless_random_question_idx_selection(self):
        """CLI execution `--mock --random --question-idx 1` consistently resolves question #1 without fallback."""
        cmd = [
            sys.executable,
            "scripts/inference_interactive.py",
            "--mock",
            "--random",
            "--question-idx", "1",
            "--no-viz",
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(proc.returncode, 0, f"Error: {proc.stderr}")
        self.assertIn("BEV-VLM INFERENCE RESULT", proc.stdout)
        # Should not fall back to generic scene description prompt
        self.assertNotIn("Describe the autonomous driving scene around the ego vehicle.", proc.stdout)
        self.assertIn("Question:", proc.stdout)

    def test_cli_headless_question_idx_out_of_bounds(self):
        """CLI handles out-of-bounds question-idx (e.g. 9999) gracefully with fallback and exit code 0."""
        cmd = [
            sys.executable,
            "scripts/inference_interactive.py",
            "--mock",
            "--random",
            "--question-idx", "9999",
            "--no-viz",
        ]
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(proc.returncode, 0, f"Error: {proc.stderr}")
        self.assertIn("BEV-VLM INFERENCE RESULT", proc.stdout)


def run_tests():
    suite = unittest.TestLoader().loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)


if __name__ == "__main__":
    run_tests()

"""
Unit tests for Compartment 2: Spatial & Metric Positional Encoding.
"""

import math
import torch
import pytest

from bev_vqa.models.projector import (
    ProjectorConfig,
    build_projector,
    build_fourier_pe_1d,
    MultiScalePeakProjector,
)


def test_build_fourier_pe_1d():
    coords = torch.tensor([-1.0, -0.5, 0.0, 0.5, 1.0])
    dim = 128
    pe = build_fourier_pe_1d(coords, dim)
    assert pe.shape == (5, 128)
    assert torch.isfinite(pe).all()
    # Check that sin and cos values are bounded in [-1, 1]
    assert (pe >= -1.0).all() and (pe <= 1.0).all()


@pytest.mark.parametrize("pe_type", [
    "1d_learned",
    "2d_cartesian_metric",
    "2d_polar_metric",
    "2d_separable_learned",
])
def test_pos_embed_forward_shapes(pe_type):
    cfg = ProjectorConfig(
        arch_type="multiscale_peak",
        spatial_shape=(8, 4),
        num_tokens=32,
        pos_embed_type=pe_type,
        hidden_channels=256,
        projector_output_size=2048,
    )
    proj = build_projector(cfg)
    proj.eval()

    bev = torch.randn(2, 128, 100, 100)
    out = proj(bev)
    assert out.shape == (2, 32, 2048)
    assert torch.isfinite(out).all()

    pe = proj.get_pos_embed()
    assert pe.shape == (1, 32, 256)
    assert torch.isfinite(pe).all()


def test_cartesian_metric_coordinates():
    cfg = ProjectorConfig(
        arch_type="multiscale_peak",
        spatial_shape=(8, 4),
        num_tokens=32,
        pos_embed_type="2d_cartesian_metric",
        hidden_channels=256,
    )
    proj = build_projector(cfg)
    assert hasattr(proj, "cartesian_pe")
    assert proj.cartesian_pe.shape == (1, 32, 256)


def test_polar_metric_coordinates():
    cfg = ProjectorConfig(
        arch_type="multiscale_peak",
        spatial_shape=(8, 4),
        num_tokens=32,
        pos_embed_type="2d_polar_metric",
        hidden_channels=256,
    )
    proj = build_projector(cfg)
    assert hasattr(proj, "polar_pe")
    assert proj.polar_pe.shape == (1, 32, 256)


def test_separable_grid_property():
    cfg = ProjectorConfig(
        arch_type="multiscale_peak",
        spatial_shape=(8, 4),
        num_tokens=32,
        pos_embed_type="2d_separable_learned",
        hidden_channels=256,
    )
    proj = build_projector(cfg)
    assert hasattr(proj, "row_embed")
    assert hasattr(proj, "col_embed")
    assert proj.row_embed.shape == (1, 8, 1, 128)
    assert proj.col_embed.shape == (1, 1, 4, 128)

"""Unit tests for the 4 unified projector architectures."""

import pytest
import torch
from bev_vqa.models.projector import ProjectorConfig, build_projector


@pytest.mark.parametrize("arch_type,expected_tokens", [
    ("dual_pooling", 32),
    ("multiscale_peak", 32),
    ("hybrid_sparse", 40),
    ("question_aware", 32),
])
def test_projector_output_shape(arch_type, expected_tokens):
    cfg = ProjectorConfig(
        in_channels=128,
        num_tokens=32,
        projector_output_size=2048,
        arch_type=arch_type,
        spatial_shape=(8, 4),
        hidden_channels=256,
        num_sparse_tokens=8,
    )
    proj = build_projector(cfg)
    bev = torch.randn(2, 128, 200, 200)
    
    if getattr(proj, "requires_text", False):
        text_embeds = torch.randn(2, 15, 2048)
        out = proj(bev, text_embeds=text_embeds)
    else:
        out = proj(bev)
        
    assert out.shape == (2, expected_tokens, 2048), f"Failed for {arch_type}: got {out.shape}"


@pytest.mark.parametrize("arch_type", [
    "dual_pooling",
    "multiscale_peak",
    "hybrid_sparse",
    "question_aware",
])
def test_projector_gradient_flow(arch_type):
    cfg = ProjectorConfig(
        in_channels=128,
        num_tokens=32,
        projector_output_size=2048,
        arch_type=arch_type,
        spatial_shape=(8, 4),
        hidden_channels=256,
        num_sparse_tokens=8,
    )
    proj = build_projector(cfg)
    proj.train()
    bev = torch.randn(2, 128, 200, 200, requires_grad=True)
    
    if getattr(proj, "requires_text", False):
        text_embeds = torch.randn(2, 15, 2048, requires_grad=True)
        out = proj(bev, text_embeds=text_embeds)
    else:
        out = proj(bev)
        
    loss = out.sum()
    loss.backward()
    
    assert bev.grad is not None
    assert not torch.isnan(bev.grad).any()
    assert torch.norm(bev.grad).item() > 0


def test_stage1_checkpoint_loading():
    checkpoint_path = "checkpoints/sota/stage1/stage1_projector_best.pt"
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    
    for arch_type in ["dual_pooling", "multiscale_peak", "hybrid_sparse", "question_aware"]:
        cfg = ProjectorConfig(
            in_channels=128,
            num_tokens=32,
            projector_output_size=2048,
            arch_type=arch_type,
            spatial_shape=(8, 4),
            hidden_channels=256,
            num_sparse_tokens=8,
        )
        proj = build_projector(cfg)
        load_res = proj.load_state_dict(state_dict, strict=False)
        
        # Verify that all conv_stack keys loaded properly (pool_shape buffer or legacy pos_embed may be unexpected)
        assert len(load_res.unexpected_keys) == 0 or all("pool_shape" in k or "pos_embed" in k for k in load_res.unexpected_keys)
        # Check that conv_stack weights match
        assert torch.equal(proj.conv_stack[0].weight, state_dict["conv_stack.0.weight"])

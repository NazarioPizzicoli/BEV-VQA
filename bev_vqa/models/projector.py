"""
Projector modules for BEV-VQA.

Validated Production Architecture:
- MultiScalePeakProjector ('multiscale_peak') with 2D Polar Metric Positional Encoding ('2d_polar_metric').
  Grid resolution: 8x4 (32 visual tokens).
  Integrates high-resolution lateral skip connections (FPN-style) with metric range-azimuth coordinates.

Supported architectures for ablation and comparative evaluation:
1. 'multiscale_peak' (Winning Architecture): Multi-Scale Peak Injection via High-Res lateral skip connection
2. 'dual_pooling': DeeperConv with parallel AvgPool + MaxPool (CBAM-style)
3. 'hybrid_sparse': 32 Grid Tokens (8x4) + 8 Top-K Salient Object Tokens = 40 Tokens (SparseDrive-style)
4. 'question_aware': Language-Guided Cross-Attention modulating BEV spatial features (InstructBLIP-style)
5. 'deeper_conv': Legacy baseline convolutional projector
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ProjectorConfig:
    """Configuration for the visual projector.
    
    Attributes:
        in_channels (int): Number of input channels (default: 128).
        num_tokens (int): Number of output tokens (default: 32).
        projector_output_size (int): Output dimension matching LLM hidden size (e.g., 2048 for Qwen2.5-3B).
        arch_type (str): Architecture type: 'multiscale_peak' (validated winning default),
            'dual_pooling', 'hybrid_sparse', 'question_aware', 'deeper_conv'.
        pos_embed_type (str): Positional encoding type: '2d_polar_metric' (validated winning default),
            '1d_learned', '2d_cartesian_metric', '2d_separable_learned'.
        hidden_channels (int): Hidden channels for Conv backbone (default: 256).
        num_conv_layers (int): Number of convolutional layers (default: 4).
        use_dual_pooling (bool): Parallel Avg + Max pooling projected back to C.
        spatial_shape (Optional[Tuple[int, int]]): Target spatial shape (default: (8, 4)).
        use_l2_pooling (bool): Salience-weighted pooling based on L2 norm of features.
        l2_pooling_gamma (float): Salience exponent/temperature for L2 pooling.
        l2_pooling_learnable (bool): If True, makes gamma a learnable parameter.
        num_sparse_tokens (int): Number of sparse object tokens for hybrid_sparse (default: 8).
    """
    in_channels: int = 128
    num_tokens: int = 32
    projector_output_size: int = 2048
    arch_type: str = "multiscale_peak"
    pos_embed_type: str = "2d_polar_metric"
    hidden_channels: int = 256
    num_conv_layers: int = 4
    use_dual_pooling: bool = False
    spatial_shape: Optional[Tuple[int, int]] = None
    use_l2_pooling: bool = False
    l2_pooling_gamma: float = 1.0
    l2_pooling_learnable: bool = False
    num_sparse_tokens: int = 8


def build_fourier_pe_1d(coords: torch.Tensor, dim: int) -> torch.Tensor:
    """Computes Fourier sinusoidal positional encodings for coordinates in [-1, 1].
    
    Args:
        coords: Tensor of shape [N] with values normalized in [-1, 1].
        dim: Target channel dimension (must be even).
        
    Returns:
        Tensor of shape [N, dim].
    """
    num_bands = dim // 2
    freqs = torch.exp(torch.linspace(0.0, math.log(100.0), num_bands, device=coords.device))
    angles = coords.unsqueeze(-1) * freqs.unsqueeze(0) * math.pi
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def get_spatial_pool_shape(num_tokens: int) -> Tuple[int, int]:
    """Determines the optimal 2D (h, w) shape to obtain exactly num_tokens."""
    root = int(round(math.sqrt(num_tokens)))
    if root * root == num_tokens:
        return (root, root)
    for h in range(root, 0, -1):
        if num_tokens % h == 0:
            return (h, num_tokens // h)
    return (1, num_tokens)


def _build_conv_stack(in_channels: int, hidden_channels: int, num_layers: int) -> nn.Sequential:
    """Builds standard 4-layer strided convolutional backbone."""
    channels = [in_channels] + [hidden_channels] * num_layers
    layers = []
    for i in range(num_layers):
        layers.extend([
            nn.Conv2d(channels[i], channels[i + 1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[i + 1]),
            nn.GELU()
        ])
    return nn.Sequential(*layers)


class BaseProjector(nn.Module):
    """Base class providing shared MLP projection and positional embedding."""
    def __init__(self, config: ProjectorConfig) -> None:
        super().__init__()
        self.config = config
        self.requires_text = False

        if config.spatial_shape is not None:
            self.pool_shape = tuple(config.spatial_shape)
            self.num_tokens = self.pool_shape[0] * self.pool_shape[1]
            self.config.num_tokens = self.num_tokens
        else:
            self.pool_shape = get_spatial_pool_shape(config.num_tokens)
            self.num_tokens = config.num_tokens

        self.register_buffer(
            "pool_shape_tensor",
            torch.tensor(self.pool_shape, dtype=torch.long),
            persistent=True,
        )

        self.conv_stack = _build_conv_stack(
            config.in_channels, config.hidden_channels, config.num_conv_layers
        )

        self.pos_embed_type = getattr(config, "pos_embed_type", "1d_learned").lower()

        if self.pos_embed_type == "1d_learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_tokens, config.hidden_channels))
            nn.init.normal_(self.pos_embed, std=0.02)
        elif self.pos_embed_type == "2d_cartesian_metric":
            H, W = self.pool_shape
            y_m = torch.linspace(-50.0 + 100.0 / (2 * H), 50.0 - 100.0 / (2 * H), H)
            x_m = torch.linspace(-50.0 + 100.0 / (2 * W), 50.0 - 100.0 / (2 * W), W)
            grid_y, grid_x = torch.meshgrid(y_m, x_m, indexing="ij")

            norm_x = (grid_x / 50.0).flatten()
            norm_y = (grid_y / 50.0).flatten()

            pe_x = build_fourier_pe_1d(norm_x, config.hidden_channels // 2)
            pe_y = build_fourier_pe_1d(norm_y, config.hidden_channels // 2)
            cartesian_grid = torch.cat([pe_x, pe_y], dim=-1).unsqueeze(0)  # [1, num_tokens, C]
            self.register_buffer("cartesian_pe", cartesian_grid, persistent=True)

            self.pe_adapt = nn.Sequential(
                nn.Linear(config.hidden_channels, config.hidden_channels),
                nn.GELU(),
                nn.Linear(config.hidden_channels, config.hidden_channels),
            )
            nn.init.zeros_(self.pe_adapt[-1].weight)
            nn.init.zeros_(self.pe_adapt[-1].bias)
        elif self.pos_embed_type == "2d_polar_metric":
            H, W = self.pool_shape
            y_m = torch.linspace(-50.0 + 100.0 / (2 * H), 50.0 - 100.0 / (2 * H), H)
            x_m = torch.linspace(-50.0 + 100.0 / (2 * W), 50.0 - 100.0 / (2 * W), W)
            grid_y, grid_x = torch.meshgrid(y_m, x_m, indexing="ij")

            r = torch.sqrt(grid_x**2 + grid_y**2).flatten()
            theta = torch.atan2(grid_y, grid_x).flatten()

            norm_r = r / (50.0 * math.sqrt(2.0))
            norm_theta = theta / math.pi

            pe_r = build_fourier_pe_1d(norm_r, config.hidden_channels // 2)
            pe_theta = build_fourier_pe_1d(norm_theta, config.hidden_channels // 2)
            polar_grid = torch.cat([pe_r, pe_theta], dim=-1).unsqueeze(0)  # [1, num_tokens, C]
            self.register_buffer("polar_pe", polar_grid, persistent=True)

            self.pe_adapt = nn.Sequential(
                nn.Linear(config.hidden_channels, config.hidden_channels),
                nn.GELU(),
                nn.Linear(config.hidden_channels, config.hidden_channels),
            )
            nn.init.zeros_(self.pe_adapt[-1].weight)
            nn.init.zeros_(self.pe_adapt[-1].bias)
        elif self.pos_embed_type == "2d_separable_learned":
            H, W = self.pool_shape
            self.row_embed = nn.Parameter(torch.zeros(1, H, 1, config.hidden_channels // 2))
            self.col_embed = nn.Parameter(torch.zeros(1, 1, W, config.hidden_channels // 2))
            nn.init.normal_(self.row_embed, std=0.02)
            nn.init.normal_(self.col_embed, std=0.02)
        else:
            raise ValueError(
                f"Unknown pos_embed_type: {self.pos_embed_type}. "
                f"Choose from ['1d_learned', '2d_cartesian_metric', '2d_polar_metric', '2d_separable_learned']"
            )

        self.mlp = nn.Sequential(
            nn.Linear(config.hidden_channels, config.projector_output_size),
            nn.GELU(),
            nn.Linear(config.projector_output_size, config.projector_output_size)
        )
        self.out_norm = nn.LayerNorm(config.projector_output_size)

    def get_pos_embed(self, num_tokens: Optional[int] = None) -> torch.Tensor:
        """Retrieves positional embedding of shape [1, num_tokens, hidden_channels]."""
        if self.pos_embed_type == "1d_learned":
            pe = self.pos_embed
        elif self.pos_embed_type == "2d_cartesian_metric":
            pe = self.cartesian_pe + self.pe_adapt(self.cartesian_pe)
        elif self.pos_embed_type == "2d_polar_metric":
            pe = self.polar_pe + self.pe_adapt(self.polar_pe)
        elif self.pos_embed_type == "2d_separable_learned":
            H, W = self.pool_shape
            row_exp = self.row_embed.expand(-1, -1, W, -1)
            col_exp = self.col_embed.expand(-1, H, -1, -1)
            pe_2d = torch.cat([row_exp, col_exp], dim=-1)
            pe = pe_2d.view(1, H * W, -1)
        else:
            pe = getattr(self, "pos_embed", None)

        if num_tokens is not None and pe is not None and pe.shape[1] > num_tokens:
            pe = pe[:, :num_tokens, :]
        return pe


class DualPoolingProjector(BaseProjector):
    """Arch 1: Dual-Pooling Projector (CBAM-style).
    
    Applies parallel Adaptive Average Pooling and Adaptive Max Pooling to capture
    both broad context (free space, drivable area) and sharp isolated peaks (cones, pedestrians).
    """
    def __init__(self, config: ProjectorConfig) -> None:
        super().__init__(config)
        self.avg_pool = nn.AdaptiveAvgPool2d(self.pool_shape)
        self.max_pool = nn.AdaptiveMaxPool2d(self.pool_shape)
        self.pool_proj = nn.Conv2d(
            config.hidden_channels * 2, config.hidden_channels, kernel_size=1
        )

    def forward(self, bev: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.conv_stack(bev)
        avg_x = self.avg_pool(x)
        max_x = self.max_pool(x)
        x = torch.cat([avg_x, max_x], dim=1)
        x = self.pool_proj(x)

        B, C, h, w = x.shape
        x = x.view(B, C, h * w).transpose(1, 2)
        x = x + self.get_pos_embed(x.shape[1])
        x = self.mlp(x)
        x = self.out_norm(x)
        return x


class MultiScalePeakProjector(BaseProjector):
    """Arch 2: Multi-Scale Peak Injection Projector (FPN-style).
    
    Bypasses deep downsampling by extracting high-frequency peak activations
    directly from Layer 1 (100x100 resolution) via Adaptive Max Pooling and residual
    injection into the deep semantic stream, preventing tiny targets from fading out.
    """
    def __init__(self, config: ProjectorConfig) -> None:
        super().__init__(config)
        self.deep_pool = nn.AdaptiveAvgPool2d(self.pool_shape)
        self.highres_pool = nn.AdaptiveMaxPool2d(self.pool_shape)
        self.lateral_conv = nn.Sequential(
            nn.Conv2d(config.hidden_channels, config.hidden_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(config.hidden_channels),
            nn.GELU(),
        )
        self.lateral_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

    def forward(self, bev: torch.Tensor, **kwargs) -> torch.Tensor:
        # Layer 1 output: [B, 256, 100, 100]
        x_highres = self.conv_stack[0:3](bev)
        # Deep stream output: [B, 256, 13, 13]
        x_deep = self.conv_stack[3:](x_highres)

        deep_pooled = self.deep_pool(x_deep)
        highres_peak = self.highres_pool(x_highres)
        highres_peak = self.lateral_conv(highres_peak)

        # Residual multi-scale fusion
        x = deep_pooled + self.lateral_scale * highres_peak

        B, C, h, w = x.shape
        x = x.view(B, C, h * w).transpose(1, 2)
        x = x + self.get_pos_embed(x.shape[1])
        x = self.mlp(x)
        x = self.out_norm(x)
        return x


class HybridSparseProjector(BaseProjector):
    """Arch 3: Hybrid Grid + Sparse Object Queries Projector (SparseDrive-style).
    
    Produces 32 Grid Tokens (8x4) for global topological layout PLUS 8 Top-K
    Salient Object Tokens extracted directly from peak coordinate activations
    in the high-resolution BEV map. Total = 40 visual tokens.
    """
    def __init__(self, config: ProjectorConfig) -> None:
        super().__init__(config)
        self.num_sparse = getattr(config, "num_sparse_tokens", 8)
        self.total_tokens = self.num_tokens + self.num_sparse
        self.config.num_tokens = self.total_tokens

        self.grid_pool = nn.AdaptiveAvgPool2d(self.pool_shape)
        self.sparse_embed = nn.Parameter(torch.zeros(1, self.num_sparse, config.hidden_channels))
        nn.init.normal_(self.sparse_embed, std=0.02)

        self.peak_norm = nn.LayerNorm(config.hidden_channels)

    def forward(self, bev: torch.Tensor, **kwargs) -> torch.Tensor:
        # High-res stage: [B, 256, 100, 100]
        x_highres = self.conv_stack[0:3](bev)
        # Deep stream: [B, 256, 13, 13]
        x_deep = self.conv_stack[3:](x_highres)

        # 1. 32 Grid Tokens
        grid_x = self.grid_pool(x_deep)
        B, C, h, w = grid_x.shape
        grid_tokens = grid_x.view(B, C, h * w).transpose(1, 2)  # [B, 32, C]
        grid_tokens = grid_tokens + self.get_pos_embed(grid_tokens.shape[1])

        # 2. 8 Sparse Object Tokens from Salience Peaks
        # Compute spatial salience map: [B, 100, 100]
        salience = torch.linalg.vector_norm(x_highres.float(), dim=1).to(x_highres.dtype)
        # Local non-maximum suppression via max pool 5x5
        local_max = F.max_pool2d(salience.unsqueeze(1), kernel_size=5, stride=1, padding=2).squeeze(1)
        peak_mask = (salience == local_max) & (salience > 1e-4)
        filtered_salience = salience * peak_mask.to(salience.dtype)

        # Flatten and get Top-K indices per sample
        flat_salience = filtered_salience.view(B, -1)
        _, topk_indices = torch.topk(flat_salience, k=self.num_sparse, dim=-1)

        # Gather feature vectors at peak indices: [B, C, 100*100] -> [B, 8, C]
        flat_feats = x_highres.view(B, C, -1)
        topk_expanded = topk_indices.unsqueeze(1).expand(-1, C, -1)
        sparse_tokens = torch.gather(flat_feats, dim=2, index=topk_expanded).transpose(1, 2)
        sparse_tokens = self.peak_norm(sparse_tokens) + self.sparse_embed

        # 3. Concatenate: [B, 32 + 8, C] = [B, 40, C]
        all_tokens = torch.cat([grid_tokens, sparse_tokens], dim=1)

        x = self.mlp(all_tokens)
        x = self.out_norm(x)
        return x


class QuestionAwareProjector(BaseProjector):
    """Arch 4: Language-Guided Cross-Attention Projector (InstructBLIP-style).
    
    Uses text question token embeddings as queries/keys to modulate BEV spatial
    features via lightweight Cross-Attention prior to spatial pooling, ensuring
    small queried targets (e.g., 'pedestrian', 'traffic cone') are prominently highlighted.
    """
    def __init__(self, config: ProjectorConfig) -> None:
        super().__init__(config)
        self.requires_text = True
        self.grid_pool = nn.AdaptiveAvgPool2d(self.pool_shape)

        self.text_proj = nn.Linear(config.projector_output_size, config.hidden_channels)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_channels, num_heads=4, batch_first=True
        )
        self.norm_attn = nn.LayerNorm(config.hidden_channels)

    def forward(self, bev: torch.Tensor, text_embeds: Optional[torch.Tensor] = None, **kwargs) -> torch.Tensor:
        x = self.conv_stack(bev)  # [B, 256, 13, 13]
        B, C, H, W = x.shape

        if text_embeds is not None:
            # Flatten spatial cells: [B, 169, 256]
            spatial_tokens = x.view(B, C, H * W).transpose(1, 2)
            # Project text embeddings to hidden_channels
            text_feat = self.text_proj(text_embeds.to(dtype=self.text_proj.weight.dtype))
            orig_dtype = spatial_tokens.dtype
            # Run cross-attention in FP32 to prevent FP16 softmax overflow spikes
            with torch.amp.autocast("cuda", enabled=False):
                s_fp32 = spatial_tokens.float()
                t_fp32 = text_feat.float()
                attn_out, _ = self.cross_attn(
                    query=s_fp32, key=t_fp32, value=t_fp32
                )
                spatial_tokens = self.norm_attn(s_fp32 + attn_out).to(dtype=orig_dtype)
            x = spatial_tokens.transpose(1, 2).view(B, C, H, W)

        x = self.grid_pool(x)
        B, C, h, w = x.shape
        x = x.view(B, C, h * w).transpose(1, 2)
        x = x + self.get_pos_embed(x.shape[1])
        x = self.mlp(x)
        x = self.out_norm(x)
        return x


class DeeperConvProjector(BaseProjector):
    """Backwards-compatible DeeperConvProjector supporting uniform or dual/l2 pooling."""
    def __init__(self, config: ProjectorConfig) -> None:
        import copy
        # Legacy DeeperConv was designed with 1d_learned positional embedding
        if config.pos_embed_type == "2d_polar_metric":
            config = copy.copy(config)
            config.pos_embed_type = "1d_learned"
        super().__init__(config)
        self.use_dual_pooling = getattr(config, "use_dual_pooling", False)
        self.use_l2_pooling = getattr(config, "use_l2_pooling", False)
        self.l2_pooling_learnable = getattr(config, "l2_pooling_learnable", False)

        if self.use_l2_pooling and self.l2_pooling_learnable:
            self.l2_gamma = nn.Parameter(
                torch.tensor(float(getattr(config, "l2_pooling_gamma", 1.0)), dtype=torch.float32)
            )
        else:
            self.l2_gamma = float(getattr(config, "l2_pooling_gamma", 1.0))

        if self.use_dual_pooling:
            self.avg_pool = nn.AdaptiveAvgPool2d(self.pool_shape)
            self.max_pool = nn.AdaptiveMaxPool2d(self.pool_shape)
            self.pool_proj = nn.Conv2d(
                config.hidden_channels * 2, config.hidden_channels, kernel_size=1
            )
        else:
            self.pool = nn.AdaptiveAvgPool2d(self.pool_shape)

    def forward(self, bev: torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.conv_stack(bev)

        if self.use_l2_pooling and self.use_dual_pooling:
            salience = torch.linalg.vector_norm(x.float(), dim=1, keepdim=True).to(x.dtype)
            gamma = self.l2_gamma
            if isinstance(gamma, torch.Tensor):
                gamma = torch.clamp(gamma, min=0.1, max=5.0)
            if gamma != 1.0:
                salience = torch.pow(salience + 1e-6, gamma)

            num = self.avg_pool(x * salience)
            den = self.avg_pool(salience)
            l2_x = num / (den + 1e-6)
            max_x = self.max_pool(x)
            x = torch.cat([l2_x, max_x], dim=1)
            x = self.pool_proj(x)
        elif self.use_l2_pooling:
            salience = torch.linalg.vector_norm(x.float(), dim=1, keepdim=True).to(x.dtype)
            gamma = self.l2_gamma
            if isinstance(gamma, torch.Tensor):
                gamma = torch.clamp(gamma, min=0.1, max=5.0)
            if gamma != 1.0:
                salience = torch.pow(salience + 1e-6, gamma)

            num = self.pool(x * salience)
            den = self.pool(salience)
            x = num / (den + 1e-6)
        elif self.use_dual_pooling:
            avg_x = self.avg_pool(x)
            max_x = self.max_pool(x)
            x = torch.cat([avg_x, max_x], dim=1)
            x = self.pool_proj(x)
        else:
            x = self.pool(x)

        B, C, h, w = x.shape
        x = x.view(B, C, h * w).transpose(1, 2)
        x = x + self.get_pos_embed(x.shape[1])
        x = self.mlp(x)
        x = self.out_norm(x)
        return x


def build_projector(config: ProjectorConfig) -> nn.Module:
    """Factory function instantiating the requested projector architecture."""
    arch = getattr(config, "arch_type", "dual_pooling").lower()
    if arch == "dual_pooling":
        return DualPoolingProjector(config)
    elif arch == "multiscale_peak":
        return MultiScalePeakProjector(config)
    elif arch == "hybrid_sparse":
        return HybridSparseProjector(config)
    elif arch == "question_aware":
        return QuestionAwareProjector(config)
    elif arch == "deeper_conv":
        return DeeperConvProjector(config)
    else:
        raise ValueError(f"Unknown projector arch_type: {arch}. Choose from: "
                         f"['dual_pooling', 'multiscale_peak', 'hybrid_sparse', 'question_aware', 'deeper_conv']")

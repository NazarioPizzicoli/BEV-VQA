"""
Visualization utilities for BEV-VLM Multimodal Inference & Visual Debugging.

Provides:
- build_multimodal_debug_figure: Hex-view surround cameras + BEV L2 heatmap + QA info card
- render_ansi_terminal_image: 24-bit ANSI truecolor image renderer using Unicode half-block (▀)
- render_ansi_bev_grid: BEV tensor heatmap rendered in ANSI truecolor with ego marker
- render_ansi_camera_mosaic: 6-camera surround mosaic rendered in ANSI truecolor
"""

import datetime
from pathlib import Path
import textwrap
from typing import Dict, List, Optional, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for headless servers
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

SURROUND_CAMERAS = [
    "CAM_FRONT_LEFT",
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_LEFT",
    "CAM_BACK",
    "CAM_BACK_RIGHT",
]

DEFAULT_OUTPUT_DIR = Path("outputs/inference")


def compute_bev_l2_heatmap(bev_tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """
    Computes L2 norm heatmap across channel dimension for a BEV feature tensor.

    Args:
        bev_tensor: [128, 200, 200], [1, 128, 200, 200], [B, 128, 200, 200], or [H, W].

    Returns:
        2D numpy array [H, W] of float32.
    """
    if isinstance(bev_tensor, torch.Tensor):
        feat = bev_tensor.detach().cpu().to(torch.float32)
        if feat.dim() == 4:
            feat = feat[0]
        if feat.dim() == 3:
            # L2 norm across channel dimension (dim=0)
            heatmap = torch.norm(feat, p=2, dim=0).numpy()
        elif feat.dim() == 2:
            heatmap = feat.numpy()
        else:
            raise ValueError(f"Unsupported BEV tensor dimensions: {feat.shape}")
    elif isinstance(bev_tensor, np.ndarray):
        arr = bev_tensor.astype(np.float32)
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim == 3:
            heatmap = np.linalg.norm(arr, axis=0)
        elif arr.ndim == 2:
            heatmap = arr
        else:
            raise ValueError(f"Unsupported BEV array dimensions: {arr.shape}")
    else:
        raise TypeError(f"Expected torch.Tensor or np.ndarray, got {type(bev_tensor)}")

    return heatmap.astype(np.float32)


def render_ansi_terminal_image(
    img_or_path: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
    width: int = 60,
) -> str:
    """
    Renders an image in 24-bit ANSI truecolor inside the terminal using the
    Unicode upper half-block character '▀' (\\u2580).

    Each character cell renders two vertical pixels:
    - Foreground color: Top pixel
    - Background color: Bottom pixel

    Args:
        img_or_path: Path to image, PIL Image, numpy array (H, W, 3), or torch.Tensor.
        width: Number of character columns in terminal output.

    Returns:
        ANSI escape sequence string representing the image.
    """
    if isinstance(img_or_path, (str, Path)):
        p = Path(img_or_path)
        if not p.exists():
            return f"[Image not found: {p}]"
        pil_img = Image.open(p).convert("RGB")
    elif isinstance(img_or_path, Image.Image):
        pil_img = img_or_path.convert("RGB")
    elif isinstance(img_or_path, torch.Tensor):
        t = img_or_path.detach().cpu()
        if t.dim() == 3 and t.shape[0] in [1, 3]:  # [C, H, W]
            t = t.permute(1, 2, 0)
        arr = t.numpy()
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(arr).convert("RGB")
    elif isinstance(img_or_path, np.ndarray):
        arr = img_or_path
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.dtype != np.uint8:
            arr = np.clip(arr * 255 if arr.max() <= 1.0 else arr, 0, 255).astype(np.uint8)
        pil_img = Image.fromarray(arr).convert("RGB")
    else:
        raise TypeError(f"Unsupported image type: {type(img_or_path)}")

    orig_w, orig_h = pil_img.size
    if orig_w == 0 or orig_h == 0:
        return "[Empty image]"

    # Terminal cells are approx 2:1 height:width ratio.
    # Since 1 cell displays 2 pixels vertically, 1 cell height = 2 vertical pixels.
    # Therefore, target_h (pixels) should match (orig_h / orig_w) * width.
    target_h = int(round((orig_h / orig_w) * width))
    if target_h % 2 != 0:
        target_h += 1
    target_h = max(2, target_h)

    resized = pil_img.resize((width, target_h), Image.Resampling.BILINEAR)
    arr = np.array(resized, dtype=np.uint8)

    lines = []
    for y in range(0, target_h, 2):
        line_parts = []
        for x in range(width):
            top_r, top_g, top_b = arr[y, x, :3]
            if y + 1 < target_h:
                bot_r, bot_g, bot_b = arr[y + 1, x, :3]
            else:
                bot_r, bot_g, bot_b = 0, 0, 0
            line_parts.append(
                f"\033[38;2;{top_r};{top_g};{top_b}m\033[48;2;{bot_r};{bot_g};{bot_b}m\u2580"
            )
        line_parts.append("\033[0m")
        lines.append("".join(line_parts))

    return "\n".join(lines)


def render_ansi_bev_grid(
    bev_tensor: Union[torch.Tensor, np.ndarray],
    width: int = 40,
    colormap: str = "viridis",
) -> str:
    """
    Projects BEV tensor to RGB heatmap and renders it with ego marker in ANSI truecolor.

    Args:
        bev_tensor: [128, 200, 200] or [H, W] BEV feature tensor.
        width: Terminal columns for rendering.
        colormap: Matplotlib colormap name ('viridis', 'inferno', etc.).

    Returns:
        ANSI escape sequence string.
    """
    heatmap = compute_bev_l2_heatmap(bev_tensor)
    H, W = heatmap.shape

    # Robust percentile normalization
    vmin = float(np.percentile(heatmap, 2))
    vmax = float(np.percentile(heatmap, 98))
    if vmax > vmin:
        normed = np.clip((heatmap - vmin) / (vmax - vmin), 0.0, 1.0)
    else:
        normed = np.zeros_like(heatmap)

    cmap = matplotlib.colormaps.get(colormap, matplotlib.colormaps["viridis"])
    rgba = cmap(normed)  # [H, W, 4]
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)

    # In BEV origin="lower", row 0 is back (-50m), row H-1 is front (+50m).
    # Flip vertically so top of terminal is +50m (forward).
    rgb = np.flipud(rgb)

    # Calculate target height (terminal cells are approx 2:1 height:width ratio; 1 cell displays 2 vertical pixels)
    target_h = int(round((H / W) * width))
    if target_h % 2 != 0:
        target_h += 1
    target_h = max(4, target_h)

    # Resize first using bilinear interpolation
    pil_img = Image.fromarray(rgb).resize((width, target_h), Image.Resampling.BILINEAR)
    arr = np.array(pil_img, dtype=np.uint8)

    # Draw sharp ego vehicle marker '▲' at the center of the resized grid
    cy, cx = target_h // 2, width // 2
    ego_color = np.array([255, 60, 60], dtype=np.uint8)    # Bright red
    ego_accent = np.array([255, 255, 255], dtype=np.uint8) # White tip

    # Sharp upward arrowhead
    if cy - 2 >= 0 and 0 <= cx < width:
        arr[cy - 2, cx] = ego_accent
    if cy - 1 >= 0:
        for dx in range(-1, 2):
            if 0 <= cx + dx < width:
                arr[cy - 1, cx + dx] = ego_color
    if 0 <= cy < target_h:
        for dx in range(-2, 3):
            if 0 <= cx + dx < width:
                arr[cy, cx + dx] = ego_color
    if cy + 1 < target_h and 0 <= cx < width:
        arr[cy + 1, cx] = ego_accent

    # Render directly to ANSI half-block format
    lines = []
    for y in range(0, target_h, 2):
        line_parts = []
        for x in range(width):
            top_r, top_g, top_b = arr[y, x, :3]
            if y + 1 < target_h:
                bot_r, bot_g, bot_b = arr[y + 1, x, :3]
            else:
                bot_r, bot_g, bot_b = 0, 0, 0
            line_parts.append(
                f"\033[38;2;{top_r};{top_g};{top_b}m\033[48;2;{bot_r};{bot_g};{bot_b}m\u2580"
            )
        line_parts.append("\033[0m")
        lines.append("".join(line_parts))

    return "\n".join(lines)


def render_ansi_camera_mosaic(
    cam_paths: Union[Dict[str, Union[str, Path]], List[Union[str, Path]]],
    width: int = 72,
) -> str:
    """
    Renders the 6 surround cameras in a 2x3 mosaic layout in ANSI truecolor.

    Args:
        cam_paths: Dict mapping camera name -> path, or list of 6 camera paths.
        width: Terminal width for the composite mosaic.

    Returns:
        ANSI string representation of the 2x3 surround mosaic.
    """
    single_w = 480
    single_h = 270

    mosaic = Image.new("RGB", (single_w * 3, single_h * 2), color=(20, 24, 33))
    draw = ImageDraw.Draw(mosaic)

    for i, cam_name in enumerate(SURROUND_CAMERAS):
        r, c = divmod(i, 3)
        x_offset = c * single_w
        y_offset = r * single_h

        img_path = None
        if isinstance(cam_paths, dict):
            img_path = cam_paths.get(cam_name)
        elif isinstance(cam_paths, (list, tuple)) and i < len(cam_paths):
            img_path = cam_paths[i]

        loaded = False
        if img_path and Path(img_path).exists():
            try:
                cam_img = Image.open(img_path).convert("RGB")
                cam_img = cam_img.resize((single_w, single_h), Image.Resampling.BILINEAR)
                mosaic.paste(cam_img, (x_offset, y_offset))
                loaded = True
            except Exception:
                loaded = False

        if not loaded:
            # Draw placeholder box
            draw.rectangle(
                [x_offset + 2, y_offset + 2, x_offset + single_w - 2, y_offset + single_h - 2],
                fill=(35, 40, 50),
                outline=(70, 80, 100),
                width=2,
            )
            draw.text(
                (x_offset + single_w // 2 - 40, y_offset + single_h // 2 - 10),
                f"{cam_name}\n[No Feed]",
                fill=(180, 190, 205),
            )

        # Draw camera name banner at top-left of each tile
        draw.rectangle(
            [x_offset, y_offset, x_offset + 140, y_offset + 20],
            fill=(0, 0, 0, 180),
        )
        draw.text((x_offset + 6, y_offset + 4), cam_name, fill=(240, 240, 245))

    return render_ansi_terminal_image(mosaic, width=width)


def build_multimodal_debug_figure(
    sample_token: str,
    cam_paths: Union[Dict[str, Union[str, Path]], List[Union[str, Path]]],
    bev_tensor: Union[torch.Tensor, np.ndarray],
    question: str,
    prediction: str,
    ground_truth: Optional[str] = None,
    category: Optional[str] = None,
    output_path: Optional[Union[str, Path]] = None,
    colormap: str = "viridis",
) -> str:
    """
    Builds a high-resolution multimodal debug dashboard figure:
    1. 6-camera surround layout (Hex-view):
       Row 0: [CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT]
       Row 1: [CAM_BACK_LEFT, CAM_BACK, CAM_BACK_RIGHT]
    2. BEV heatmap panel: L2 norm across channels [128, 200, 200] with colormap,
       metric axes [-50m, +50m], ego-vehicle center arrow/marker '▲' pointing forward.
    3. Information Box: Sample Token, Category, Question, Ground Truth (green),
       Generated Prediction (cyan/blue).

    Saves high-res image (150 DPI) to output_path or
    'outputs/inference/{sample_token}_{timestamp}.png'.

    Args:
        sample_token: nuScenes sample token.
        cam_paths: dict or list of 6 camera image paths.
        bev_tensor: [128, 200, 200] or [1, 128, 200, 200] tensor/array.
        question: Question text.
        prediction: Generated answer string.
        ground_truth: Optional ground truth answer.
        category: Optional task category (e.g. 'DriveLM - Planning', 'NuScenes-QA - Exist').
        output_path: Optional destination file path.
        colormap: Colormap for BEV heatmap ('viridis', 'inferno', etc.).

    Returns:
        String path to the saved PNG dashboard.
    """
    # Determine destination output path ({sample_token}_{timestamp}.png)
    if output_path is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        token_str = sample_token if sample_token else "unknown"
        dest_path = DEFAULT_OUTPUT_DIR / f"{token_str}_{ts}.png"
    else:
        dest_path = Path(output_path)

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # Style definitions
    bg_color = "#0f172a"        # Slate 900
    card_bg = "#1e293b"         # Slate 800
    border_color = "#334155"    # Slate 700
    text_primary = "#f8fafc"    # Slate 50
    text_secondary = "#94a3b8"  # Slate 400
    gt_green = "#22c55e"        # Green 500
    gt_box_bg = "#064e3b"       # Green 900
    pred_cyan = "#38bdf8"       # Sky 400
    pred_box_bg = "#0c4a6e"      # Sky 900
    accent_purple = "#a855f7"   # Purple 500

    fig = plt.figure(figsize=(20, 13.5), facecolor=bg_color)
    gs = fig.add_gridspec(
        3,
        3,
        height_ratios=[1.0, 1.0, 1.25],
        width_ratios=[1.0, 1.0, 1.0],
        hspace=0.22,
        wspace=0.18,
        left=0.04,
        right=0.96,
        top=0.94,
        bottom=0.04,
    )

    # Title Banner
    fig.suptitle(
        f"BEV-VLM MULTIMODAL INFERENCE & DEBUG DASHBOARD  |  Token: {sample_token}",
        color=text_primary,
        fontsize=16,
        fontweight="bold",
        y=0.975,
    )

    # Hex-view: 6 surround cameras
    for idx, cam_name in enumerate(SURROUND_CAMERAS):
        r, c = divmod(idx, 3)
        ax_cam = fig.add_subplot(gs[r, c])
        ax_cam.set_facecolor(card_bg)

        img_path = None
        if isinstance(cam_paths, dict):
            img_path = cam_paths.get(cam_name)
        elif isinstance(cam_paths, (list, tuple)) and idx < len(cam_paths):
            img_path = cam_paths[idx]

        loaded = False
        if img_path and Path(img_path).exists():
            try:
                img = Image.open(img_path).convert("RGB")
                ax_cam.imshow(img)
                loaded = True
            except Exception:
                loaded = False

        if not loaded:
            ax_cam.text(
                0.5,
                0.5,
                f"{cam_name}\n[Image Not Available]",
                color=text_secondary,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="bold",
            )
            ax_cam.set_xlim(0, 1)
            ax_cam.set_ylim(0, 1)

        # Camera label badge
        ax_cam.set_title(
            f"[{cam_name}]",
            color=text_primary,
            fontsize=11,
            fontweight="bold",
            pad=6,
        )
        ax_cam.set_xticks([])
        ax_cam.set_yticks([])
        for spine in ax_cam.spines.values():
            spine.set_edgecolor(border_color)
            spine.set_linewidth(1.5)

    # BEV Heatmap Panel (Row 2, Col 0)
    ax_bev = fig.add_subplot(gs[2, 0])
    ax_bev.set_facecolor("#030712")

    heatmap = compute_bev_l2_heatmap(bev_tensor)
    vmin = float(np.percentile(heatmap, 1))
    vmax = float(np.percentile(heatmap, 99))
    if vmax <= vmin:
        vmax = vmin + 1.0

    cmap_obj = matplotlib.colormaps.get(colormap, matplotlib.colormaps["viridis"])
    im = ax_bev.imshow(
        heatmap,
        cmap=cmap_obj,
        extent=[-50, 50, -50, 50],
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        interpolation="bilinear",
    )

    # Distance concentric guide rings (20m, 40m)
    for radius in [20, 40]:
        circle = patches.Circle(
            (0, 0),
            radius,
            fill=False,
            color="white",
            linestyle="--",
            linewidth=0.8,
            alpha=0.4,
        )
        ax_bev.add_patch(circle)
        ax_bev.text(
            radius * 0.707,
            radius * 0.707,
            f"{radius}m",
            color="white",
            fontsize=7,
            alpha=0.6,
            va="center",
        )

    # Ego Vehicle marker: Arrow ▲ pointing forward (+Y)
    ax_bev.scatter(
        [0],
        [0],
        marker="^",
        s=220,
        c="#ef4444",  # Red 500
        edgecolors="white",
        linewidths=1.5,
        zorder=10,
        label="Ego Vehicle",
    )

    ax_bev.set_xlim(-50, 50)
    ax_bev.set_ylim(-50, 50)
    ax_bev.set_xlabel("Lateral X (m)", color=text_secondary, fontsize=10)
    ax_bev.set_ylabel("Longitudinal Y (m) [▲ Forward]", color=text_secondary, fontsize=10)
    ax_bev.set_title(
        f"BEV Feature Map (L2 Norm, {colormap})",
        color=text_primary,
        fontsize=11,
        fontweight="bold",
        pad=6,
    )
    ax_bev.tick_params(colors=text_secondary, labelsize=9)
    ax_bev.grid(True, linestyle=":", alpha=0.35, color="#64748b")
    for spine in ax_bev.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(1.5)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax_bev, orientation="vertical", fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=text_secondary, labelsize=8)
    cbar.set_label("L2 Norm", color=text_secondary, fontsize=9)
    cbar.outline.set_edgecolor(border_color)

    # Information Box (Row 2, Spanning Col 1 and 2)
    ax_info = fig.add_subplot(gs[2, 1:])
    ax_info.set_facecolor(card_bg)
    ax_info.set_xticks([])
    ax_info.set_yticks([])
    for spine in ax_info.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(1.5)

    # Helper to wrap text cleanly and adapt font size for long multi-sentence descriptions
    def _wrap_card_text(txt: str, width: int = 74):
        if not txt:
            return "", 10.5
        paras = str(txt).split("\n")
        wrapped = []
        for p in paras:
            if len(p) > width:
                wrapped.extend(textwrap.wrap(p, width=width))
            else:
                wrapped.append(p)
        line_count = len(wrapped)
        if line_count > 5:
            fsize = 8.5
        elif line_count > 3:
            fsize = 9.5
        else:
            fsize = 10.5
        return "\n".join(wrapped), fsize

    # Draw styled cards inside the info panel
    cat_text = category if category else "Autonomous Driving VQA"
    ax_info.set_title(
        f"INFERENCE DETAILS  |  Category: {cat_text}",
        color=accent_purple,
        fontsize=12,
        fontweight="bold",
        loc="left",
        pad=10,
    )

    # Metadata lines
    ax_info.text(
        0.03,
        0.88,
        f"Sample Token:  {sample_token}",
        color=text_secondary,
        fontsize=10,
        fontfamily="monospace",
        transform=ax_info.transAxes,
    )

    # Question Box
    q_rect = patches.FancyBboxPatch(
        (0.03, 0.62),
        0.94,
        0.20,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor="#0f172a",
        edgecolor="#475569",
        linewidth=1.2,
        transform=ax_info.transAxes,
    )
    ax_info.add_patch(q_rect)
    ax_info.text(
        0.05,
        0.77,
        "[QUESTION]",
        color="#fbbf24",  # Amber 400
        fontsize=9,
        fontweight="bold",
        transform=ax_info.transAxes,
    )
    q_text, q_fsize = _wrap_card_text(question, width=74)
    ax_info.text(
        0.05,
        0.73,
        q_text,
        color=text_primary,
        fontsize=q_fsize,
        fontweight="bold",
        va="top",
        transform=ax_info.transAxes,
    )

    # Ground Truth Box
    gt_rect = patches.FancyBboxPatch(
        (0.03, 0.34),
        0.94,
        0.23,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor=gt_box_bg,
        edgecolor=gt_green,
        linewidth=1.5,
        transform=ax_info.transAxes,
    )
    ax_info.add_patch(gt_rect)
    ax_info.text(
        0.05,
        0.51,
        "[GROUND TRUTH]",
        color=gt_green,
        fontsize=9,
        fontweight="bold",
        transform=ax_info.transAxes,
    )
    gt_display = ground_truth if ground_truth is not None else "[Not provided / Custom prompt]"
    gt_text, gt_fsize = _wrap_card_text(gt_display, width=74)
    ax_info.text(
        0.05,
        0.47,
        gt_text,
        color="#ecfdf5",
        fontsize=gt_fsize,
        va="top",
        transform=ax_info.transAxes,
    )

    # Generated Prediction Box
    pred_rect = patches.FancyBboxPatch(
        (0.03, 0.06),
        0.94,
        0.23,
        boxstyle="round,pad=0.015,rounding_size=0.02",
        facecolor=pred_box_bg,
        edgecolor=pred_cyan,
        linewidth=1.5,
        transform=ax_info.transAxes,
    )
    ax_info.add_patch(pred_rect)
    ax_info.text(
        0.05,
        0.23,
        "[BEV-VLM GENERATED PREDICTION]",
        color=pred_cyan,
        fontsize=9,
        fontweight="bold",
        transform=ax_info.transAxes,
    )
    pred_text, pred_fsize = _wrap_card_text(prediction, width=74)
    ax_info.text(
        0.05,
        0.19,
        pred_text,
        color="#f0f9ff",
        fontsize=pred_fsize,
        fontweight="bold",
        va="top",
        transform=ax_info.transAxes,
    )

    # Check match status indicator if GT is available
    if ground_truth is not None and ground_truth.strip():
        norm_gt = ground_truth.strip().lower()
        norm_pred = prediction.strip().lower()
        is_match = (norm_pred == norm_gt) or (norm_gt in norm_pred.split()) or (norm_gt in norm_pred)
        badge_text = "MATCH ✓" if is_match else "MISMATCH ✗"
        badge_color = gt_green if is_match else "#f97316"  # Orange 500
        ax_info.text(
            0.92,
            0.88,
            badge_text,
            color=badge_color,
            fontsize=12,
            fontweight="bold",
            ha="right",
            transform=ax_info.transAxes,
        )

    # Save figure
    fig.savefig(
        str(dest_path),
        dpi=150,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)

    return str(dest_path)

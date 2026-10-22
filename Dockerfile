# ==============================================================================
# Dockerfile for BEV-VLM (Bird's-Eye-View Vision-Language Model)
# CUDA 12.1 + PyTorch 2.4 + Python 3.11 with NVIDIA GPU acceleration
# ==============================================================================

FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

LABEL maintainer="Nazario Pizzicoli <nazariopizzicoli1@gmail.com>"
LABEL description="BEV-VLM: High-Efficiency Multimodal Bird's-Eye-View VLM for Autonomous Driving"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Install essential system dependencies and libraries for imaging/OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast, reliable Python package management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copy dependency manifest and install dependencies
COPY pyproject.toml /app/pyproject.toml
RUN uv pip install --system --no-cache -e ".[dev]"

# Pre-download required NLTK resources for offline evaluation
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# Copy repository source code
COPY . /app

# Ensure bev-vqa package is registered in editable mode
RUN uv pip install --system --no-cache --no-deps -e .

# Configure entrypoint and default command
RUN chmod +x /app/docker-entrypoint.sh

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["bash"]

# =============================================================================
# Avatar Generation Worker — RunPod Serverless (GPU)
# Generates 4 avatar video clips using Wan 2.5 text-to-video
# =============================================================================

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install ffmpeg for video encoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download the Wan 2.5 model during build to avoid cold-start delays
RUN python3 -c "\
from diffusers import AutoencoderKLWan, WanPipeline; \
import torch; \
vae = AutoencoderKLWan.from_pretrained('Wan-AI/Wan2.5-T2V-1.3B', subfolder='vae', torch_dtype=torch.float32); \
pipe = WanPipeline.from_pretrained('Wan-AI/Wan2.5-T2V-1.3B', vae=vae, torch_dtype=torch.float16); \
print('Wan 2.5 model downloaded successfully')"

# Copy handler
COPY handler.py /app/handler.py

CMD ["python3", "-u", "handler.py"]

"""RunPod Serverless handler for avatar video generation.

Uses Wan 2.5 (Wan-AI/Wan2.5-T2V-1.3B) to generate 4 avatar clips:
- idle: subtle breathing/blinking loop
- thinking: head tilt, looking up
- talking_happy: animated speaking with smile
- talking_sad: subdued speaking with concern

Generated clips are POSTed back to the SmartTalker server via:
    POST {smarttalker_url}/api/v1/avatars/{avatar_id}/upload/{state}
"""

from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import imageio
import numpy as np
import requests
import runpod
import torch
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.utils import export_to_video


# ── Model loading (runs once at cold start) ─────────────────────────────

MODEL_ID = "Wan-AI/Wan2.5-T2V-1.3B"

_pipeline = None


def load_model():
    """Load the Wan 2.5 pipeline (fp16, GPU)."""
    global _pipeline
    if _pipeline is not None:
        return

    vae = AutoencoderKLWan.from_pretrained(
        MODEL_ID,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    _pipeline = WanPipeline.from_pretrained(
        MODEL_ID,
        vae=vae,
        torch_dtype=torch.float16,
    )
    _pipeline.to("cuda")
    _pipeline.enable_model_cpu_offload()
    print(f"[avatar-gen] Wan 2.5 loaded on GPU")


# ── Clip generation prompts ─────────────────────────────────────────────

CLIP_PROMPTS = {
    "idle": (
        "A professional person standing still in a neutral pose, "
        "subtle breathing motion, occasional blink, calm expression, "
        "soft studio lighting, 4K quality, portrait framing, "
        "clean background, photorealistic"
    ),
    "thinking": (
        "A professional person tilting head slightly, looking upward "
        "thoughtfully, hand near chin, contemplative expression, "
        "soft studio lighting, 4K quality, portrait framing, "
        "clean background, photorealistic"
    ),
    "talking_happy": (
        "A professional person speaking animatedly with a warm smile, "
        "natural hand gestures, friendly expression, mouth moving "
        "as if talking, soft studio lighting, 4K quality, "
        "portrait framing, clean background, photorealistic"
    ),
    "talking_sad": (
        "A professional person speaking with a concerned sympathetic "
        "expression, subtle frown, gentle hand gestures, empathetic "
        "tone, soft studio lighting, 4K quality, portrait framing, "
        "clean background, photorealistic"
    ),
}

# Video generation parameters
VIDEO_PARAMS = {
    "num_frames": 81,
    "height": 480,
    "width": 832,
    "guidance_scale": 5.0,
    "num_inference_steps": 30,
}

FPS = 16


def generate_clip(state: str, description: str = "") -> str:
    """Generate a single avatar clip and return the path to the mp4.

    Args:
        state: Clip state name (idle, thinking, talking_happy, talking_sad).
        description: Optional character description to prepend to prompt.

    Returns:
        Path to the generated mp4 file.
    """
    load_model()

    base_prompt = CLIP_PROMPTS[state]
    prompt = f"{description}. {base_prompt}" if description else base_prompt

    print(f"[avatar-gen] Generating '{state}' clip...")
    start = time.perf_counter()

    output = _pipeline(
        prompt=prompt,
        negative_prompt="blurry, low quality, distorted face, extra limbs",
        **VIDEO_PARAMS,
    )

    elapsed = time.perf_counter() - start
    print(f"[avatar-gen] '{state}' generated in {elapsed:.1f}s")

    # Export to mp4
    temp_dir = tempfile.mkdtemp(prefix="avatar_")
    raw_path = os.path.join(temp_dir, f"{state}_raw.mp4")
    final_path = os.path.join(temp_dir, f"{state}.mp4")

    export_to_video(output.frames[0], raw_path, fps=FPS)

    # Re-encode with ffmpeg for browser compatibility (h264 + yuv420p)
    os.system(
        f'ffmpeg -y -i "{raw_path}" '
        f'-c:v libx264 -pix_fmt yuv420p -crf 23 '
        f'-movflags +faststart -an "{final_path}" '
        f'-loglevel warning'
    )

    if os.path.exists(final_path) and os.path.getsize(final_path) > 0:
        return final_path

    # Fallback: use raw output if ffmpeg failed
    return raw_path


def upload_clip(
    clip_path: str,
    state: str,
    smarttalker_url: str,
    avatar_id: str,
    license_key: str = "",
) -> bool:
    """Upload a generated clip to the SmartTalker server.

    Args:
        clip_path: Local path to the mp4 file.
        state: Clip state name.
        smarttalker_url: Base URL of the SmartTalker server.
        avatar_id: Avatar identifier.
        license_key: Optional API key for authentication.

    Returns:
        True if upload succeeded.
    """
    url = f"{smarttalker_url.rstrip('/')}/api/v1/avatars/{avatar_id}/upload/{state}"
    headers = {}
    if license_key:
        headers["X-API-Key"] = license_key

    with open(clip_path, "rb") as f:
        files = {"file": (f"{state}.mp4", f, "video/mp4")}
        resp = requests.post(url, files=files, headers=headers, timeout=120)

    if resp.status_code == 200:
        print(f"[avatar-gen] Uploaded '{state}' to {url}")
        return True
    else:
        print(f"[avatar-gen] Upload failed: {resp.status_code} — {resp.text}")
        return False


# ── RunPod handler ──────────────────────────────────────────────────────


def handler(event: dict) -> dict:
    """RunPod Serverless handler.

    Input schema:
        {
            "input": {
                "description": "A young Arab man in a suit",
                "name": "Ahmad",
                "role": "Customer Service Agent",
                "avatar_id": "ahmad-01",
                "smarttalker_url": "https://my-server.com",
                "license_key": "optional-api-key",
                "states": ["idle", "thinking", "talking_happy", "talking_sad"]
            }
        }

    Returns:
        {
            "status": "success",
            "avatar_id": "ahmad-01",
            "clips_generated": 4,
            "clips_uploaded": 4,
            "total_time_s": 120.5
        }
    """
    start = time.time()
    inp = event.get("input", {})

    description = inp.get("description", "")
    name = inp.get("name", "Avatar")
    role = inp.get("role", "")
    avatar_id = inp.get("avatar_id", "default")
    smarttalker_url = inp.get("smarttalker_url", "")
    license_key = inp.get("license_key", "")
    states = inp.get("states", list(CLIP_PROMPTS.keys()))

    if not smarttalker_url:
        return {"error": "smarttalker_url is required"}

    # Build character description from inputs
    char_desc = description
    if name and not char_desc:
        char_desc = f"A person named {name}"
    if role:
        char_desc += f", working as {role}"

    generated = 0
    uploaded = 0

    for state in states:
        if state not in CLIP_PROMPTS:
            print(f"[avatar-gen] Skipping unknown state: {state}")
            continue

        try:
            clip_path = generate_clip(state, char_desc)
            generated += 1

            if upload_clip(clip_path, state, smarttalker_url, avatar_id, license_key):
                uploaded += 1
        except Exception as exc:
            print(f"[avatar-gen] Error generating '{state}': {exc}")

    total_time = round(time.time() - start, 1)

    return {
        "status": "success" if uploaded == len(states) else "partial",
        "avatar_id": avatar_id,
        "clips_generated": generated,
        "clips_uploaded": uploaded,
        "total_time_s": total_time,
    }


# ── Entry point ─────────────────────────────────────────────────────────

runpod.serverless.start({"handler": handler})

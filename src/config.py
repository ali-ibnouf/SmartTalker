"""Application configuration using Pydantic BaseSettings.

Loads all configuration from environment variables and .env file.
Sections: ASR, LLM, TTS, Video, Upscale, API, Storage, WhatsApp.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration for SmartTalker.

    All values are loaded from environment variables and/or a .env file.
    Grouped by pipeline layer for clarity.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── General ──────────────────────────────────────────────────────────
    app_name: str = Field(default="SmartTalker", description="Application name")
    app_env: str = Field(default="development", description="Environment (development/staging/production)")
    debug: bool = Field(default=True, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")

    # ── API Server ───────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", description="API bind host")
    api_port: int = Field(default=8000, description="API bind port")
    api_workers: int = Field(default=1, description="Uvicorn worker count")
    cors_origins: str = Field(default="*", description="CORS allowed origins (comma-separated)")
    api_key: Optional[str] = Field(default=None, description="API key for authentication (optional)")
    rate_limit_per_minute: int = Field(default=60, description="Rate limit per IP per minute")

    # ── ASR (Fun-ASR Nano) ───────────────────────────────────────────────
    asr_model_id: str = Field(
        default="FunAudioLLM/Fun-ASR-Nano-2512",
        description="Fun-ASR model identifier",
    )
    asr_vad_model: str = Field(default="fsmn-vad", description="VAD model name")
    asr_device: str = Field(default="cuda:0", description="ASR compute device")
    asr_model_dir: Path = Field(default=Path("./models/funasr"), description="ASR model directory")

    # ── LLM (Qwen 2.5 via Ollama) ────────────────────────────────────────
    llm_model_name: str = Field(default="qwen2.5:14b", description="Ollama model tag")
    llm_base_url: str = Field(default="http://localhost:11434", description="Ollama API base URL")
    llm_timeout: int = Field(default=30, description="LLM request timeout in seconds")
    llm_max_tokens: int = Field(default=512, description="Max generation tokens")
    llm_temperature: float = Field(default=0.7, description="Sampling temperature")
    llm_max_history: int = Field(default=10, description="Max conversation history turns")

    # ── TTS (CosyVoice 3.0) ─────────────────────────────────────────────
    tts_model_dir: Path = Field(default=Path("./models/cosyvoice"), description="CosyVoice model directory")
    tts_device: str = Field(default="cuda:0", description="TTS compute device")
    tts_sample_rate: int = Field(default=22050, description="Output sample rate in Hz")
    tts_default_voice: str = Field(default="default", description="Default voice ID")
    tts_max_text_length: int = Field(default=1000, description="Max text length for synthesis")

    # ── Video (EchoMimicV2) ──────────────────────────────────────────────
    video_model_dir: Path = Field(default=Path("./models/echomimic"), description="EchoMimicV2 model directory")
    video_device: str = Field(default="cuda:0", description="Video compute device")
    video_fps: int = Field(default=25, description="Output video FPS")
    video_resolution: str = Field(default="512x512", description="Output resolution (WxH)")
    video_enabled: bool = Field(default=False, description="Enable video generation")

    # ── Upscale (RealESRGAN + CodeFormer) ────────────────────────────────
    upscale_model_dir: Path = Field(default=Path("./models/upscale"), description="Upscale model directory")
    upscale_device: str = Field(default="cuda:0", description="Upscale compute device")
    upscale_target_resolution: str = Field(default="1080p", description="Target upscale resolution")
    upscale_enabled: bool = Field(default=False, description="Enable upscaling")

    # ── Redis ────────────────────────────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    redis_ttl: int = Field(default=3600, description="Default cache TTL in seconds")

    # ── Storage ──────────────────────────────────────────────────────────
    storage_backend: str = Field(default="local", description="Storage backend (local)")
    storage_base_dir: Path = Field(default=Path("./outputs"), description="Output storage directory")
    storage_max_file_age_hours: int = Field(default=24, description="Auto-cleanup age in hours")
    static_files_dir: Path = Field(default=Path("./files"), description="Static files directory")

    # ── WhatsApp (Meta Business API) ─────────────────────────────────────
    whatsapp_verify_token: Optional[str] = Field(default=None, description="Webhook verify token")
    whatsapp_access_token: Optional[str] = Field(default=None, description="Graph API access token")
    whatsapp_phone_number_id: Optional[str] = Field(default=None, description="Phone number ID")
    whatsapp_app_secret: Optional[str] = Field(default=None, description="App secret for signature verification")
    whatsapp_api_version: str = Field(default="v18.0", description="Graph API version")
    whatsapp_webhook_url: Optional[str] = Field(default=None, description="Public webhook URL")

    # ── GPU ───────────────────────────────────────────────────────────────
    cuda_visible_devices: str = Field(default="0", description="CUDA_VISIBLE_DEVICES")
    gpu_memory_fraction: float = Field(default=0.9, description="Max GPU memory fraction")

    # ── WebRTC ───────────────────────────────────────────────────────────
    webrtc_enabled: bool = Field(default=False, description="Enable WebRTC signaling endpoint")
    webrtc_stun_servers: str = Field(
        default="stun:stun.l.google.com:19302",
        description="Comma-separated STUN servers",
    )
    webrtc_turn_server: Optional[str] = Field(default=None, description="TURN server URL")
    webrtc_turn_username: Optional[str] = Field(default=None, description="TURN username")
    webrtc_turn_password: Optional[str] = Field(default=None, description="TURN password")

    # ── Validators ───────────────────────────────────────────────────────

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure log level is a valid Python logging level."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in valid_levels:
            msg = f"Invalid log level '{v}'. Must be one of: {valid_levels}"
            raise ValueError(msg)
        return upper

    @field_validator("asr_model_dir", "tts_model_dir", "video_model_dir", "upscale_model_dir", "storage_base_dir", "static_files_dir")
    @classmethod
    def ensure_dir_exists(cls, v: Path) -> Path:
        """Create directory if it does not exist."""
        v.mkdir(parents=True, exist_ok=True)
        return v

    @field_validator("llm_base_url")
    @classmethod
    def validate_url(cls, v: str) -> str:
        """Ensure the URL has a valid scheme."""
        if not v.startswith(("http://", "https://")):
            msg = f"Invalid URL '{v}'. Must start with http:// or https://"
            raise ValueError(msg)
        return v.rstrip("/")

    @field_validator("gpu_memory_fraction")
    @classmethod
    def validate_gpu_fraction(cls, v: float) -> float:
        """Ensure GPU memory fraction is between 0.1 and 1.0."""
        if not 0.1 <= v <= 1.0:
            msg = f"GPU memory fraction must be between 0.1 and 1.0, got {v}"
            raise ValueError(msg)
        return v

    @field_validator("video_resolution")
    @classmethod
    def validate_resolution(cls, v: str) -> str:
        """Ensure resolution string is in WxH format."""
        parts = v.lower().split("x")
        if len(parts) != 2:
            msg = f"Resolution must be in WxH format, got '{v}'"
            raise ValueError(msg)
        try:
            w, h = int(parts[0]), int(parts[1])
            if w <= 0 or h <= 0:
                raise ValueError
        except ValueError:
            msg = f"Resolution must contain positive integers, got '{v}'"
            raise ValueError(msg) from None
        return v

    @model_validator(mode="after")
    def validate_production_config(self) -> "Settings":
        """Ensure production environment has required security settings."""
        if self.app_env == "production" and self.api_key is None:
            raise ValueError(
                "API_KEY must be set when APP_ENV=production. "
                "Set API_KEY in your environment or .env file."
            )
        return self

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins string into a list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def video_resolution_tuple(self) -> tuple[int, int]:
        """Parse resolution string into (width, height) tuple."""
        w, h = self.video_resolution.lower().split("x")
        return int(w), int(h)


def get_settings() -> Settings:
    """Create and return a cached Settings instance.

    Returns:
        Settings: Application configuration loaded from environment.
    """
    return Settings()

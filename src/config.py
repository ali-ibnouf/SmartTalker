"""Application configuration using Pydantic BaseSettings.

Loads all configuration from environment variables and .env file.
Sections: ASR, LLM, TTS, API, Storage, WhatsApp, Avatar Clips.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# Subscription plan tier definitions — price_per_second = $0.002
PLAN_TIERS: dict[str, dict] = {
    "starter": {
        "monthly_seconds": 50_000,
        "max_avatars": 1,
        "max_concurrent": 1,
        "price_monthly": 100,
        "price_yearly": 960,
    },
    "professional": {
        "monthly_seconds": 100_000,
        "max_avatars": 3,
        "max_concurrent": 3,
        "price_monthly": 200,
        "price_yearly": 1920,
    },
    "business": {
        "monthly_seconds": 200_000,
        "max_avatars": 10,
        "max_concurrent": 10,
        "price_monthly": 400,
        "price_yearly": 3840,
    },
    "enterprise": {
        "monthly_seconds": 400_000,
        "max_avatars": 999,
        "max_concurrent": 20,
        "price_monthly": 800,
        "price_yearly": 7680,
    },
}

# Top-up packages — one-time extra seconds purchases (never expire)
TOPUP_PACKAGES: dict[str, dict] = {
    "small": {"seconds": 10_000, "price": 20},
    "medium": {"seconds": 25_000, "price": 50},
    "large": {"seconds": 50_000, "price": 100},
}

# ── Supported Languages (32) ─────────────────────────────────────────────────
# Each entry: code, name (English), name_native, rtl (right-to-left)
SUPPORTED_LANGUAGES: list[dict[str, str | bool]] = [
    {"code": "ar", "name": "Arabic", "name_native": "العربية", "rtl": True},
    {"code": "en", "name": "English", "name_native": "English", "rtl": False},
    {"code": "fr", "name": "French", "name_native": "Français", "rtl": False},
    {"code": "tr", "name": "Turkish", "name_native": "Türkçe", "rtl": False},
    {"code": "es", "name": "Spanish", "name_native": "Español", "rtl": False},
    {"code": "pt", "name": "Portuguese", "name_native": "Português", "rtl": False},
    {"code": "de", "name": "German", "name_native": "Deutsch", "rtl": False},
    {"code": "it", "name": "Italian", "name_native": "Italiano", "rtl": False},
    {"code": "nl", "name": "Dutch", "name_native": "Nederlands", "rtl": False},
    {"code": "ru", "name": "Russian", "name_native": "Русский", "rtl": False},
    {"code": "zh", "name": "Chinese", "name_native": "中文", "rtl": False},
    {"code": "ja", "name": "Japanese", "name_native": "日本語", "rtl": False},
    {"code": "ko", "name": "Korean", "name_native": "한국어", "rtl": False},
    {"code": "hi", "name": "Hindi", "name_native": "हिन्दी", "rtl": False},
    {"code": "bn", "name": "Bengali", "name_native": "বাংলা", "rtl": False},
    {"code": "ur", "name": "Urdu", "name_native": "اردو", "rtl": True},
    {"code": "ms", "name": "Malay", "name_native": "Bahasa Melayu", "rtl": False},
    {"code": "id", "name": "Indonesian", "name_native": "Bahasa Indonesia", "rtl": False},
    {"code": "th", "name": "Thai", "name_native": "ไทย", "rtl": False},
    {"code": "vi", "name": "Vietnamese", "name_native": "Tiếng Việt", "rtl": False},
    {"code": "tl", "name": "Filipino", "name_native": "Filipino", "rtl": False},
    {"code": "sw", "name": "Swahili", "name_native": "Kiswahili", "rtl": False},
    {"code": "ha", "name": "Hausa", "name_native": "Hausa", "rtl": False},
    {"code": "am", "name": "Amharic", "name_native": "አማርኛ", "rtl": False},
    {"code": "so", "name": "Somali", "name_native": "Soomaali", "rtl": False},
    {"code": "fa", "name": "Persian", "name_native": "فارسی", "rtl": True},
    {"code": "he", "name": "Hebrew", "name_native": "עברית", "rtl": True},
    {"code": "ku", "name": "Kurdish", "name_native": "کوردی", "rtl": True},
    {"code": "ps", "name": "Pashto", "name_native": "پښتو", "rtl": True},
    {"code": "pl", "name": "Polish", "name_native": "Polski", "rtl": False},
    {"code": "uk", "name": "Ukrainian", "name_native": "Українська", "rtl": False},
    {"code": "ro", "name": "Romanian", "name_native": "Română", "rtl": False},
]

# Quick lookup sets derived from the list above
LANGUAGE_CODES: frozenset[str] = frozenset(lang["code"] for lang in SUPPORTED_LANGUAGES)
RTL_LANGUAGES: frozenset[str] = frozenset(
    lang["code"] for lang in SUPPORTED_LANGUAGES if lang["rtl"]
)
DEFAULT_LANGUAGE = "ar"


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
    debug: bool = Field(default=False, description="Enable debug mode (must be False in production)")
    log_level: str = Field(default="INFO", description="Logging level")

    # ── Database (PostgreSQL) ──────────────────────────────────────────
    database_url: str = Field(
        default="postgresql+asyncpg://smarttalker:smarttalker@localhost:5432/smarttalker",
        description="SQLAlchemy async database URL",
    )

    # ── API Server ───────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0", description="API bind host")
    api_port: int = Field(default=8000, description="API bind port")
    api_workers: int = Field(default=1, description="Uvicorn worker count")
    cors_origins: str = Field(default="*", description="CORS allowed origins (comma-separated)")
    api_key: Optional[str] = Field(default=None, description="API key for authentication (optional)")
    admin_api_key: Optional[str] = Field(default=None, description="Separate admin API key for /admin/* endpoints")
    rate_limit_per_minute: int = Field(default=60, description="Rate limit per IP per minute")
    pipeline_timeout: int = Field(default=120, description="Max seconds for a single pipeline request")

    # ── DashScope (shared API key for LLM, ASR, TTS) ─────────────────────
    dashscope_api_key: str = Field(default="", description="DashScope API key (used by LLM, ASR, TTS)")
    dashscope_base_url: str = Field(
        default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        description="DashScope OpenAI-compatible REST base URL",
    )
    dashscope_ws_url: str = Field(
        default="wss://dashscope-intl.aliyuncs.com/api-ws/v1/realtime",
        description="DashScope WebSocket URL for realtime ASR/TTS",
    )

    # ── ASR (DashScope qwen3-asr) ──────────────────────────────────────
    asr_model: str = Field(default="qwen3-asr-flash-realtime", description="DashScope ASR model")

    # ── LLM (Qwen3 via DashScope, OpenAI-compatible) ──────────────────────
    llm_model_name: str = Field(default="qwen3-max", description="DashScope LLM model name")
    llm_base_url: str = Field(
        default="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        description="DashScope OpenAI-compatible API base URL",
    )
    llm_api_key: str = Field(default="", description="DashScope API key (fallback, prefer dashscope_api_key)")
    llm_timeout: int = Field(default=30, description="LLM request timeout in seconds")
    llm_max_tokens: int = Field(default=512, description="Max generation tokens")
    llm_temperature: float = Field(default=0.7, description="Sampling temperature")
    llm_max_history: int = Field(default=10, description="Max conversation history turns")

    # ── TTS (DashScope qwen3-tts) ─────────────────────────────────────────
    tts_model: str = Field(default="qwen3-tts-vc-realtime", description="DashScope TTS model")
    tts_sample_rate: int = Field(default=48000, description="Output sample rate in Hz (DashScope outputs 48kHz)")
    tts_max_text_length: int = Field(default=1000, description="Max text length for synthesis")

    # ── RunPod Serverless ─────────────────────────────────────────────────
    runpod_api_key: str = Field(default="", description="RunPod API key")
    runpod_endpoint_musetalk: str = Field(default="", description="RunPod MuseTalk endpoint URL")
    runpod_endpoint_preprocess: str = Field(default="", description="RunPod face preprocess endpoint URL")

    # ── Cloudflare R2 ─────────────────────────────────────────────────────
    r2_account_id: str = Field(default="", description="Cloudflare R2 account ID")
    r2_access_key_id: str = Field(default="", description="R2 S3-compatible access key ID")
    r2_secret_access_key: str = Field(default="", description="R2 S3-compatible secret access key")
    r2_bucket: str = Field(default="maskki-media", description="R2 bucket name")
    r2_public_url: str = Field(default="", description="R2 public URL (e.g. https://media.maskki.com)")

    # ── Redis ────────────────────────────────────────────────────────────
    redis_url: str = Field(default="redis://localhost:6379/0", description="Redis connection URL")
    redis_ttl: int = Field(default=3600, description="Default cache TTL in seconds")

    # ── Storage ──────────────────────────────────────────────────────────
    storage_backend: str = Field(default="local", description="Storage backend (local)")
    storage_base_dir: Path = Field(default=Path("./outputs"), description="Output storage directory")
    storage_max_file_age_hours: int = Field(default=24, description="Auto-cleanup age in hours")
    static_files_dir: Path = Field(default=Path("./files"), description="Static files directory")

    # ── Avatar Clips ─────────────────────────────────────────────────────
    clips_dir: Path = Field(default=Path("./clips"), description="Avatar video clips directory")

    # ── Knowledge Base (RAG) ───────────────────────────────────────────
    kb_enabled: bool = Field(default=True, description="Enable Knowledge Base / RAG")
    kb_storage_dir: Path = Field(default=Path("./data/kb"), description="KB storage directory (ChromaDB + docs)")
    kb_embedding_model: str = Field(default="text-embedding-v3", description="DashScope embedding model name")
    kb_embedding_api_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        description="DashScope embedding API base URL",
    )
    kb_embedding_api_key: str = Field(default="", description="DashScope embedding API key (reuse LLM key)")
    kb_chunk_size: int = Field(default=500, description="Document chunk size in characters")
    kb_chunk_overlap: int = Field(default=50, description="Overlap between chunks in characters")
    kb_top_k: int = Field(default=3, description="Number of context chunks to retrieve")
    kb_confidence_threshold: float = Field(default=0.6, description="Min similarity for KB answer (0.0-1.0)")

    # ── Training Engine ────────────────────────────────────────────────
    training_enabled: bool = Field(default=True, description="Enable training engine")
    training_db_path: Path = Field(default=Path("./data/training.db"), description="SQLite fallback for training data")
    training_escalation_threshold: float = Field(default=0.5, description="Default confidence threshold for escalation")
    training_go_live_threshold: float = Field(default=100.0, description="Overall progress % required for go-live")

    # ── Guardrails ──────────────────────────────────────────────────
    guardrails_enabled: bool = Field(default=True, description="Enable content guardrails engine")
    guardrails_max_response_length: int = Field(default=2000, description="Max characters in LLM response")
    guardrails_blocked_topics: str = Field(default="", description="Comma-separated default blocked topic keywords")
    guardrails_required_disclaimers: str = Field(default="", description="Comma-separated default required disclaimer phrases")

    # ── Billing ───────────────────────────────────────────────────────
    billing_enabled: bool = Field(default=True, description="Enable billing engine")
    billing_rate_per_second: float = Field(default=0.002, description="Cost per second per session (USD)")
    billing_grace_period_s: int = Field(default=5, description="Grace period in seconds before billing starts")

    # ── WhatsApp (Meta Business API) ─────────────────────────────────────
    whatsapp_verify_token: Optional[str] = Field(default=None, description="Webhook verify token")
    whatsapp_access_token: Optional[str] = Field(default=None, description="Graph API access token")
    whatsapp_phone_number_id: Optional[str] = Field(default=None, description="Phone number ID")
    whatsapp_app_secret: Optional[str] = Field(default=None, description="App secret for signature verification")
    whatsapp_api_version: str = Field(default="v18.0", description="Graph API version")
    whatsapp_webhook_url: Optional[str] = Field(default=None, description="Public webhook URL")

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

    @field_validator("storage_base_dir", "static_files_dir", "clips_dir", "kb_storage_dir")
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

    @model_validator(mode="after")
    def validate_production_config(self) -> "Settings":
        """Ensure production environment has required security settings."""
        if self.app_env == "production":
            if self.api_key is None:
                raise ValueError(
                    "API_KEY must be set when APP_ENV=production. "
                    "Set API_KEY in your environment or .env file."
                )
            if self.debug:
                raise ValueError(
                    "DEBUG=true is forbidden in production. "
                    "Set DEBUG=false for production deployments."
                )
            # Enforce WhatsApp secrets are not placeholders
            if self.whatsapp_access_token and self.whatsapp_app_secret is None:
                raise ValueError(
                    "WHATSAPP_APP_SECRET must be set when WHATSAPP_ACCESS_TOKEN is configured. "
                    "This is required for webhook signature verification."
                )
        return self

    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins string into a list."""
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


_settings_instance: Settings | None = None


def get_settings() -> Settings:
    """Return the singleton Settings instance.

    Creates it on first call, then returns the cached copy.

    Returns:
        Settings: Application configuration loaded from environment.
    """
    global _settings_instance
    if _settings_instance is None:
        _settings_instance = Settings()
    return _settings_instance

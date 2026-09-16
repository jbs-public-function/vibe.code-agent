"""Settings loaded from environment and `.env` (ADR-10, FR-00.4, NFR-00.4)."""

from __future__ import annotations

from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_BIND = "127.0.0.1"

# Tag suggested by validation when VLLM_IMAGE_TAG is unset or "latest".
SUGGESTED_IMAGE_TAG = "v0.11.2"

# Cross-component ports (ADR-4). Only the bind address is configurable (XR-1).
CONTROLLER_PORT = 50051
GATEWAY_PORT = 8080
ADMIN_PORT = 8081


class Settings(BaseSettings):
    """vibe configuration.

    Every key is overridable via an environment variable or `.env` (ADR-10).
    """

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    hf_token: str = ""
    vibe_data_dir: str = "~/.cache/huggingface"
    vibe_bind: str = DEFAULT_BIND
    vibe_swap_on_request: bool = False
    vibe_swap_timeout: int = 180
    vibe_load_timeout: int = 600
    vibe_log_level: str = "info"
    vllm_image_tag: str = ""

    @field_validator("vllm_image_tag")
    @classmethod
    def _check_image_tag(cls, v: str) -> str:
        tag = (v or "").strip()
        if not tag or tag.lower() == "latest":
            raise ValueError(
                f"VLLM_IMAGE_TAG must be a pinned tag (e.g. {SUGGESTED_IMAGE_TAG}); "
                "'latest' is rejected (XR-6)"
            )
        return tag

    @field_validator("vibe_swap_timeout", "vibe_load_timeout")
    @classmethod
    def _check_positive_timeout(cls, v: int) -> int:
        if v < 1:
            raise ValueError("timeouts must be >= 1 second")
        return v

    @property
    def hf_home(self) -> Path:
        """Shared HF cache dir used by vLLM containers and the laboratory (ADR-8)."""
        return Path(self.vibe_data_dir).expanduser()

    @property
    def image(self) -> str:
        """Full pinned image reference (ADR-3)."""
        return f"vllm/vllm-openai:{self.vllm_image_tag}"

    @property
    def controller_target(self) -> str:
        return f"{self.vibe_bind}:{CONTROLLER_PORT}"

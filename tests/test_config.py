"""Settings validation tests (NFR-00.4)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from vibe.config import SUGGESTED_IMAGE_TAG, Settings

PINNED_TAG = "v0.11.2"


def test_suggested_image_tag_constant():
    assert SUGGESTED_IMAGE_TAG == "v0.11.2"


def test_pinned_tag_accepted_and_image():
    settings = Settings(vllm_image_tag=PINNED_TAG)
    assert settings.vllm_image_tag == PINNED_TAG
    assert settings.image == f"vllm/vllm-openai:{PINNED_TAG}"


@pytest.mark.parametrize("tag", ["latest", ""])
def test_unpinned_tags_rejected(tag):
    with pytest.raises(ValidationError):
        Settings(vllm_image_tag=tag)


@pytest.mark.parametrize("tag", ["latest", ""])
def test_tag_error_message_mentions_suggested_tag(tag):
    with pytest.raises(ValidationError) as excinfo:
        Settings(vllm_image_tag=tag)
    assert SUGGESTED_IMAGE_TAG in str(excinfo.value)
    assert "v0.11.2" in str(excinfo.value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vibe_swap_timeout": 0},
        {"vibe_load_timeout": 0},
    ],
)
def test_zero_timeouts_rejected(kwargs):
    with pytest.raises(ValidationError):
        Settings(vllm_image_tag=PINNED_TAG, **kwargs)


def test_hf_home_expands_tilde():
    settings = Settings(
        vllm_image_tag=PINNED_TAG,
        vibe_data_dir="~/hf-cache",
    )
    assert "~" not in str(settings.hf_home)
    assert settings.hf_home == Path("~/hf-cache").expanduser()


def test_default_vibe_data_dir():
    settings = Settings(vllm_image_tag=PINNED_TAG)
    assert settings.hf_home == Path("~/.cache/huggingface").expanduser()

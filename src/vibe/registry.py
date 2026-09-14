"""`models.yaml` registry — the single source of truth for model parameters (ADR-2, XR-3).

Validation rules (FR-00.2):
- duplicate `alias`, `host_port`, or `container_name` are rejected
- `alias` must match ``^[a-z0-9][a-z0-9.-]*$``
- `gpu_memory_utilization` must be in ``(0, 1]``
- `max_model_len` must be >= 1
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

ALIAS_RE = re.compile(r"^[a-z0-9][a-z0-9.-]*$")
PORT_MIN, PORT_MAX = 1, 65535


class ModelSpec(BaseModel):
    """One entry of `models.yaml` (FR-00.1)."""

    model_config = ConfigDict(extra="forbid")

    alias: str
    display_name: str
    hf_repo: str
    quantization: str = "none"
    max_model_len: int
    gpu_memory_utilization: float
    container_name: str
    host_port: int
    extra_args: list[str] = Field(default_factory=list)
    use_case: str = ""
    tags: list[str] = Field(default_factory=list)

    @field_validator("alias")
    @classmethod
    def _check_alias(cls, v: str) -> str:
        if not ALIAS_RE.match(v):
            raise ValueError(
                f"alias '{v}' must match {ALIAS_RE.pattern} "
                "(lowercase letters, digits, '.', '-'; must start with letter/digit)"
            )
        return v

    @field_validator("gpu_memory_utilization")
    @classmethod
    def _check_gmem(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError(f"gpu_memory_utilization must be in (0, 1], got {v}")
        return v

    @field_validator("max_model_len")
    @classmethod
    def _check_mml(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"max_model_len must be >= 1, got {v}")
        return v

    @field_validator("host_port")
    @classmethod
    def _check_port(cls, v: int) -> int:
        if not PORT_MIN <= v <= PORT_MAX:
            raise ValueError(f"host_port must be in [{PORT_MIN}, {PORT_MAX}], got {v}")
        return v


class RegistryError(ValueError):
    """All registry validation problems collected at once (FR-00.4)."""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("\n".join(errors))


class Registry:
    """Immutable snapshot of `models.yaml` with alias lookups."""

    def __init__(self, specs: list[ModelSpec], path: Path | None = None) -> None:
        self._by_alias: dict[str, ModelSpec] = {s.alias: s for s in specs}
        self.path = path

    # -- construction -----------------------------------------------------

    @classmethod
    def from_dict(cls, data: dict[str, Any], path: Path | None = None) -> Registry:
        errors: list[str] = []
        raw_models = data.get("models", []) if isinstance(data, dict) else []
        if not isinstance(raw_models, list):
            raise RegistryError(["models.yaml: top-level 'models' must be a list"])

        specs: list[ModelSpec] = []
        seen_alias: list[str] = []
        seen_port: list[int] = []
        seen_container: list[str] = []
        for i, raw in enumerate(raw_models):
            prefix = f"models[{i}]"
            if not isinstance(raw, dict):
                errors.append(f"{prefix}: entry must be a mapping, got {type(raw).__name__}")
                continue
            try:
                spec = ModelSpec.model_validate(raw)
            except ValidationError as exc:
                for err in exc.errors():
                    loc = ".".join(str(part) for part in err["loc"])
                    errors.append(f"{prefix}{'.' + loc if loc else ''}: {err['msg']}")
                continue
            specs.append(spec)
            seen_alias.append(spec.alias)
            seen_port.append(spec.host_port)
            seen_container.append(spec.container_name)

        # Cross-entry uniqueness (FR-00.2)
        groups: list[tuple[str, list[Any]]] = [
            ("alias", seen_alias),
            ("host_port", seen_port),
            ("container_name", seen_container),
        ]
        for field, values in groups:
            for dupe, count in Counter(values).items():
                if count > 1:
                    errors.append(f"duplicate {field}: {dupe}")

        if errors:
            raise RegistryError(errors)
        return cls(specs, path=path)

    @classmethod
    def load(cls, path: Path) -> Registry:
        try:
            text = path.read_text()
        except FileNotFoundError as exc:
            raise RegistryError([f"models.yaml not found at {path}"]) from exc
        try:
            data = yaml.safe_load(text) or {}
        except yaml.YAMLError as exc:
            raise RegistryError([f"models.yaml is not valid YAML: {exc}"]) from exc
        return cls.from_dict(data, path=path)

    @classmethod
    def default_path(cls) -> Path:
        """Repo-root `models.yaml` next to `src/vibe/registry.py`."""
        return Path(__file__).resolve().parents[2] / "models.yaml"

    # -- access -----------------------------------------------------------

    def get(self, alias: str) -> ModelSpec:
        try:
            return self._by_alias[alias]
        except KeyError:
            raise KeyError(alias) from None

    def by_port(self, port: int) -> ModelSpec | None:
        for spec in self._by_alias.values():
            if spec.host_port == port:
                return spec
        return None

    def all(self) -> list[ModelSpec]:
        return list(self._by_alias.values())

    @property
    def aliases(self) -> list[str]:
        return list(self._by_alias)

    def __len__(self) -> int:
        return len(self._by_alias)

    def __contains__(self, alias: str) -> bool:
        return alias in self._by_alias

    def __iter__(self) -> Iterator[ModelSpec]:
        return iter(self._by_alias.values())

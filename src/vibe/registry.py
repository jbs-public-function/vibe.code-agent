"""`models.yaml` registry — the single source of truth for model parameters (ADR-2, XR-3).

Validation rules (FR-00.2):
- duplicate `alias`, `host_port`, or `container_name` are rejected
- `alias` must match ``^[a-z0-9][a-z0-9.-]*$``
- `gpu_memory_utilization` must be in ``(0, 1]``
- `max_model_len` must be >= 1

Phase 4 (FR-04.5/FR-04.6, NFR-04.3): `models.yaml` is a multi-document
YAML file — the original hand-edited document followed by `vibe register`
blocks appended with a marker comment. `load` merges every document's
`models:` list in order; `write_entry` appends atomically (temp file +
`os.replace`), preserving the existing content byte-for-byte.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import textwrap
import threading
import uuid
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

if TYPE_CHECKING:
    from vibe.controller.cache import CacheScanner

ALIAS_RE = re.compile(r"^[a-z0-9][a-z0-9.-]*$")
PORT_MIN, PORT_MAX = 1, 65535

# `vibe register` defaults allocate host ports in this range (FR-04.5).
REGISTER_PORT_MIN, REGISTER_PORT_MAX = 8010, 8099

# IMP-02.20: serialize `write_entry` end-to-end (load → duplicate checks →
# append → os.replace) so two concurrent `WriteRegistration` calls cannot
# both pass the checks (e.g. the same auto-port) and produce a file that
# fails to reload. A process-local lock suffices: the controller is the
# only writer of models.yaml (XR-2/XR-3); everything else only reads it.
_WRITE_LOCK = threading.Lock()

QUANTIZATIONS = {"awq", "gptq", "none"}
TOOL_PARSERS = ("hermes", "llama3_json")

# Fallback `max_model_len` source when the repo's config.json has no
# `max_position_embeddings` (FR-04.5 heuristics table).
DEFAULT_MAX_MODEL_LEN = 8192

# Register heuristics for a 16 GB card (FR-04.5 table).
_GIB = 1024**3
_HEURISTIC_SMALL_GIB = 3.0
_HEURISTIC_MID_GIB = 6.0
_HEURISTIC_MAX_GIB = 10.0

# Warn about VRAM when the estimate (weights + ~2 GiB overhead) leaves less
# than ~6 GiB of headroom on the 16 GB card.
_VRAM_WARN_GIB = 8.0
_VRAM_OVERHEAD_GIB = 2.0


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
        """Load `models.yaml`, merging every YAML document (FR-04.6).

        The file is a multi-document stream: the hand-edited document
        followed by `vibe register` blocks appended by `write_entry`
        (NFR-04.3). All `models:` lists are merged in document order.
        """
        try:
            text = path.read_text()
        except FileNotFoundError as exc:
            raise RegistryError([f"models.yaml not found at {path}"]) from exc
        try:
            documents = list(yaml.safe_load_all(text))
        except yaml.YAMLError as exc:
            raise RegistryError([f"models.yaml is not valid YAML: {exc}"]) from exc
        merged: list[Any] = []
        for document in documents:
            if document is None:
                continue
            if not isinstance(document, dict):
                raise RegistryError(
                    ["models.yaml: each YAML document must be a mapping"]
                )
            models = document.get("models", [])
            if not isinstance(models, list):
                raise RegistryError(
                    ["models.yaml: top-level 'models' must be a list"]
                )
            merged.extend(models)
        return cls.from_dict({"models": merged}, path=path)

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

    # -- serialization -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Registry snapshot in `models.yaml` shape; round-trips `from_dict`."""
        return {"models": [spec.model_dump() for spec in self._by_alias.values()]}

    # -- reload (FR-04.6) --------------------------------------------------

    def reload(self) -> list[str]:
        """Re-read `self.path` into a new snapshot; `[]` on success.

        Validate-before-swap: on failure the errors are returned and the
        live registry keeps serving the last good snapshot, so existing
        references (servicer, runner) stay valid.
        """
        path = self.path or self.default_path()
        try:
            new = Registry.load(path)
        except RegistryError as exc:
            return exc.errors
        self._by_alias = new._by_alias
        return []

    # -- registration (FR-04.5) --------------------------------------------

    def propose(
        self,
        spec: dict[str, Any] | ModelSpec,
        cache: CacheScanner,
    ) -> tuple[dict[str, Any], list[str], list[str]]:
        """Validate a registration and fill every default (FR-04.5 dry run).

        Returns `(entry, warnings, errors)`; errors never overwrite the
        registry. Defaults follow the 16 GB heuristics table, weights come
        from the HF cache (`cache.weights_bytes`), `max_model_len` starts
        from the snapshot's `config.json` `max_position_embeddings` capped
        per bucket, and `host_port` is the next free port in
        `REGISTER_PORT_MIN`-`REGISTER_PORT_MAX`.
        """
        raw = spec.model_dump() if isinstance(spec, ModelSpec) else dict(spec)
        warnings: list[str] = []
        errors: list[str] = []

        hf_repo = str(raw.get("hf_repo") or "").strip()
        alias = str(raw.get("alias") or "").strip()
        if not hf_repo:
            errors.append("hf_repo is required")
        elif hf_repo.startswith("~"):
            # IMP-02.11: vLLM cannot see an unexpanded `~` (there is no
            # mount for it) — require the absolute path instead.
            errors.append(
                f"hf_repo must be an absolute path, not an unexpanded '~' — "
                f"vLLM cannot see '{hf_repo}'; use the full path"
            )
        elif not _is_local_path(hf_repo) and "/" not in hf_repo:
            errors.append(
                f"hf_repo must be 'org/repo' or an absolute path, got '{hf_repo}'"
            )
        if not alias:
            errors.append("alias is required")
        elif not ALIAS_RE.match(alias):
            errors.append(
                f"alias '{alias}' must match {ALIAS_RE.pattern} "
                "(lowercase letters, digits, '.', '-'; must start with letter/digit)"
            )

        display_name = str(raw.get("display_name") or "").strip()
        if not display_name and hf_repo:
            display_name = _humanize_repo(hf_repo)

        container_name = str(raw.get("container_name") or "").strip()
        if not container_name and alias:
            container_name = f"vibe-{alias}"

        host_port = _to_int(raw.get("host_port"))
        if host_port is None:
            errors.append("host_port must be an integer")
            host_port = 0
        if host_port:
            if not PORT_MIN <= host_port <= PORT_MAX:
                errors.append(
                    f"host_port must be in [{PORT_MIN}, {PORT_MAX}], got {host_port}"
                )
        else:
            # IMP-02.24: compute the auto-port from the on-disk registry
            # when a path is set, so the port `propose` suggests is the one
            # `write_entry` re-checks against the freshly loaded file.
            host_port = self._next_free_port(self.path)
            if host_port is None:
                errors.append(
                    f"no free port in {REGISTER_PORT_MIN}–{REGISTER_PORT_MAX}"
                )

        quantization = str(raw.get("quantization") or "none").strip() or "none"
        if quantization not in QUANTIZATIONS:
            errors.append(
                "quantization must be one of "
                + ", ".join(sorted(QUANTIZATIONS))
                + f", got '{quantization}'"
            )

        tool_parser = str(raw.get("tool_parser") or "").strip()
        extra_args = [str(arg) for arg in raw.get("extra_args") or []]
        if tool_parser:
            if tool_parser in TOOL_PARSERS:
                extra_args = [
                    "--enable-auto-tool-choice",
                    "--tool-call-parser",
                    tool_parser,
                ]
            else:
                errors.append(
                    f"tool_parser must be one of {', '.join(TOOL_PARSERS)}, "
                    f"got '{tool_parser}'"
                )

        # Weights, files, and config.json from the cache (or a local dir).
        weights: int | None = None
        config: dict[str, Any] | None = None
        files: list[Path] = []
        if hf_repo:
            files = cache.repo_files(hf_repo)
            weights = cache.weights_bytes(hf_repo)
            if _is_local_path(hf_repo):
                if not Path(hf_repo).expanduser().is_dir():
                    errors.append(f"local model path does not exist: {hf_repo}")
            elif weights is None:
                errors.append(
                    f"download the repo first: `vibe download {hf_repo}`"
                )
            # IMP-02.21: real GGUF repos ship a README and config.json, so
            # "every file is .gguf" never matches them — detect GGUF repos
            # as "has .gguf weights and no .safetensors/.bin weights".
            has_gguf = any(path.suffix == ".gguf" for path in files)
            has_weights = any(
                path.suffix in {".safetensors", ".bin"} for path in files
            )
            if has_gguf and not has_weights:
                errors.append(
                    "GGUF-only repos are not served by vLLM in v1 — "
                    "use laboratory/inference_gguf.py (Phase 5)"
                )
            elif weights == 0:
                # IMP-02.22: cached but weightless — taking the ≤3 GiB
                # bucket would produce an unservable vLLM entry.
                errors.append(
                    f"no *.safetensors weights found in '{hf_repo}' — "
                    "vLLM cannot serve it"
                )
            config = _read_config_json(files)

        # Size heuristics for a 16 GB card (FR-04.5 table).
        gpu_memory_utilization = 0.0
        max_model_len = 1
        heuristic: tuple[float, int] | None = None
        weights_gib = 0.0
        if weights is not None:
            weights_gib = weights / _GIB
            heuristic = _heuristic_for_bytes(weights)
            if heuristic is None:
                errors.append(
                    f"model weights ~{weights_gib:.1f} GiB exceed the "
                    f"{_HEURISTIC_MAX_GIB:.0f} GiB limit for a 16 GB card"
                )
            else:
                gpu_mem, cap = heuristic
                gpu_memory_utilization = gpu_mem
                max_position_embeddings = (
                    config.get("max_position_embeddings")
                    if config is not None
                    else None
                )
                config_mml = (
                    max_position_embeddings
                    if isinstance(max_position_embeddings, int)
                    else DEFAULT_MAX_MODEL_LEN
                )
                max_model_len = max(min(config_mml, cap), 1)
                if weights_gib >= _VRAM_WARN_GIB:
                    estimate = weights_gib + _VRAM_OVERHEAD_GIB
                    warnings.append(
                        f"VRAM estimate ~{estimate:.1f} GiB (weights "
                        f"{weights_gib:.1f} GiB + ~{_VRAM_OVERHEAD_GIB:.0f} "
                        "GiB overhead) is close to the 16 GB limit"
                    )

        # Explicit caller overrides win over the heuristics.
        caller_mml = _to_int(raw.get("max_model_len"))
        if caller_mml is None:
            errors.append("max_model_len must be an integer")
        elif caller_mml:
            if caller_mml < 1:
                errors.append(f"max_model_len must be >= 1, got {caller_mml}")
            else:
                max_model_len = caller_mml

        caller_gmem = _to_float(raw.get("gpu_memory_utilization"))
        if caller_gmem is None:
            errors.append("gpu_memory_utilization must be a number")
        elif caller_gmem:
            if not 0.0 < caller_gmem <= 1.0:
                errors.append(
                    f"gpu_memory_utilization must be in (0, 1], got {caller_gmem}"
                )
            else:
                gpu_memory_utilization = caller_gmem
                if heuristic is not None and caller_gmem > heuristic[0]:
                    warnings.append(
                        f"gpu_memory_utilization {caller_gmem} exceeds the "
                        f"recommended {heuristic[0]} for ~{weights_gib:.1f} GiB "
                        "of weights"
                    )

        # config.json quantization_config cross-check (spec heuristics).
        has_quant_config: bool | None = None
        if config is not None:
            has_quant_config = "quantization_config" in config
        if quantization != "none" and has_quant_config is False:
            warnings.append(
                "config.json has no quantization_config but "
                f"--quantization {quantization} was requested"
            )
        elif quantization == "none" and has_quant_config is True:
            warnings.append(
                "config.json declares quantization_config but "
                "--quantization none was requested"
            )

        # Never collide with the live registry (FR-00.2).
        if alias and alias in self._by_alias:
            errors.append(f"duplicate alias: {alias}")
        if host_port and self.by_port(host_port) is not None:
            errors.append(f"duplicate host_port: {host_port}")
        if container_name and any(
            spec.container_name == container_name for spec in self._by_alias.values()
        ):
            errors.append(f"duplicate container_name: {container_name}")

        entry: dict[str, Any] = {
            "alias": alias,
            "display_name": display_name,
            "hf_repo": hf_repo,
            "quantization": quantization,
            "max_model_len": max_model_len,
            "gpu_memory_utilization": gpu_memory_utilization,
            "container_name": container_name,
            "host_port": host_port,
            "extra_args": extra_args,
            "use_case": str(raw.get("use_case") or ""),
            "tags": [str(tag) for tag in raw.get("tags") or []],
        }
        return entry, warnings, errors

    def render_entry(self, entry: dict[str, Any]) -> str:
        """The exact YAML document `write_entry` would append (IMP-01.5).

        `ProposeRegistration` fills `ModelInfo.yaml_preview` from this so the
        admin can print the controller's YAML instead of re-implementing the
        serialisation (XR-3).
        """
        return _registration_document(entry)

    def write_entry(self, entry: dict[str, Any]) -> list[str]:
        """Append one validated entry to `models.yaml` atomically (FR-04.5).

        The entry is validated as a `ModelSpec` and against the current
        on-disk registry; nothing is written unless every check passes
        (NFR-04.4). The block is appended with a `vibe register` marker
        comment (NFR-04.3); the write is a temp file + `os.replace` under a
        process-local lock, and the merged text is re-validated before the
        replace (IMP-02.20).
        """
        try:
            spec = ModelSpec.model_validate(entry)
        except ValidationError as exc:
            return _format_validation_errors(exc, prefix="models[0]")

        with _WRITE_LOCK:
            path = self.path or self.default_path()
            try:
                current = Registry.load(path)
            except RegistryError as exc:
                return exc.errors
            errors: list[str] = []
            if spec.alias in current:
                errors.append(f"duplicate alias: {spec.alias}")
            if current.by_port(spec.host_port) is not None:
                errors.append(f"duplicate host_port: {spec.host_port}")
            if any(s.container_name == spec.container_name for s in current.all()):
                errors.append(f"duplicate container_name: {spec.container_name}")
            if errors:
                return errors

            try:
                original = path.read_text()
            except OSError as exc:
                return [f"cannot read {path}: {exc}"]
            if original and not original.endswith("\n"):
                original += "\n"
            merged = original + self.render_entry(entry)
            try:
                # IMP-02.20: re-validate the exact bytes that will land on
                # disk — a concurrent hand-edit can invalidate the merge
                # between the check above and the replace below.
                documents = yaml.safe_load_all(merged)
                merged_models = [
                    model
                    for document in documents
                    if isinstance(document, dict)
                    for model in document.get("models", [])
                ]
                Registry.from_dict({"models": merged_models})
            except RegistryError as exc:
                return exc.errors
            tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
            try:
                tmp.write_text(merged)
                # IMP-02.23: keep the file's mode across the replace and
                # fsync the temp before renaming so a crash cannot leave a
                # truncated models.yaml.
                shutil.copymode(path, tmp)
                with tmp.open("rb") as fh:
                    os.fsync(fh.fileno())
                os.replace(tmp, path)
            except OSError as exc:
                tmp.unlink(missing_ok=True)
                return [f"cannot write {path}: {exc}"]
        return []

    def _next_free_port(self, path: Path | None = None) -> int | None:
        """First unused port in `REGISTER_PORT_MIN`-`REGISTER_PORT_MAX`.

        With a `path`, the in-memory snapshot is refreshed from the on-disk
        registry first (IMP-02.24); a broken file degrades to the live
        snapshot so `propose` still answers.
        """
        used = {spec.host_port for spec in self._by_alias.values()}
        if path is not None:
            try:
                used |= {spec.host_port for spec in Registry.load(path)}
            except RegistryError:
                pass
        for port in range(REGISTER_PORT_MIN, REGISTER_PORT_MAX + 1):
            if port not in used:
                return port
        return None

    @property
    def aliases(self) -> list[str]:
        return list(self._by_alias)

    def __len__(self) -> int:
        return len(self._by_alias)

    def __contains__(self, alias: str) -> bool:
        return alias in self._by_alias

    def __iter__(self) -> Iterator[ModelSpec]:
        return iter(self._by_alias.values())


# -- registration helpers ----------------------------------------------------


def _format_validation_errors(exc: ValidationError, prefix: str) -> list[str]:
    """Pydantic errors in the `from_dict` message style (`models[i].field: msg`)."""
    errors = []
    for err in exc.errors():
        loc = ".".join(str(part) for part in err["loc"])
        errors.append(f"{prefix}{'.' + loc if loc else ''}: {err['msg']}")
    return errors


def _registration_document(entry: dict[str, Any]) -> str:
    """One appended YAML document with the `vibe register` marker (NFR-04.3)."""
    block = yaml.safe_dump([entry], sort_keys=False, default_flow_style=False)
    indented = textwrap.indent(block, "  ").strip("\n")
    date = datetime.now(UTC).date().isoformat()
    return f"\n---\n# --- added by vibe register {date} ---\nmodels:\n{indented}\n"


def _read_config_json(files: list[Path]) -> dict[str, Any] | None:
    """The snapshot's config.json as a dict, or None when absent/broken."""
    for path in files:
        if path.name != "config.json":
            continue
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError):
            return None
        return data if isinstance(data, dict) else None
    return None


def _heuristic_for_bytes(weights: int) -> tuple[float, int] | None:
    """`(gpu_memory_utilization, max_model_len cap)` per the FR-04.5 table."""
    gib = weights / _GIB
    if gib <= _HEURISTIC_SMALL_GIB:
        return (0.60, 32768)
    if gib <= _HEURISTIC_MID_GIB:
        return (0.72, 32768)
    if gib <= _HEURISTIC_MAX_GIB:
        return (0.85, 8192)
    return None  # too large for a 16 GB card


def _humanize_repo(hf_repo: str) -> str:
    """Title-ish `display_name` from the repo's last segment (FR-04.5)."""
    segment = hf_repo.rstrip("/").split("/")[-1]
    return " ".join(part for part in re.split(r"[-_]+", segment) if part)


def _is_local_path(repo: str) -> bool:
    """Whether `repo` is an absolute local path (Phase 5 round-trip).

    IMP-02.11: `~` paths are rejected by `propose` before this runs — vLLM
    cannot see an unexpanded tilde, and only absolute paths get a mount.
    """
    return repo.startswith("/")


def _to_int(value: Any) -> int | None:
    """`int(value)`; absent/empty means 0 (auto), unparseable means None."""
    if value is None or value == "":
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _to_float(value: Any) -> float | None:
    """`float(value)`; absent/empty means 0.0 (auto), unparseable means None."""
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

"""Docker adapter — the only module allowed to import `docker` (NFR-00.3, XR-5)."""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from pathlib import Path

import docker
import docker.errors
import docker.models.containers
import docker.types
import httpx

from vibe.config import Settings
from vibe.registry import ModelSpec

MANAGED_LABEL = "vibe.managed"   # label value "true" marks managed containers (ADR-3)
ALIAS_LABEL = "vibe.alias"       # label holds the registry alias
CONTAINER_PORT = 8000            # vLLM listens here inside the container
LEGACY_NAME_PREFIX = "vllm-"     # v0 containers: vllm-deepseek, vllm-llama, vllm-qwen
READINESS_POLL_SECONDS = 2.0     # NFR-00.5: poll interval <= 2 s

# Failure diagnostics (FR-00.11).
_LAST_LOGS_CHARS = 8000
_GATED_HINT = " — the model repo may be gated; check HF_TOKEN in .env"


# -- errors -------------------------------------------------------------


class DockerError(RuntimeError):
    """Base class for all docker_ctl failures."""


class DockerUnavailableError(DockerError):
    """Docker daemon unreachable."""


class ImagePullError(DockerError):
    """Image pull failed (network / auth / not found)."""


class ContainerExited(DockerError):
    """Container died before readiness; carries last log lines (FR-00.11)."""

    def __init__(self, message: str, last_logs: str = "") -> None:
        self.last_logs = last_logs
        super().__init__(message)


class ReadinessTimeout(DockerError):
    """Readiness deadline hit; carries last log lines (FR-00.11)."""

    def __init__(self, message: str, last_logs: str = "") -> None:
        self.last_logs = last_logs
        super().__init__(message)


def volumes_for(spec: ModelSpec, hf_home: Path) -> dict[str, dict[str, str]]:
    """Bind mounts for a model container (FR-00.9, M5.6).

    Always mounts the shared HF cache read-write at
    `/root/.cache/huggingface`. When `spec.hf_repo` is an absolute local
    path (vibe register allows absolute paths under `laboratory/models/`),
    that exact path is additionally mounted read-only at the same location
    inside the container so vLLM can read the exported model. Repo ids
    (`org/name`) and relative/`~` paths are treated as repo ids and get no
    extra mount.
    """
    volumes: dict[str, dict[str, str]] = {
        str(hf_home): {"bind": "/root/.cache/huggingface", "mode": "rw"}
    }
    if spec.hf_repo.startswith("/"):
        path = Path(spec.hf_repo).resolve()
        if not path.exists():
            raise DockerError(f"local model path not found: {path}")
        volumes[str(path)] = {"bind": str(path), "mode": "ro"}
    return volumes


# -- adapter ------------------------------------------------------------


class DockerCtl:
    """All Docker access for vibe — the single integration point (NFR-00.3)."""

    def __init__(
        self, settings: Settings, client: docker.DockerClient | None = None
    ) -> None:
        self.settings: Settings = settings
        self._client: docker.DockerClient | None = client

    @property
    def client(self) -> docker.DockerClient:
        """Lazy `docker.from_env()`; raises `DockerUnavailableError` if unreachable."""
        if self._client is None:
            try:
                self._client = docker.from_env()
            except docker.errors.DockerException as exc:
                raise DockerUnavailableError(
                    "cannot reach Docker daemon — is docker running?"
                ) from exc
        return self._client

    # -- discovery --------------------------------------------------------

    def find_managed(self) -> list[docker.models.containers.Container]:
        """Containers labelled `vibe.managed=true` (ADR-3)."""
        return self.client.containers.list(
            all=True, filters={"label": f"{MANAGED_LABEL}=true"}
        )

    def find_legacy(self) -> list[docker.models.containers.Container]:
        """v0 containers named `vllm-*` (vllm-deepseek / vllm-llama / vllm-qwen)."""
        return [
            container
            for container in self.client.containers.list(all=True)
            if (container.name or "").startswith(LEGACY_NAME_PREFIX)
        ]

    def purge_all(self) -> list[str]:
        """Force-remove every managed and legacy container; return names found.

        Race-tolerant: containers that vanish between listing and removal are
        ignored. Legacy containers are purged first (FR-00.5).
        """
        names: list[str] = []
        for container in [*self.find_legacy(), *self.find_managed()]:
            names.append(container.name or "<unnamed>")
            try:
                container.remove(force=True)
            except (docker.errors.NotFound, docker.errors.APIError):
                continue
        return names

    # -- lifecycle ----------------------------------------------------------

    def start(self, spec: ModelSpec) -> docker.models.containers.Container:
        """Run the model detached with loopback bind and labels (FR-00.9, ADR-3)."""
        self._ensure_image()
        cmd = self._build_cmd(spec)
        try:
            return self._run_detached(spec, cmd)
        except docker.errors.ImageNotFound:
            # Image vanished between ensure and run (rare race): pull once, retry.
            try:
                self.client.images.pull(self.settings.image)
            except docker.errors.DockerException as exc:
                raise ImagePullError(
                    f"failed to pull {self.settings.image}: {exc}"
                ) from exc
            try:
                return self._run_detached(spec, cmd)
            except docker.errors.ImageNotFound as exc:
                raise ImagePullError(
                    f"failed to pull {self.settings.image}: {exc}"
                ) from exc

    def _ensure_image(self) -> None:
        """Pull the pinned image if it is not cached locally."""
        try:
            self.client.images.get(self.settings.image)
            return
        except docker.errors.ImageNotFound:
            pass
        except docker.errors.DockerException as exc:
            raise DockerError(
                f"failed to inspect image {self.settings.image}: {exc}"
            ) from exc
        try:
            self.client.images.pull(self.settings.image)
        except docker.errors.DockerException as exc:
            raise ImagePullError(
                f"failed to pull {self.settings.image}: {exc}"
            ) from exc

    def _build_cmd(self, spec: ModelSpec) -> list[str]:
        """vLLM engine arguments derived from the registry entry."""
        cmd = ["--model", spec.hf_repo, "--served-model-name", spec.alias]
        if spec.quantization and spec.quantization != "none":
            cmd += ["--quantization", spec.quantization]
        cmd += ["--max-model-len", str(spec.max_model_len),
                "--gpu-memory-utilization", str(spec.gpu_memory_utilization)]
        cmd += list(spec.extra_args)
        return cmd

    def _run_detached(
        self, spec: ModelSpec, cmd: list[str]
    ) -> docker.models.containers.Container:
        try:
            return self.client.containers.run(
                image=self.settings.image,
                name=spec.container_name,
                detach=True,
                # Loopback bind only (XR-1).
                ports={f"{CONTAINER_PORT}/tcp": ("127.0.0.1", spec.host_port)},
                device_requests=[
                    docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
                ],
                ipc_mode="host",
                volumes=volumes_for(spec, self.settings.hf_home),
                environment=(
                    {"HF_TOKEN": self.settings.hf_token}
                    if self.settings.hf_token
                    else {}
                ),
                labels={MANAGED_LABEL: "true", ALIAS_LABEL: spec.alias},
                command=cmd,
            )
        except docker.errors.ImageNotFound:
            raise  # handled by start()
        except docker.errors.DockerException as exc:
            raise DockerError(
                f"failed to start container {spec.container_name}: {exc}"
            ) from exc

    def stop(self, container: docker.models.containers.Container) -> None:
        """Stop and remove the container; idempotent."""
        try:
            container.remove(force=True)
        except docker.errors.NotFound:
            pass

    def logs(
        self,
        container: docker.models.containers.Container,
        tail: int | None = None,
        follow: bool = False,
    ) -> Iterator[str]:
        """Yield timestamped log lines.

        Blocking: callers must wrap this in `asyncio.to_thread`; with
        `follow=True` the generator never terminates on its own.
        """
        stream = container.logs(
            stream=True,
            follow=follow,
            tail=tail if tail is not None else "all",
            timestamps=True,
        )
        for chunk in stream:
            if isinstance(chunk, bytes):
                yield chunk.decode("utf-8", errors="replace").rstrip("\n")
            else:
                yield str(chunk).rstrip("\n")

    def container_info(self, container: docker.models.containers.Container) -> dict:
        """Runtime snapshot of the container for `vibe status` (FR-00.3)."""
        attrs = container.attrs
        ports = attrs.get("NetworkSettings", {}).get("Ports", {})
        binding = ports.get(f"{CONTAINER_PORT}/tcp")
        host_port = int(binding[0]["HostPort"]) if binding else None
        labels = attrs.get("Config", {}).get("Labels") or {}
        return {
            "name": attrs["Name"].lstrip("/"),
            "id": attrs["Id"][:12],
            "status": attrs["State"]["Status"],
            "started_at": attrs["State"]["StartedAt"],
            "host_port": host_port,
            "image": attrs["Config"]["Image"],
            "alias": labels.get(ALIAS_LABEL),
        }

    def _last_logs(
        self, container: docker.models.containers.Container, n: int = 50
    ) -> str:
        """Tail of the container's logs for failure messages (FR-00.11)."""
        try:
            text = container.logs(tail=n, timestamps=True).decode("utf-8", "replace")
        except docker.errors.NotFound:
            return ""
        if len(text) > _LAST_LOGS_CHARS:
            return text[-_LAST_LOGS_CHARS:]
        return text

    # -- readiness --------------------------------------------------------

    async def wait_ready(self, spec: ModelSpec, timeout_s: int) -> None:
        """Poll `/health` then `/v1/models` until the alias is served (FR-00.10).

        Raises `ContainerExited` if the container dies and `ReadinessTimeout`
        at the deadline; both carry the last log lines (FR-00.11).
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{spec.host_port}", timeout=2.0
        ) as http:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break

                try:
                    container = self.client.containers.get(spec.container_name)
                except docker.errors.NotFound:
                    raise ContainerExited(
                        f"container {spec.container_name} exited before becoming ready"
                    ) from None

                if container.attrs["State"]["Status"] in ("exited", "dead"):
                    last_logs = self._last_logs(container)
                    message = (
                        f"container {spec.container_name} exited before becoming ready"
                    )
                    if "401" in last_logs or "gated" in last_logs:
                        message += _GATED_HINT
                    raise ContainerExited(message, last_logs=last_logs)

                try:
                    health = await http.get("/health")
                    if health.status_code == 200:
                        models = await http.get("/v1/models")
                        if models.status_code == 200:
                            data = models.json()
                            served = [m.get("id") for m in data.get("data", [])]
                            if spec.alias in served:
                                return
                except (httpx.HTTPError, ValueError):
                    pass  # server not up (or not serving JSON) yet — keep polling

                await asyncio.sleep(min(READINESS_POLL_SECONDS, max(remaining, 0.05)))

        try:
            container = self.client.containers.get(spec.container_name)
            last_logs = self._last_logs(container)
        except docker.errors.NotFound:
            last_logs = ""
        message = (
            f"{spec.alias} not ready within {timeout_s}s "
            f"at http://127.0.0.1:{spec.host_port}"
        )
        if "401" in last_logs or "gated" in last_logs:
            message += _GATED_HINT
        raise ReadinessTimeout(message, last_logs=last_logs)

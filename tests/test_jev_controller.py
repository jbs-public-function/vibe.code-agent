"""Jev controller tests (doc 06 §B): resolution, runner branches, state,
servicer RPCs. No docker/pynvml imports — duck-typed fakes throughout.
"""

from __future__ import annotations

import json
import types
from pathlib import Path
from typing import Any

import grpc
import pytest

from vibe.config import Settings
from vibe.controller.logbuffer import RingBuffer
from vibe.controller.servicer import ModelControllerServicer
from vibe.docker_ctl import ALIAS_LABEL, ReadinessTimeout
from vibe.gen.vibe.v1 import control_pb2
from vibe.gpu import GpuStats
from vibe.jev import JevError, SchemaStore, canonical_hash, load_builtin
from vibe.jev.control import JevLoadSpec, JevState, resolve_jev
from vibe.jobs import JobInProgress, JobRunner, JobState
from vibe.registry import ModelSpec, Registry, RegistryError

GOOD_INLINE = json.dumps(
    {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "InlineChoice",
        "type": "object",
        "properties": {
            "decision_type": {"const": "choice"},
            "result": {"type": "string", "enum": ["YES", "NO"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "required": ["decision_type", "result", "confidence"],
        "additionalProperties": False,
    }
)
GOOD_INLINE_SHA = canonical_hash(json.loads(GOOD_INLINE))

OK_PROBE = types.SimpleNamespace()


async def ok_probe(host_port: int, alias: str, schema: dict) -> None:
    """Stand-in for `probe_jev` that records the call and succeeds."""


def managed_container(
    name: str = "vibe-qwen2.5-coder-7b",
    alias: str = "qwen2.5-coder-7b",
) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        id="c" * 12, name=name, labels={ALIAS_LABEL: alias}
    )


def info_for(container: types.SimpleNamespace, alias: str) -> dict:
    return {
        "name": container.name,
        "id": container.id,
        "status": "running",
        "started_at": None,
        "host_port": 8030,
        "image": "vllm/vllm-openai:v0.11.2",
        "alias": alias,
    }


async def drain(agen) -> list:
    return [event async for event in agen]


def make_model(**overrides: Any) -> ModelSpec:
    data: dict[str, Any] = {
        "alias": "qwen2.5-coder-7b",
        "display_name": "Qwen2.5 Coder 7B",
        "hf_repo": "Qwen/Qwen2.5-Coder-7B-Instruct-AWQ",
        "quantization": "awq",
        "max_model_len": 32768,
        "gpu_memory_utilization": 0.7,
        "container_name": "vibe-qwen2.5-coder-7b",
        "host_port": 8030,
    }
    data.update(overrides)
    return ModelSpec.model_validate(data)


def install_probe(monkeypatch: pytest.MonkeyPatch, probe) -> None:
    monkeypatch.setattr("vibe.jobs.probe_jev", probe)


def purge_spy(monkeypatch: pytest.MonkeyPatch, docker) -> list:
    """Wrap `purge_all` to record invocations; returns the call list."""
    calls: list = []

    def spy() -> list[str]:
        calls.append(1)
        return []

    monkeypatch.setattr(docker, "purge_all", spy)
    return calls


async def must(coro):
    """Await a servicer coroutine; the generated pyi types them as Optional."""
    result = await coro
    assert result is not None
    return result


# -- B1: resolve_jev precedence -----------------------------------------------


def test_resolve_jev_precedence_matrix(tmp_path: Path) -> None:
    store = SchemaStore(tmp_path)
    store.put("named", GOOD_INLINE)
    store.put("default", GOOD_INLINE)  # user override of the builtin default
    model = make_model(jev_schema="named")

    # inline wins over everything, even with a schema_name present
    resolved = resolve_jev(
        JevLoadSpec(schema_name="named", schema_json=GOOD_INLINE), model, store
    )
    assert (resolved.source, resolved.sha256) == ("include-json", GOOD_INLINE_SHA)

    # inline without a name derives the name from the title
    resolved = resolve_jev(JevLoadSpec(schema_json=GOOD_INLINE), model, store)
    assert resolved.name == "InlineChoice"

    # schema_name resolves from the store
    resolved = resolve_jev(JevLoadSpec(schema_name="named"), model, store)
    assert (resolved.source, resolved.name) == ("store", "named")

    # unknown schema_name is a typed error
    with pytest.raises(JevError) as excinfo:
        resolve_jev(JevLoadSpec(schema_name="nope"), model, store)
    assert excinfo.value.code == "jev_schema_not_found"

    # ModelSpec.jev_schema is the fallback when no spec fields are set
    resolved = resolve_jev(JevLoadSpec(), model, store)
    assert (resolved.source, resolved.name) == ("model-default", "named")

    # otherwise: the (overridden) default
    model = make_model(jev_schema=None)
    resolved = resolve_jev(JevLoadSpec(), model, store)
    assert (resolved.source, resolved.name) == ("default", "default")
    assert resolved.sha256 == GOOD_INLINE_SHA  # the override, not the builtin

    # with no override, the builtin default
    store2 = SchemaStore(tmp_path / "fresh")
    resolved = resolve_jev(JevLoadSpec(), model, store2)
    assert (resolved.source, resolved.name) == ("default", "default")
    assert resolved.schema == load_builtin("default")

    # invalid inline text is a typed error
    with pytest.raises(JevError) as excinfo:
        resolve_jev(JevLoadSpec(schema_json="{bad"), model, store)
    assert excinfo.value.code == "jev_schema_invalid"

    # invalid save_as name fails before anything else
    with pytest.raises(JevError) as excinfo:
        resolve_jev(
            JevLoadSpec(schema_json=GOOD_INLINE, save_as="BAD NAME"), model, store
        )
    assert excinfo.value.code == "jev_schema_invalid"


# -- B2–B8: runner branches ---------------------------------------------------


async def test_fresh_load_jev_on_attaches(
    settings: Settings, registry: Registry, fake_docker, monkeypatch: pytest.MonkeyPatch
):
    probes: list = []

    async def recording_probe(host_port, alias, schema):
        probes.append((host_port, alias, schema))

    install_probe(monkeypatch, recording_probe)
    runner = JobRunner(settings, registry, docker=fake_docker)
    events = await drain(runner.load("qwen2.5-coder-7b", jev_mode="on"))

    messages = [event.message for event in events]
    assert any("jev: probing" in message for message in messages)
    assert any("jev: attached 'default'" in message for message in messages)
    assert events[-1].state == JobState.READY
    assert "READY with Jev 'default'" in events[-1].message
    assert events[-1].error_code in (None, "")

    active = runner.jev_state.current()
    assert active is not None
    assert active.alias == "qwen2.5-coder-7b"
    assert active.name == "default"
    assert active.sha256 == canonical_hash(load_builtin("default"))
    assert len(probes) == 1
    assert probes[0][0] == 8030
    # the probe receives the stored form; `probe_jev` strips `$schema`
    # for the wire payload (covered by the J0 wire_schema tests).
    assert probes[0][2] == load_builtin("default")
    # state is persisted
    assert (settings.jev_dir / ".active.json").is_file()


async def test_fast_path_attach_no_docker_touch(
    settings: Settings, registry: Registry, fake_docker, monkeypatch: pytest.MonkeyPatch
):
    container = managed_container()
    fake_docker.managed = [container]
    fake_docker.info_by_name[container.name] = info_for(container, "qwen2.5-coder-7b")
    purges = purge_spy(monkeypatch, fake_docker)
    install_probe(monkeypatch, ok_probe)
    runner = JobRunner(settings, registry, docker=fake_docker)

    events = await drain(
        runner.load(
            "qwen2.5-coder-7b",
            jev_mode="on",
            jev=JevLoadSpec(schema_name="default"),
        )
    )
    assert purges == []  # the fast path never purges
    assert fake_docker.started == []  # and never restarts the container
    assert any("jev: attached 'default'" in event.message for event in events)

    # same hash again → "already attached", no re-set
    active = runner.jev_state.current()
    assert active is not None
    applied_at = active.applied_at_unix_ms
    events = await drain(
        runner.load(
            "qwen2.5-coder-7b",
            jev_mode="on",
            jev=JevLoadSpec(schema_name="default"),
        )
    )
    assert any("already attached" in event.message for event in events)
    active = runner.jev_state.current()
    assert active is not None
    assert active.applied_at_unix_ms == applied_at

    # a different schema replaces it (hash updates)
    events = await drain(
        runner.load(
            "qwen2.5-coder-7b",
            jev_mode="on",
            jev=JevLoadSpec(schema_name="yes-no"),
        )
    )
    assert any("jev: attached 'yes-no'" in event.message for event in events)
    active = runner.jev_state.current()
    assert active is not None
    assert active.name == "yes-no"


async def test_jev_off_detaches_without_docker(
    settings: Settings, registry: Registry, fake_docker, monkeypatch: pytest.MonkeyPatch
):
    container = managed_container()
    fake_docker.managed = [container]
    fake_docker.info_by_name[container.name] = info_for(container, "qwen2.5-coder-7b")
    purges = purge_spy(monkeypatch, fake_docker)
    install_probe(monkeypatch, ok_probe)
    runner = JobRunner(settings, registry, docker=fake_docker)

    await drain(runner.load("qwen2.5-coder-7b", jev_mode="on"))
    assert runner.jev_state.current() is not None

    events = await drain(runner.load("qwen2.5-coder-7b", jev_mode="off"))
    assert purges == [] and fake_docker.started == []
    assert any("jev: detached 'default'" in event.message for event in events)
    assert events[-1].state == JobState.READY
    assert runner.jev_state.current() is None
    assert not (settings.jev_dir / ".active.json").exists()

    # detaching again is idempotent: an info event, not an error
    events = await drain(runner.load("qwen2.5-coder-7b", jev_mode="off"))
    assert any("not active" in event.message for event in events)


async def test_unload_and_alias_change_clear_jev(
    settings: Settings, registry: Registry, fake_docker, monkeypatch: pytest.MonkeyPatch
):
    install_probe(monkeypatch, ok_probe)
    runner = JobRunner(settings, registry, docker=fake_docker)
    await drain(runner.load("qwen2.5-coder-7b", jev_mode="on"))
    assert runner.jev_state.current() is not None

    # a load of a different alias purges and clears Jev
    await drain(runner.load("llama-3.1-8b"))
    assert runner.jev_state.current() is None

    # unload clears too
    await drain(runner.load("qwen2.5-coder-7b", jev_mode="on"))
    assert runner.jev_state.current() is not None
    await drain(runner.unload())
    assert runner.jev_state.current() is None


async def test_failed_fresh_load_leaves_jev_cleared(
    settings: Settings, registry: Registry, fake_docker, monkeypatch: pytest.MonkeyPatch
):
    install_probe(monkeypatch, ok_probe)
    runner = JobRunner(settings, registry, docker=fake_docker)
    await drain(runner.load("qwen2.5-coder-7b", jev_mode="on"))
    assert runner.jev_state.current() is not None

    fake_docker.wait_ready_error = ReadinessTimeout("timed out", "")
    with pytest.raises(ReadinessTimeout):
        await drain(runner.load("qwen2.5-coder-7b", force=True, jev_mode="on"))
    assert runner.jev_state.current() is None


async def test_invalid_inline_schema_no_purge(
    settings: Settings, registry: Registry, fake_docker, monkeypatch: pytest.MonkeyPatch
):
    purges = purge_spy(monkeypatch, fake_docker)
    runner = JobRunner(settings, registry, docker=fake_docker)

    events = await drain(
        runner.load(
            "qwen2.5-coder-7b",
            jev_mode="on",
            jev=JevLoadSpec(schema_json="{not json"),
        )
    )
    assert purges == []  # JEV-02.6: no purge occurred
    assert fake_docker.started == []
    assert events[-1].state == JobState.ERROR  # nothing loaded → ERROR
    assert events[-1].level == "error"
    assert events[-1].error_code == "jev_schema_invalid"
    assert runner.jev_state.current() is None


async def test_probe_failure_keeps_model_ready(
    settings: Settings, registry: Registry, fake_docker, monkeypatch: pytest.MonkeyPatch
):
    async def failing_probe(host_port, alias, schema):
        raise JevError("jev_probe_failed", "vLLM rejected the schema (HTTP 400)")

    install_probe(monkeypatch, failing_probe)
    runner = JobRunner(settings, registry, docker=fake_docker)

    events = await drain(runner.load("qwen2.5-coder-7b", jev_mode="on"))
    # the model was started and is READY; Jev is not
    assert len(fake_docker.started) == 1
    assert events[-1].state == JobState.READY
    assert events[-1].level == "error"
    assert events[-1].error_code == "jev_probe_failed"
    assert runner.jev_state.current() is None


async def test_fast_path_probe_failure_detaches_previous(
    settings: Settings, registry: Registry, fake_docker, monkeypatch: pytest.MonkeyPatch
):
    """JEV-02.15: a backend-rejected schema on the fast path leaves the
    model in plain mode — the previous attachment is detached too."""
    container = managed_container()
    fake_docker.managed = [container]
    fake_docker.info_by_name[container.name] = info_for(container, "qwen2.5-coder-7b")
    purges = purge_spy(monkeypatch, fake_docker)
    install_probe(monkeypatch, ok_probe)
    runner = JobRunner(settings, registry, docker=fake_docker)

    # attach 'default' first (fast path)
    await drain(
        runner.load(
            "qwen2.5-coder-7b",
            jev_mode="on",
            jev=JevLoadSpec(schema_name="default"),
        )
    )
    assert runner.jev_state.current() is not None

    # a different schema that the backend rejects must clear the attachment
    async def failing_probe(host_port, alias, schema):
        raise JevError("jev_probe_failed", "vLLM rejected the schema (HTTP 400)")

    install_probe(monkeypatch, failing_probe)
    events = await drain(
        runner.load(
            "qwen2.5-coder-7b",
            jev_mode="on",
            jev=JevLoadSpec(schema_name="yes-no"),
        )
    )
    assert purges == []  # fast path: the container is never touched
    assert fake_docker.started == []
    assert events[-1].state == JobState.READY
    assert events[-1].level == "error"
    assert events[-1].error_code == "jev_probe_failed"
    assert runner.jev_state.current() is None


async def test_save_as_persists_only_after_successful_probe(
    settings: Settings, registry: Registry, fake_docker, monkeypatch: pytest.MonkeyPatch
):
    async def failing_probe(host_port, alias, schema):
        raise JevError("jev_probe_failed", "nope")

    install_probe(monkeypatch, failing_probe)
    runner = JobRunner(settings, registry, docker=fake_docker)
    await drain(
        runner.load(
            "qwen2.5-coder-7b",
            jev_mode="on",
            jev=JevLoadSpec(schema_json=GOOD_INLINE, save_as="mine"),
        )
    )
    assert not (settings.jev_dir / "mine.json").exists()  # failed probe → no save

    install_probe(monkeypatch, ok_probe)
    events = await drain(
        runner.load(
            "qwen2.5-coder-7b",
            jev_mode="on",
            jev=JevLoadSpec(schema_json=GOOD_INLINE, save_as="mine"),
        )
    )
    assert any("saved inline schema as 'mine'" in event.message for event in events)
    assert (settings.jev_dir / "mine.json").is_file()
    stored = runner.jev_store.get("mine")
    assert stored.source == "user"
    assert stored.sha256 == GOOD_INLINE_SHA


async def test_jev_false_model_opt_out(
    settings: Settings, registry: Registry, fake_docker
):
    registry2 = Registry.from_dict(
        {"models": [{**make_model().model_dump(), "jev": False}]}
    )
    runner = JobRunner(settings, registry2, docker=fake_docker)
    events = await drain(runner.load("qwen2.5-coder-7b", jev_mode="on"))
    assert events[-1].error_code == "jev_unsupported_model"
    assert fake_docker.started == []


async def test_reasoning_tag_warns(
    settings: Settings, registry: Registry, fake_docker, monkeypatch: pytest.MonkeyPatch
):
    install_probe(monkeypatch, ok_probe)
    registry2 = Registry.from_dict(
        {"models": [{**make_model().model_dump(), "tags": ["reasoning"]}]}
    )
    runner = JobRunner(settings, registry2, docker=fake_docker)
    events = await drain(runner.load("qwen2.5-coder-7b", jev_mode="on"))
    warns = [event for event in events if event.level == "warn"]
    assert any("<think>" in event.message for event in warns)


async def test_attach_respects_single_job_guard(
    settings: Settings, registry: Registry, fake_docker, monkeypatch: pytest.MonkeyPatch
):
    install_probe(monkeypatch, ok_probe)
    runner = JobRunner(settings, registry, docker=fake_docker)
    gen = runner.load("qwen2.5-coder-7b", jev_mode="on")
    await anext(gen)  # job slot taken
    with pytest.raises(JobInProgress):
        await anext(runner.load("qwen2.5-coder-7b", jev_mode="on"))
    await drain(gen)


# -- B9: JevState persistence/reconcile ---------------------------------------


def test_jev_state_persistence_round_trip(tmp_path: Path) -> None:
    path = tmp_path / ".active.json"
    state = JevState(path)
    state.load()
    assert state.current() is None

    state.set("qwen2.5-coder-7b", "default", "default", load_builtin("default"))
    reloaded = JevState(path)
    assert reloaded.load() == []
    active = reloaded.current()
    assert active is not None
    assert (active.alias, active.name, active.source) == (
        "qwen2.5-coder-7b",
        "default",
        "default",
    )
    assert active.sha256 == canonical_hash(load_builtin("default"))


def test_jev_state_reconcile(tmp_path: Path) -> None:
    path = tmp_path / ".active.json"
    state = JevState(path)
    state.set("qwen2.5-coder-7b", "default", "default", load_builtin("default"))

    assert state.reconcile({"qwen2.5-coder-7b", "llama-3.1-8b"}) is False
    assert state.current() is not None
    assert state.reconcile(set()) is True  # container gone → dropped
    assert state.current() is None
    assert not path.exists()

    state.set("qwen2.5-coder-7b", "default", "default", load_builtin("default"))
    assert state.reconcile({"llama-3.1-8b"}) is True  # alias mismatch → dropped
    assert state.current() is None


def test_jev_state_corrupt_file_dropped(tmp_path: Path) -> None:
    path = tmp_path / ".active.json"
    path.write_text("{not json", encoding="utf-8")
    state = JevState(path)
    warnings = state.load()
    assert state.current() is None
    assert any("corrupt" in warning for warning in warnings)


# -- B10/B12: servicer-level (real runner, fake docker, direct calls) ---------


class FakeContext:
    """Minimal grpc context stand-in for direct servicer calls."""

    def __init__(self) -> None:
        self.aborted: tuple[grpc.StatusCode, str] | None = None

    async def abort(self, code, details):
        self.aborted = (code, details)
        raise grpc.aio.AbortError(code, details)


def make_servicer(
    settings: Settings,
    registry: Registry,
    docker,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[ModelControllerServicer, JobRunner]:
    install_probe(monkeypatch, ok_probe)
    runner = JobRunner(settings, registry, docker=docker)
    servicer = ModelControllerServicer(
        settings,
        registry,
        runner,
        docker,
        lambda: GpuStats(0, 0, 0, ()),
        RingBuffer(),
    )
    return servicer, runner


async def test_get_status_reports_jev_and_stale(
    settings: Settings, registry: Registry, fake_docker, monkeypatch: pytest.MonkeyPatch
):
    container = managed_container()
    fake_docker.managed = [container]
    fake_docker.info_by_name[container.name] = info_for(container, "qwen2.5-coder-7b")
    servicer, runner = make_servicer(settings, registry, fake_docker, monkeypatch)

    # Attach via the fast path (the container is already READY).
    await drain(
        runner.load(
            "qwen2.5-coder-7b",
            jev_mode="on",
            jev=JevLoadSpec(schema_name="default"),
        )
    )

    status = await servicer.GetStatus(control_pb2.GetStatusRequest(), FakeContext())
    assert status.HasField("jev")
    assert status.jev.schema_name == "default"
    # an explicit `--schema default` resolves from the store (J4)
    assert status.jev.source == "store"
    assert status.jev.stale is False
    assert status.jev.sha256 == canonical_hash(load_builtin("default"))

    # editing the named store schema marks it stale
    runner.jev_store.put("default", GOOD_INLINE)
    status = await servicer.GetStatus(control_pb2.GetStatusRequest(), FakeContext())
    assert status.jev.stale is True

    # detaching clears Status.jev
    await drain(runner.load("qwen2.5-coder-7b", jev_mode="off"))
    status = await servicer.GetStatus(control_pb2.GetStatusRequest(), FakeContext())
    assert not status.HasField("jev")


async def test_get_active_jev_body(
    settings: Settings, registry: Registry, fake_docker, monkeypatch: pytest.MonkeyPatch
):
    container = managed_container()
    fake_docker.managed = [container]
    fake_docker.info_by_name[container.name] = info_for(container, "qwen2.5-coder-7b")
    servicer, runner = make_servicer(settings, registry, fake_docker, monkeypatch)
    await drain(
        runner.load(
            "qwen2.5-coder-7b",
            jev_mode="on",
            jev=JevLoadSpec(schema_name="default"),
        )
    )
    active = await servicer.GetActiveJev(
        control_pb2.GetActiveJevRequest(), FakeContext()
    )
    assert active.HasField("info")
    assert json.loads(active.schema_json) == load_builtin("default")

    await drain(runner.load("qwen2.5-coder-7b", jev_mode="off"))
    active = await servicer.GetActiveJev(
        control_pb2.GetActiveJevRequest(), FakeContext()
    )
    assert not active.HasField("info")


async def test_schema_rpcs_validation_as_data_and_status_codes(
    settings: Settings, registry: Registry, fake_docker, monkeypatch: pytest.MonkeyPatch
):
    servicer, _ = make_servicer(settings, registry, fake_docker, monkeypatch)

    # validation failures are data, not gRPC errors
    response = await must(
        servicer.PutJevSchema(
            control_pb2.PutJevSchemaRequest(name="bad", schema_json="{nope"),
            FakeContext(),
        )
    )
    assert response.validation.ok is False
    assert response.error == ""

    response = await must(
        servicer.PutJevSchema(
            control_pb2.PutJevSchemaRequest(name="good", schema_json=GOOD_INLINE),
            FakeContext(),
        )
    )
    assert response.validation.ok is True
    assert response.validation.sha256 == GOOD_INLINE_SHA

    got = await must(
        servicer.GetJevSchema(
            control_pb2.GetJevSchemaRequest(name="good"), FakeContext()
        )
    )
    assert got.name == "good" and got.source == "user"
    assert json.loads(got.schema_json) == json.loads(GOOD_INLINE)

    # unknown name → NOT_FOUND
    context = FakeContext()
    with pytest.raises(grpc.aio.AbortError):
        await servicer.GetJevSchema(
            control_pb2.GetJevSchemaRequest(name="nope"), context
        )
    assert context.aborted is not None
    assert context.aborted[0] == grpc.StatusCode.NOT_FOUND

    # deleting a built-in → FAILED_PRECONDITION
    context = FakeContext()
    with pytest.raises(grpc.aio.AbortError):
        await servicer.DeleteJevSchema(
            control_pb2.DeleteJevSchemaRequest(name="yes-no"), context
        )
    assert context.aborted is not None
    assert context.aborted[0] == grpc.StatusCode.FAILED_PRECONDITION

    # deleting an override reports the revert
    response = await must(
        servicer.PutJevSchema(
            control_pb2.PutJevSchemaRequest(name="default", schema_json=GOOD_INLINE),
            FakeContext(),
        )
    )
    assert response.validation.ok is True
    deleted = await must(
        servicer.DeleteJevSchema(
            control_pb2.DeleteJevSchemaRequest(name="default"), FakeContext()
        )
    )
    assert deleted.ok is True and deleted.reverted_to == "builtin"

    # dry-run validation
    validated = await must(
        servicer.ValidateJevSchema(
            control_pb2.ValidateJevSchemaRequest(schema_json=GOOD_INLINE),
            FakeContext(),
        )
    )
    assert validated.validation.ok is True

    # list shows every visible schema
    listing = await must(
        servicer.ListJevSchemas(
            control_pb2.ListJevSchemasRequest(), FakeContext()
        )
    )
    names = {entry.name for entry in listing.schemas}
    assert {"default", "yes-no", "yes-no-unsure", "good"} <= names


async def test_status_jev_populated_end_to_end_after_fresh_load(
    settings: Settings, registry: Registry, fake_docker, monkeypatch: pytest.MonkeyPatch
):
    container = managed_container()
    fake_docker.managed = [container]
    fake_docker.info_by_name[container.name] = info_for(container, "qwen2.5-coder-7b")
    servicer, runner = make_servicer(settings, registry, fake_docker, monkeypatch)

    await drain(
        runner.load(
            "qwen2.5-coder-7b",
            jev_mode="on",
            jev=JevLoadSpec(schema_name="yes-no"),
        )
    )
    status = await servicer.GetStatus(control_pb2.GetStatusRequest(), FakeContext())
    assert status.HasField("jev")
    assert status.jev.schema_name == "yes-no"
    assert status.jev.title == "YesNo"


# -- B13: registry parsing ----------------------------------------------------


def test_registry_jev_fields_parsing() -> None:
    data = make_model().model_dump()
    data["jev_schema"] = "yes-no"
    data["jev"] = True
    spec = ModelSpec.model_validate(data)
    assert spec.jev_schema == "yes-no"
    assert spec.jev is True

    # a missing named schema is NOT a config error (JEV-02.9): store
    # contents change at runtime, so only the name format is checked.
    registry = Registry.from_dict(
        {"models": [{**data, "jev_schema": "missing-schema"}]}
    )
    assert registry.get("qwen2.5-coder-7b").jev_schema == "missing-schema"

    # an invalid name format IS a config error
    with pytest.raises(RegistryError):
        Registry.from_dict({"models": [{**data, "jev_schema": "BAD NAME"}]})


def test_registry_round_trip_keeps_jev_fields() -> None:
    data = make_model().model_dump()
    data["jev_schema"] = "yes-no"
    registry = Registry.from_dict({"models": [data]})
    assert registry.to_dict()["models"][0]["jev_schema"] == "yes-no"


# -- B12: gRPC message-size bound ---------------------------------------------


async def test_oversized_schema_rejected_at_grpc_layer(
    settings: Settings, registry: Registry, fake_docker, monkeypatch: pytest.MonkeyPatch
):
    """JEV-02.8: > 256 KiB messages are rejected with RESOURCE_EXHAUSTED."""
    import socket

    from vibe.controller.server import build_server
    from vibe.gen.vibe.v1 import control_pb2_grpc

    install_probe(monkeypatch, ok_probe)
    runner = JobRunner(settings, registry, docker=fake_docker)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = await build_server(
        settings,
        registry,
        runner=runner,
        docker=fake_docker,
        gpu_fn=lambda: GpuStats(0, 0, 0, ()),
        buffer=RingBuffer(),
        port=port,
    )
    await server.start()
    channel = grpc.aio.insecure_channel(
        f"127.0.0.1:{port}", options=(("grpc.enable_http_proxy", 0),)
    )
    stub = control_pb2_grpc.ModelControllerStub(channel)
    try:
        big = json.dumps({"x": "y" * (300 * 1024)})
        with pytest.raises(grpc.aio.AioRpcError) as excinfo:
            await stub.ValidateJevSchema(
                control_pb2.ValidateJevSchemaRequest(schema_json=big)
            )
        assert excinfo.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
    finally:
        await channel.close()
        await server.stop(grace=0)

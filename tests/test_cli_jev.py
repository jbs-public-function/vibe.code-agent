"""CLI tests for the Jev commands (improvements-2 doc 04, D1–D8).

Same pattern as `tests/test_cli.py`: a recording fake `ControllerClient` is
monkeypatched into `vibe.cli.ControllerClient` and the Typer app is invoked
through `CliRunner`. This file is self-contained so the two modules stay
disjoint.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

import pytest
import typer.testing
from typer.testing import CliRunner

import vibe.cli as cli
from vibe.cli import app
from vibe.client import ControllerUnavailable, JevReadOnly, JevSchemaNotFound
from vibe.gen.vibe.v1 import control_pb2
from vibe.jev import MAX_SCHEMA_BYTES

VALID_SCHEMA = json.dumps(
    {
        "title": "yes-no",
        "type": "object",
        "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
        "required": ["answer"],
    }
)


class FakeClient:
    """Recording in-memory stand-in for the Jev part of the client contract."""

    def __init__(self) -> None:
        self.target = "test:50051"
        self.calls: list[str] = []
        self.closed = False
        self.load_events: list[control_pb2.JobEvent] = []
        self.load_error: Exception | None = None
        self.load_calls: list[tuple[str, int, control_pb2.JevSpec | None]] = []
        self.status_value = control_pb2.Status()
        self.status_error: Exception | None = None
        self.active_value = control_pb2.ActiveJev()
        self.active_error: Exception | None = None
        self.schemas: list[control_pb2.JevSchemaEntry] = []
        self.list_error: Exception | None = None
        self.schema_value = control_pb2.GetJevSchemaResponse()
        self.get_error: Exception | None = None
        self.get_calls: list[str] = []
        self.put_value = control_pb2.PutJevSchemaResponse(
            validation=control_pb2.JevValidation(ok=True)
        )
        self.put_error: Exception | None = None
        self.put_calls: list[tuple[str, str]] = []
        self.validate_value = control_pb2.JevValidation(ok=True)
        self.validate_error: Exception | None = None
        self.validate_calls: list[str] = []
        self.delete_value = control_pb2.DeleteJevSchemaResponse(ok=True)
        self.delete_error: Exception | None = None
        self.delete_calls: list[str] = []

    async def close(self) -> None:
        self.closed = True

    async def load(
        self,
        alias: str,
        force: bool = False,
        keep_failed: bool = False,
        timeout_s: int | None = None,
        jev_mode: int = control_pb2.JEV_MODE_UNSPECIFIED,
        jev: control_pb2.JevSpec | None = None,
    ) -> AsyncIterator[control_pb2.JobEvent]:
        self.calls.append("load")
        self.load_calls.append((alias, jev_mode, jev))
        if self.load_error is not None:
            raise self.load_error
        for event in self.load_events:
            yield event

    async def status(self) -> control_pb2.Status:
        self.calls.append("status")
        if self.status_error is not None:
            raise self.status_error
        return self.status_value

    async def gpu(self) -> control_pb2.GpuStats:
        self.calls.append("gpu")
        return control_pb2.GpuStats()

    async def get_active_jev(self) -> control_pb2.ActiveJev:
        self.calls.append("get_active_jev")
        if self.active_error is not None:
            raise self.active_error
        return self.active_value

    async def list_jev_schemas(self) -> list[control_pb2.JevSchemaEntry]:
        self.calls.append("list_jev_schemas")
        if self.list_error is not None:
            raise self.list_error
        return list(self.schemas)

    async def get_jev_schema(self, name: str) -> control_pb2.GetJevSchemaResponse:
        self.calls.append("get_jev_schema")
        self.get_calls.append(name)
        if self.get_error is not None:
            raise self.get_error
        return self.schema_value

    async def put_jev_schema(
        self, name: str, schema_json: str
    ) -> control_pb2.PutJevSchemaResponse:
        self.calls.append("put_jev_schema")
        self.put_calls.append((name, schema_json))
        if self.put_error is not None:
            raise self.put_error
        return self.put_value

    async def delete_jev_schema(
        self, name: str
    ) -> control_pb2.DeleteJevSchemaResponse:
        self.calls.append("delete_jev_schema")
        self.delete_calls.append(name)
        if self.delete_error is not None:
            raise self.delete_error
        return self.delete_value

    async def validate_jev_schema(self, schema_json: str) -> control_pb2.JevValidation:
        self.calls.append("validate_jev_schema")
        self.validate_calls.append(schema_json)
        if self.validate_error is not None:
            raise self.validate_error
        return self.validate_value


@pytest.fixture
def fake_client() -> FakeClient:
    """A fresh recording fake for one command invocation."""
    return FakeClient()


@pytest.fixture
def runner(fake_client: FakeClient, monkeypatch: pytest.MonkeyPatch) -> CliRunner:
    """`CliRunner` wired to the fake via the `vibe.cli.ControllerClient` symbol."""
    monkeypatch.setenv("VLLM_IMAGE_TAG", "v0.11.2")  # Settings() validation
    monkeypatch.setattr(cli, "ControllerClient", lambda settings: fake_client)
    return CliRunner()


def _ready(message: str = "ready") -> control_pb2.JobEvent:
    return control_pb2.JobEvent(state=control_pb2.READY, message=message)


# -- D1: option conflicts ----------------------------------------------------


def test_d1_option_conflicts_exit_two(runner: CliRunner, fake_client: FakeClient) -> None:
    """--include-json+--schema, --no-jev+--jev, --save-as alone ⇒ exit 2, no RPC."""
    cases = [
        ["load", "x", "--include-json", "schema.json", "--schema", "yes-no"],
        ["load", "x", "--jev", "--no-jev"],
        ["load", "x", "--no-jev", "--include-json", "schema.json"],
        ["load", "x", "--no-jev", "--schema", "yes-no"],
        ["load", "x", "--no-jev", "--save-as", "mine"],
        ["load", "x", "--save-as", "mine"],
    ]
    for args in cases:
        result = runner.invoke(app, args)
        assert result.exit_code == 2, args
        assert "error:" in result.stderr, args
    assert fake_client.load_calls == []


# -- D2: --include-json file handling ----------------------------------------


def test_d2_include_json_reads_file_and_sends_inline(
    runner: CliRunner, fake_client: FakeClient, tmp_path
) -> None:
    schema_file = tmp_path / "schema.json"
    schema_file.write_text(VALID_SCHEMA, encoding="utf-8")
    fake_client.load_events = [_ready()]
    result = runner.invoke(app, ["load", "x", "--include-json", str(schema_file)])
    assert result.exit_code == 0
    alias, jev_mode, jev = fake_client.load_calls[0]
    assert alias == "x"
    assert jev_mode == control_pb2.JEV_ON
    assert jev is not None
    assert jev.schema_json == VALID_SCHEMA
    assert jev.schema_name == "" and jev.save_as == ""


def test_d2_include_json_missing_file_exit_two_no_rpc(
    runner: CliRunner, fake_client: FakeClient
) -> None:
    result = runner.invoke(app, ["load", "x", "--include-json", "/nonexistent/schema.json"])
    assert result.exit_code == 2
    assert "/nonexistent/schema.json" in result.stderr
    assert fake_client.load_calls == []


def test_d2_include_json_too_large_exit_two_no_rpc(
    runner: CliRunner, fake_client: FakeClient, tmp_path
) -> None:
    schema_file = tmp_path / "big.json"
    schema_file.write_text('{"pad": "' + "a" * (MAX_SCHEMA_BYTES + 1) + '"}')
    result = runner.invoke(app, ["load", "x", "--include-json", str(schema_file)])
    assert result.exit_code == 2
    assert "maximum is" in result.stderr
    assert fake_client.load_calls == []


def test_d2_include_json_bad_json_exit_two_no_rpc(
    runner: CliRunner, fake_client: FakeClient, tmp_path
) -> None:
    schema_file = tmp_path / "bad.json"
    schema_file.write_text("{not json")
    result = runner.invoke(app, ["load", "x", "--include-json", str(schema_file)])
    assert result.exit_code == 2
    assert "not valid JSON" in result.stderr
    assert "line 1" in result.stderr
    assert fake_client.load_calls == []


def test_d2_include_json_warnings_printed_but_loads(
    runner: CliRunner, fake_client: FakeClient, tmp_path
) -> None:
    schema_file = tmp_path / "loose.json"
    schema_file.write_text(json.dumps({"type": "object"}))
    fake_client.load_events = [_ready()]
    result = runner.invoke(app, ["load", "x", "--include-json", str(schema_file)])
    assert result.exit_code == 0
    assert "warning:" in result.stderr
    assert len(fake_client.load_calls) == 1


# -- D3: --include-json - (stdin) --------------------------------------------


def test_d3_include_json_stdin_reads_piped_schema(
    runner: CliRunner, fake_client: FakeClient
) -> None:
    fake_client.load_events = [_ready()]
    result = runner.invoke(app, ["load", "x", "--include-json", "-"], input=VALID_SCHEMA)
    assert result.exit_code == 0
    _, jev_mode, jev = fake_client.load_calls[0]
    assert jev_mode == control_pb2.JEV_ON
    assert jev is not None and jev.schema_json == VALID_SCHEMA


def test_d3_include_json_stdin_tty_rejected(
    runner: CliRunner,
    fake_client: FakeClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # CliRunner's `input=` only takes str/bytes, so make the harness's stdin
    # wrapper report itself as a TTY — the CLI must refuse before reading.
    monkeypatch.setattr(
        typer.testing._NamedTextIOWrapper, "isatty", lambda self: True
    )
    result = runner.invoke(app, ["load", "x", "--include-json", "-"], input="{}")
    assert result.exit_code == 2
    assert "TTY" in result.stderr
    assert fake_client.load_calls == []


# -- D4: exit codes from terminal event error_code ---------------------------


@pytest.mark.parametrize(
    ("error_code", "state", "expected_exit"),
    [
        ("jev_schema_invalid", control_pb2.READY, 2),
        ("jev_schema_not_found", control_pb2.ERROR, 3),
        ("jev_probe_failed", control_pb2.READY, 1),
        ("jev_unsupported_model", control_pb2.READY, 1),
        ("jev_something_else", control_pb2.READY, 1),
    ],
)
def test_d4_load_jev_error_codes(
    runner: CliRunner,
    fake_client: FakeClient,
    error_code: str,
    state: control_pb2.ModelState,
    expected_exit: int,
) -> None:
    fake_client.load_events = [
        control_pb2.JobEvent(
            state=state, message="jev: nope", level="error", error_code=error_code
        )
    ]
    result = runner.invoke(app, ["load", "x", "--jev"])
    assert result.exit_code == expected_exit
    assert f"[{error_code}]" in result.stderr


# -- D5: plain load sends JEV_MODE_UNSPECIFIED -------------------------------


def test_d5_plain_load_sends_unspecified(runner: CliRunner, fake_client: FakeClient) -> None:
    fake_client.load_events = [_ready()]
    result = runner.invoke(app, ["load", "x"])
    assert result.exit_code == 0
    assert fake_client.load_calls == [("x", control_pb2.JEV_MODE_UNSPECIFIED, None)]


# -- D6: status --json gains the jev key -------------------------------------


def test_d6_status_json_jev_inactive_null(runner: CliRunner, fake_client: FakeClient) -> None:
    result = runner.invoke(app, ["status", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["jev"] is None


def test_d6_status_json_jev_populated_snake_case_ints(
    runner: CliRunner, fake_client: FakeClient
) -> None:
    fake_client.status_value = control_pb2.Status(
        active_alias="qwen2.5-coder-7b",
        state=control_pb2.READY,
        jev=control_pb2.JevInfo(
            schema_name="yes-no",
            sha256="abcdef012345",
            source="store",
            applied_at_unix_ms=1726000000123,
            stale=True,
            title="Yes/No",
        ),
    )
    result = runner.invoke(app, ["status", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    jev = data["jev"]
    assert jev["schema_name"] == "yes-no"
    assert jev["sha256"] == "abcdef012345"
    assert jev["source"] == "store"
    assert jev["stale"] is True
    assert jev["title"] == "Yes/No"
    assert jev["applied_at_unix_ms"] == 1726000000123
    assert isinstance(jev["applied_at_unix_ms"], int)


def test_d6_status_text_jev_line_and_stale_hint(
    runner: CliRunner, fake_client: FakeClient
) -> None:
    fake_client.status_value = control_pb2.Status(
        active_alias="qwen2.5-coder-7b",
        state=control_pb2.READY,
        jev=control_pb2.JevInfo(
            schema_name="yes-no", source="store", sha256="abcdef012345", stale=True
        ),
    )
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Jev: yes-no (store, sha abcdef012345)" in result.stdout
    assert "stale" in result.stdout


def test_d6_status_text_jev_off_when_inactive(
    runner: CliRunner, fake_client: FakeClient
) -> None:
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "Jev: off" in result.stdout


# -- D7: schema validate offline vs --name -----------------------------------


@pytest.fixture
def offline_runner(monkeypatch: pytest.MonkeyPatch) -> CliRunner:
    """A runner whose `ControllerClient` explodes if anything calls it."""

    def _no_controller(settings):  # pragma: no cover - never called
        raise AssertionError("controller contacted for offline validation")

    monkeypatch.setenv("VLLM_IMAGE_TAG", "v0.11.2")
    monkeypatch.setattr(cli, "ControllerClient", _no_controller)
    return CliRunner()


def test_d7_validate_file_offline(offline_runner: CliRunner, tmp_path) -> None:
    schema_file = tmp_path / "schema.json"
    schema_file.write_text(VALID_SCHEMA, encoding="utf-8")
    result = offline_runner.invoke(app, ["jev", "schema", "validate", str(schema_file)])
    assert result.exit_code == 0
    assert "schema is valid" in result.stdout


def test_d7_validate_stdin_offline(offline_runner: CliRunner) -> None:
    result = offline_runner.invoke(
        app, ["jev", "schema", "validate", "-"], input=VALID_SCHEMA
    )
    assert result.exit_code == 0
    assert "schema is valid" in result.stdout


def test_d7_validate_file_invalid_offline(offline_runner: CliRunner, tmp_path) -> None:
    schema_file = tmp_path / "bad.json"
    schema_file.write_text("{nope")
    result = offline_runner.invoke(app, ["jev", "schema", "validate", str(schema_file)])
    assert result.exit_code == 2
    assert "not valid JSON" in result.stderr


def test_d7_validate_file_warnings_still_exit_zero(
    offline_runner: CliRunner, tmp_path
) -> None:
    schema_file = tmp_path / "loose.json"
    schema_file.write_text(json.dumps({"type": "object"}))
    result = offline_runner.invoke(app, ["jev", "schema", "validate", str(schema_file)])
    assert result.exit_code == 0
    assert "warning:" in result.stderr
    assert "schema is valid" in result.stdout


def test_d7_validate_missing_file(runner: CliRunner, fake_client: FakeClient) -> None:
    result = runner.invoke(app, ["jev", "schema", "validate", "/nonexistent/schema.json"])
    assert result.exit_code == 2
    assert "/nonexistent/schema.json" in result.stderr


def test_d7_validate_needs_file_or_name(
    runner: CliRunner, fake_client: FakeClient
) -> None:
    result = runner.invoke(app, ["jev", "schema", "validate"])
    assert result.exit_code == 2
    assert fake_client.calls == []


def test_d7_validate_file_and_name_conflict(
    runner: CliRunner, fake_client: FakeClient
) -> None:
    result = runner.invoke(app, ["jev", "schema", "validate", "f.json", "--name", "x"])
    assert result.exit_code == 2
    assert fake_client.calls == []


def test_d7_validate_name_uses_controller(
    runner: CliRunner, fake_client: FakeClient
) -> None:
    fake_client.schema_value = control_pb2.GetJevSchemaResponse(
        name="yes-no", schema_json=VALID_SCHEMA
    )
    fake_client.validate_value = control_pb2.JevValidation(
        ok=True, warnings=["warn1"], sha256="abcdef012345"
    )
    result = runner.invoke(app, ["jev", "schema", "validate", "--name", "yes-no"])
    assert result.exit_code == 0
    assert fake_client.get_calls == ["yes-no"]
    assert fake_client.validate_calls == [VALID_SCHEMA]
    assert "warn1" in result.stderr
    assert "valid" in result.stdout


def test_d7_validate_name_invalid_schema(
    runner: CliRunner, fake_client: FakeClient
) -> None:
    fake_client.schema_value = control_pb2.GetJevSchemaResponse(
        name="yes-no", schema_json=VALID_SCHEMA
    )
    fake_client.validate_value = control_pb2.JevValidation(
        ok=False, errors=["bad schema"]
    )
    result = runner.invoke(app, ["jev", "schema", "validate", "--name", "yes-no"])
    assert result.exit_code == 2
    assert "bad schema" in result.stderr


def test_d7_validate_name_controller_down(
    runner: CliRunner, fake_client: FakeClient
) -> None:
    fake_client.get_error = ControllerUnavailable("down")
    result = runner.invoke(app, ["jev", "schema", "validate", "--name", "yes-no"])
    assert result.exit_code == 1
    assert "controller unreachable" in result.stderr


def test_d7_validate_name_unknown_schema(
    runner: CliRunner, fake_client: FakeClient
) -> None:
    fake_client.get_error = JevSchemaNotFound("no Jev schema named 'nope'")
    result = runner.invoke(app, ["jev", "schema", "validate", "--name", "nope"])
    assert result.exit_code == 3
    assert "nope" in result.stderr


# -- D8: schema list/show/put/rm/export --------------------------------------


def test_d8_schema_list_table(runner: CliRunner, fake_client: FakeClient) -> None:
    fake_client.schemas = [
        control_pb2.JevSchemaEntry(
            name="default",
            source="builtin",
            sha256="a" * 32,
            size_bytes=3 * 2**20,
            title="Default",
        ),
        control_pb2.JevSchemaEntry(
            name="yes-no", source="user", sha256="b" * 32, size_bytes=100, active=True
        ),
    ]
    result = runner.invoke(app, ["jev", "schema", "list"])
    assert result.exit_code == 0
    assert "NAME" in result.stdout
    assert "SOURCE" in result.stdout
    assert "SHA12" in result.stdout
    assert "SIZE" in result.stdout
    assert "ACTIVE" in result.stdout
    assert "default" in result.stdout and "builtin" in result.stdout
    assert "yes-no" in result.stdout and "user" in result.stdout
    assert "a" * 12 in result.stdout and "b" * 12 in result.stdout
    assert "3.0 MiB" in result.stdout
    assert "*" in result.stdout


def test_d8_schema_list_empty(runner: CliRunner, fake_client: FakeClient) -> None:
    result = runner.invoke(app, ["jev", "schema", "list"])
    assert result.exit_code == 0
    assert "no Jev schemas" in result.stdout


def test_d8_schema_list_json(runner: CliRunner, fake_client: FakeClient) -> None:
    fake_client.schemas = [
        control_pb2.JevSchemaEntry(
            name="default",
            source="builtin",
            sha256="c" * 32,
            size_bytes=3,
            mtime_unix_ms=1726000000123,
            active=True,
        )
    ]
    result = runner.invoke(app, ["jev", "schema", "list", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data[0]["name"] == "default"
    assert data[0]["source"] == "builtin"
    assert data[0]["active"] is True
    assert data[0]["mtime_unix_ms"] == 1726000000123
    assert isinstance(data[0]["mtime_unix_ms"], int)


def test_d8_schema_show_pretty(runner: CliRunner, fake_client: FakeClient) -> None:
    fake_client.schema_value = control_pb2.GetJevSchemaResponse(
        name="yes-no", schema_json='{"title": "Yes", "type": "object"}'
    )
    result = runner.invoke(app, ["jev", "schema", "show", "yes-no"])
    assert result.exit_code == 0
    assert '"title": "Yes"' in result.stdout
    assert '\n  "type": "object"' in result.stdout


def test_d8_schema_show_raw_exact_bytes(
    runner: CliRunner, fake_client: FakeClient
) -> None:
    raw = '{"type":"object"}\n'
    fake_client.schema_value = control_pb2.GetJevSchemaResponse(
        name="yes-no", schema_json=raw
    )
    result = runner.invoke(app, ["jev", "schema", "show", "yes-no", "--raw"])
    assert result.exit_code == 0
    assert result.stdout == raw


def test_d8_schema_show_unknown_exits_three(
    runner: CliRunner, fake_client: FakeClient
) -> None:
    fake_client.get_error = JevSchemaNotFound("no Jev schema named 'nope'")
    result = runner.invoke(app, ["jev", "schema", "show", "nope"])
    assert result.exit_code == 3
    assert "nope" in result.stderr


def test_d8_schema_put_from_file(runner: CliRunner, fake_client: FakeClient, tmp_path) -> None:
    schema_file = tmp_path / "s.json"
    schema_file.write_text(VALID_SCHEMA, encoding="utf-8")
    fake_client.put_value = control_pb2.PutJevSchemaResponse(
        validation=control_pb2.JevValidation(
            ok=True, warnings=["w1"], sha256="abcdef012345", size_bytes=123
        )
    )
    result = runner.invoke(app, ["jev", "schema", "put", "yes-no", str(schema_file)])
    assert result.exit_code == 0
    assert fake_client.put_calls == [("yes-no", VALID_SCHEMA)]
    assert "stored schema 'yes-no'" in result.stdout
    assert "w1" in result.stderr


def test_d8_schema_put_from_stdin(runner: CliRunner, fake_client: FakeClient) -> None:
    result = runner.invoke(app, ["jev", "schema", "put", "yes-no", "-"], input=VALID_SCHEMA)
    assert result.exit_code == 0
    assert fake_client.put_calls == [("yes-no", VALID_SCHEMA)]


def test_d8_schema_put_invalid_schema_no_rpc(
    runner: CliRunner, fake_client: FakeClient, tmp_path
) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{nope")
    result = runner.invoke(app, ["jev", "schema", "put", "yes-no", str(bad)])
    assert result.exit_code == 2
    assert fake_client.put_calls == []


def test_d8_schema_put_invalid_name_no_rpc(
    runner: CliRunner, fake_client: FakeClient, tmp_path
) -> None:
    good = tmp_path / "ok.json"
    good.write_text(VALID_SCHEMA, encoding="utf-8")
    result = runner.invoke(app, ["jev", "schema", "put", "Bad Name!", str(good)])
    assert result.exit_code == 2
    assert "invalid schema name" in result.stderr
    assert fake_client.put_calls == []


def test_d8_schema_put_controller_refuses_validation(
    runner: CliRunner, fake_client: FakeClient, tmp_path
) -> None:
    good = tmp_path / "ok.json"
    good.write_text(VALID_SCHEMA, encoding="utf-8")
    fake_client.put_value = control_pb2.PutJevSchemaResponse(
        validation=control_pb2.JevValidation(ok=False, errors=["bad schema"])
    )
    result = runner.invoke(app, ["jev", "schema", "put", "yes-no", str(good)])
    assert result.exit_code == 2
    assert "bad schema" in result.stderr


def test_d8_schema_put_store_level_refusal(
    runner: CliRunner, fake_client: FakeClient, tmp_path
) -> None:
    good = tmp_path / "ok.json"
    good.write_text(VALID_SCHEMA, encoding="utf-8")
    fake_client.put_value = control_pb2.PutJevSchemaResponse(
        error="at most 256 user schemas are allowed"
    )
    result = runner.invoke(app, ["jev", "schema", "put", "yes-no", str(good)])
    assert result.exit_code == 2
    assert "at most 256 user schemas" in result.stderr


def test_d8_schema_rm_user(runner: CliRunner, fake_client: FakeClient) -> None:
    result = runner.invoke(app, ["jev", "schema", "rm", "yes-no"])
    assert result.exit_code == 0
    assert fake_client.delete_calls == ["yes-no"]
    assert "deleted schema 'yes-no'" in result.stdout


def test_d8_schema_rm_builtin_exit_one(
    runner: CliRunner, fake_client: FakeClient
) -> None:
    fake_client.delete_error = JevReadOnly()
    result = runner.invoke(app, ["jev", "schema", "rm", "default"])
    assert result.exit_code == 1
    assert "built-in schemas are read-only" in result.stderr


def test_d8_schema_rm_override_reverts(
    runner: CliRunner, fake_client: FakeClient
) -> None:
    fake_client.delete_value = control_pb2.DeleteJevSchemaResponse(
        ok=True, reverted_to="builtin"
    )
    result = runner.invoke(app, ["jev", "schema", "rm", "default"])
    assert result.exit_code == 0
    assert "reverted to built-in" in result.stdout


def test_d8_schema_rm_unknown_exits_three(
    runner: CliRunner, fake_client: FakeClient
) -> None:
    fake_client.delete_error = JevSchemaNotFound("no Jev schema named 'nope'")
    result = runner.invoke(app, ["jev", "schema", "rm", "nope"])
    assert result.exit_code == 3


def test_d8_schema_rm_active_notice(runner: CliRunner, fake_client: FakeClient) -> None:
    fake_client.active_value = control_pb2.ActiveJev(
        info=control_pb2.JevInfo(schema_name="yes-no", sha256="ab")
    )
    result = runner.invoke(app, ["jev", "schema", "rm", "yes-no"])
    assert result.exit_code == 0
    assert "deleted schema 'yes-no'" in result.stdout
    assert "note:" in result.stdout and "active" in result.stdout


def test_d8_schema_export_stdout(runner: CliRunner, fake_client: FakeClient) -> None:
    body = '{"type": "object"}'
    fake_client.schema_value = control_pb2.GetJevSchemaResponse(
        name="yes-no", schema_json=body
    )
    result = runner.invoke(app, ["jev", "schema", "export", "yes-no"])
    assert result.exit_code == 0
    assert result.stdout == body + "\n"


def test_d8_schema_export_to_file(
    runner: CliRunner, fake_client: FakeClient, tmp_path
) -> None:
    out = tmp_path / "out.json"
    body = '{"type": "object"}\n'
    fake_client.schema_value = control_pb2.GetJevSchemaResponse(
        name="yes-no", schema_json=body
    )
    result = runner.invoke(app, ["jev", "schema", "export", "yes-no", "-o", str(out)])
    assert result.exit_code == 0
    assert out.read_text(encoding="utf-8") == body
    assert "exported 'yes-no'" in result.stdout


def test_d8_schema_export_unknown_exits_three(
    runner: CliRunner, fake_client: FakeClient
) -> None:
    fake_client.get_error = JevSchemaNotFound("no Jev schema named 'nope'")
    result = runner.invoke(app, ["jev", "schema", "export", "nope"])
    assert result.exit_code == 3


# -- jev status / off --------------------------------------------------------


def test_jev_status_off(runner: CliRunner, fake_client: FakeClient) -> None:
    result = runner.invoke(app, ["jev", "status"])
    assert result.exit_code == 0
    assert "Jev: off" in result.stdout


def test_jev_status_active(runner: CliRunner, fake_client: FakeClient) -> None:
    fake_client.active_value = control_pb2.ActiveJev(
        info=control_pb2.JevInfo(
            schema_name="yes-no",
            source="store",
            sha256="abcdef012345",
            applied_at_unix_ms=1726000000123,
            stale=True,
            title="Yes/No",
        ),
        schema_json=VALID_SCHEMA,
    )
    result = runner.invoke(app, ["jev", "status"])
    assert result.exit_code == 0
    assert "yes-no" in result.stdout
    assert "store" in result.stdout
    assert "abcdef012345" in result.stdout
    assert "stale" in result.stdout
    assert "Yes/No" in result.stdout


def test_jev_status_json_active(runner: CliRunner, fake_client: FakeClient) -> None:
    fake_client.active_value = control_pb2.ActiveJev(
        info=control_pb2.JevInfo(
            schema_name="yes-no",
            source="store",
            sha256="abcdef012345",
            applied_at_unix_ms=1726000000123,
            stale=False,
        )
    )
    result = runner.invoke(app, ["jev", "status", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data["active"] is True
    assert data["info"]["schema_name"] == "yes-no"
    assert data["info"]["applied_at_unix_ms"] == 1726000000123
    assert isinstance(data["info"]["applied_at_unix_ms"], int)


def test_jev_status_json_inactive(runner: CliRunner, fake_client: FakeClient) -> None:
    result = runner.invoke(app, ["jev", "status", "--json"])
    assert result.exit_code == 0
    data = json.loads(result.stdout)
    assert data == {"active": False, "info": None}


def test_jev_off_detaches_active_model(
    runner: CliRunner, fake_client: FakeClient
) -> None:
    fake_client.status_value = control_pb2.Status(
        active_alias="qwen2.5-coder-7b", state=control_pb2.READY
    )
    fake_client.load_events = [
        control_pb2.JobEvent(message="jev: detached 'yes-no'"),
        _ready("qwen2.5-coder-7b is already READY — Jev off"),
    ]
    result = runner.invoke(app, ["jev", "off"])
    assert result.exit_code == 0
    assert fake_client.load_calls == [
        ("qwen2.5-coder-7b", control_pb2.JEV_OFF, None)
    ]
    assert "detached" in result.stdout


def test_jev_off_not_active_idempotent(
    runner: CliRunner, fake_client: FakeClient
) -> None:
    fake_client.status_value = control_pb2.Status(
        active_alias="qwen2.5-coder-7b", state=control_pb2.READY
    )
    fake_client.load_events = [
        control_pb2.JobEvent(message="jev: not active — nothing to detach"),
        _ready("qwen2.5-coder-7b is already READY — Jev off"),
    ]
    result = runner.invoke(app, ["jev", "off"])
    assert result.exit_code == 0
    assert "nothing to detach" in result.stdout


def test_jev_off_no_active_model_skips_rpc(
    runner: CliRunner, fake_client: FakeClient
) -> None:
    result = runner.invoke(app, ["jev", "off"])
    assert result.exit_code == 0
    assert "nothing to detach" in result.stdout
    assert "load" not in fake_client.calls


# -- JEV-04.3: fast-path attach output ---------------------------------------


def test_load_jev_fast_path_says_attached_without_restart(
    runner: CliRunner, fake_client: FakeClient
) -> None:
    fake_client.load_events = [
        control_pb2.JobEvent(message="jev: probing 'yes-no'…"),
        control_pb2.JobEvent(
            message="jev: attached 'yes-no' (store, sha abcdef012345)",
            state=control_pb2.READY,
        ),
    ]
    result = runner.invoke(app, ["load", "x", "--jev"])
    assert result.exit_code == 0
    assert "attached without restart" in result.stdout


def test_load_jev_fresh_load_has_no_fast_path_message(
    runner: CliRunner, fake_client: FakeClient
) -> None:
    fake_client.load_events = [
        control_pb2.JobEvent(message="starting", state=control_pb2.LOADING),
        control_pb2.JobEvent(message="ready"),
        control_pb2.JobEvent(
            message="jev: attached 'yes-no' (store, sha abcdef012345)",
            state=control_pb2.READY,
        ),
    ]
    result = runner.invoke(app, ["load", "x", "--jev"])
    assert result.exit_code == 0
    assert "attached without restart" not in result.stdout


# -- load option plumbing ----------------------------------------------------


def test_load_schema_plumbs_name(runner: CliRunner, fake_client: FakeClient) -> None:
    fake_client.load_events = [_ready()]
    result = runner.invoke(app, ["load", "x", "--schema", "yes-no"])
    assert result.exit_code == 0
    _, jev_mode, jev = fake_client.load_calls[0]
    assert jev_mode == control_pb2.JEV_ON
    assert jev is not None and jev.schema_name == "yes-no" and jev.schema_json == ""


def test_load_schema_invalid_name_no_rpc(runner: CliRunner, fake_client: FakeClient) -> None:
    result = runner.invoke(app, ["load", "x", "--schema", "Bad Name"])
    assert result.exit_code == 2
    assert fake_client.load_calls == []


def test_load_save_as_plumbs_spec(runner: CliRunner, fake_client: FakeClient, tmp_path) -> None:
    schema_file = tmp_path / "s.json"
    schema_file.write_text(VALID_SCHEMA, encoding="utf-8")
    fake_client.load_events = [_ready()]
    result = runner.invoke(
        app, ["load", "x", "--include-json", str(schema_file), "--save-as", "yes-no"]
    )
    assert result.exit_code == 0
    _, jev_mode, jev = fake_client.load_calls[0]
    assert jev_mode == control_pb2.JEV_ON
    assert jev is not None
    assert jev.schema_json == VALID_SCHEMA
    assert jev.save_as == "yes-no"


def test_load_save_as_invalid_name_no_rpc(
    runner: CliRunner, fake_client: FakeClient, tmp_path
) -> None:
    schema_file = tmp_path / "s.json"
    schema_file.write_text(VALID_SCHEMA, encoding="utf-8")
    result = runner.invoke(
        app, ["load", "x", "--include-json", str(schema_file), "--save-as", "Bad Name"]
    )
    assert result.exit_code == 2
    assert fake_client.load_calls == []


def test_load_no_jev_sends_off(runner: CliRunner, fake_client: FakeClient) -> None:
    fake_client.load_events = [
        control_pb2.JobEvent(message="jev: detached 'yes-no'"),
        _ready("qwen2.5-coder-7b is already READY — Jev off"),
    ]
    result = runner.invoke(app, ["load", "x", "--no-jev"])
    assert result.exit_code == 0
    assert fake_client.load_calls == [("x", control_pb2.JEV_OFF, None)]
    assert "attached without restart" not in result.stdout

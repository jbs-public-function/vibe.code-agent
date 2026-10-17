"""NFR-02.2: committed codegen must match a fresh `grpc_tools.protoc` run.

Regenerating `proto/vibe/v1/control.proto` into a temp dir must reproduce
`src/vibe/gen/vibe/v1/` byte-for-byte. Any drift means someone edited the
proto without running `make proto` (or committed with a different
grpcio-tools version).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

pytest.importorskip("grpc_tools", reason="grpc_tools not installed")

REPO = Path(__file__).resolve().parents[1]
PROTO = REPO / "proto" / "vibe" / "v1" / "control.proto"
GEN_DIR = REPO / "src" / "vibe" / "gen" / "vibe" / "v1"

GENERATED = ("control_pb2.py", "control_pb2_grpc.py", "control_pb2.pyi")


def _regenerate(out_dir: Path) -> Path:
    """Run `grpc_tools.protoc` into `out_dir`; return the generated package dir."""
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            "-I",
            str(REPO / "proto"),
            f"--python_out={out_dir}",
            f"--grpc_python_out={out_dir}",
            f"--pyi_out={out_dir}",
            str(PROTO),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"grpc_tools.protoc failed:\n{result.stdout}\n{result.stderr}"
    )
    return out_dir / "vibe" / "v1"


def test_codegen_matches_commit() -> None:
    """Freshly generated code must be identical to the committed files."""
    tmp = Path(tempfile.mkdtemp(prefix="vibe-codegen-"))
    try:
        fresh = _regenerate(tmp)
        drifted = [
            name
            for name in GENERATED
            if (fresh / name).read_bytes() != (GEN_DIR / name).read_bytes()
        ]
        assert not drifted, (
            "committed codegen differs from a fresh regeneration for: "
            f"{', '.join(drifted)} — run `make proto` and commit the result"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

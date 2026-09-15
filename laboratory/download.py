"""Download a model snapshot into the shared Hugging Face cache (FR-05.4, FR-05.10)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import common


def build_parser() -> argparse.ArgumentParser:
    """Build the download.py argument parser (FR-05.4)."""
    parser = argparse.ArgumentParser(
        prog="download.py",
        description="Download a model snapshot from the Hugging Face Hub into HF_HOME (FR-05.4).",
    )
    parser.add_argument(
        "repo",
        help="Hugging Face repo id (e.g. Qwen/Qwen2.5-Coder-1.5B-Instruct)",
    )
    parser.add_argument(
        "--include",
        action="append",
        metavar="PATTERN",
        help="only download files matching PATTERN (repeatable, fnmatch globs)",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="repo revision: branch, tag, or commit hash",
    )
    common.add_common_args(parser)
    return parser


def _snapshot_download(
    repo_id: str,
    allow_patterns: list[str] | None,
    revision: str | None,
) -> str:
    """Download a snapshot and return its local path (seam for tests).

    Downloads into `$VIBE_DATA_DIR` when set, else HF's default cache
    (`~/.cache/huggingface`, ADR-8): huggingface_hub defaults there, so no
    explicit `cache_dir` is passed in that case.
    """
    from huggingface_hub import snapshot_download

    kwargs: dict[str, Any] = {
        "repo_id": repo_id,
        "allow_patterns": allow_patterns or None,
        "revision": revision,
        "token": os.environ.get("HF_TOKEN"),
    }
    cache_dir = os.environ.get("VIBE_DATA_DIR")
    if cache_dir:
        kwargs["cache_dir"] = str(Path(cache_dir).expanduser())
    return str(snapshot_download(**kwargs))


def main(argv: list[str] | None = None) -> int:
    """Run the download. Exit codes: 0 ok, 1 runtime error, 2 usage error."""
    parser = build_parser()
    args = parser.parse_args(argv)
    run = common.start_run(
        "download",
        list(argv) if argv is not None else sys.argv[1:],
        runs_dir=args.run_dir,
    )
    common.setup_logging()
    common.log(run, f"download.py {args.repo}")
    seed = common.parse_seed(args.seed)
    run.log_line(f"seed: {seed}")

    # Downloads do not train: the estimate is tiny (NFR-05.3), and serving
    # can safely coexist, but the guard is still honored (NFR-05.4).
    allowed, message = common.vram_guard(0.2, force=args.force)
    common.log(run, message)
    if not allowed:
        return common.EXIT_RUNTIME
    allowed, message = common.serving_guard(args.allow_coexist)
    common.log(run, message)
    if not allowed:
        return common.EXIT_RUNTIME

    try:
        snapshot = _snapshot_download(args.repo, args.include, args.revision)
    except Exception as exc:
        common.log(run, f"download failed: {exc}")
        return common.EXIT_RUNTIME
    print(snapshot)  # machine-readable result on stdout
    common.log(run, f"snapshot: {snapshot}")
    run.write_metrics({"snapshot": snapshot, "repo": args.repo, "seed": seed})
    common.print_peak_vram(run)
    return common.EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())

"""Unit tests for the pure nvidia-smi CSV parsers in `vibe.gpu`."""

from __future__ import annotations

import pytest

from vibe.gpu import parse_gpu_csv, parse_process_csv


# -- parse_gpu_csv ----------------------------------------------------------


def test_parse_gpu_csv_basic_row():
    stats = parse_gpu_csv("1234, 16384, 42\n")
    assert stats.used_mib == 1234
    assert stats.total_mib == 16384
    assert stats.utilization_pct == 42
    assert stats.processes == ()


def test_parse_gpu_csv_empty_cells_default_to_zero():
    stats = parse_gpu_csv(",,\n")
    assert stats.used_mib == 0
    assert stats.total_mib == 0
    assert stats.utilization_pct == 0
    assert stats.processes == ()


def test_parse_gpu_csv_two_field_row_is_malformed():
    # The parser requires the three columns (used,total,utilization); a row
    # with fewer fields is malformed input, not a row of empty cells.
    with pytest.raises(ValueError):
        parse_gpu_csv(",\n")


def test_parse_gpu_csv_garbage_raises_value_error():
    with pytest.raises(ValueError):
        parse_gpu_csv("garbage")


# -- parse_process_csv --------------------------------------------------------


def test_parse_process_csv_two_rows():
    procs = parse_process_csv("123, python, 5000\n456, /usr/bin/docker, 200\n")
    assert len(procs) == 2
    first, second = procs
    assert first.pid == 123
    assert first.name == "python"
    assert first.used_mib == 5000
    assert second.pid == 456
    assert second.name == "/usr/bin/docker"
    assert second.used_mib == 200


def test_parse_process_csv_name_containing_comma():
    procs = parse_process_csv('123, "python, worker", 5000\n')
    assert len(procs) == 1
    proc = procs[0]
    assert proc.pid == 123
    assert proc.used_mib == 5000
    assert proc.name  # name is rebuilt by joining the middle fields


def test_parse_process_csv_skips_garbage_rows():
    procs = parse_process_csv(
        "not,a,row,tooshort\n"
        "abc, name, 123\n"
        "789, valid proc, 42\n"
    )
    assert len(procs) == 1
    proc = procs[0]
    assert proc.pid == 789
    assert proc.name == "valid proc"
    assert proc.used_mib == 42

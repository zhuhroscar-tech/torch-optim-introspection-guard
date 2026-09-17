"""Tests for the CLI entry point: argument parsing, --version, --json,
--no-color, and exit codes -- independent of whether torch is installed.
Mirrors the test_cli.py pattern already used across the fleet (e.g.
rng-leak-audit, causality-audit) for a repo that had none."""
from __future__ import annotations

import json

import pytest

from torch_optim_introspection_guard.cli import main


def test_version_flag(capsys):
    code = main(["--version"])
    out = capsys.readouterr().out
    assert code == 0
    assert "torch-optim-introspection-guard" in out


def test_json_output_is_valid_json_and_reports_guard_status(capsys):
    torch = pytest.importorskip("torch")
    code = main(["--json"])
    out = capsys.readouterr().out
    report = json.loads(out)
    assert "torch_version" in report
    assert report["torch_version"] == torch.__version__
    assert "guard_fully_effective" in report
    assert code in (0, 1)


def test_json_exit_code_matches_guard_fully_effective(capsys):
    torch = pytest.importorskip("torch")
    code = main(["--json"])
    out = capsys.readouterr().out
    report = json.loads(out)
    assert code == (0 if report["guard_fully_effective"] else 1)


def test_text_output_no_color_has_no_ansi_escapes(capsys):
    pytest.importorskip("torch")
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "\x1b[" not in out


def test_text_output_reports_per_optimizer_results(capsys):
    pytest.importorskip("torch")
    main(["--no-color"])
    out = capsys.readouterr().out
    assert "torch version" in out
    assert "per-optimizer results" in out


def test_seed_flag_is_accepted_and_deterministic(capsys):
    torch = pytest.importorskip("torch")
    main(["--json", "--seed", "7"])
    out1 = capsys.readouterr().out
    main(["--json", "--seed", "7"])
    out2 = capsys.readouterr().out
    assert json.loads(out1) == json.loads(out2)

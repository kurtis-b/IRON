# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from pathlib import Path
from types import SimpleNamespace

from iron.common import compilation as comp


def _write_file(path: Path, text: str = ""):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_aiecc_compilation_rule_prefers_cpp_driver_and_primes_path(
    tmp_path, monkeypatch
):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    mlir_aie_dir = tmp_path / "mlir_aie"
    peano_dir = tmp_path / "llvm_aie"
    fake_python = tmp_path / "venv" / "bin" / "python3"
    aiecc_bin = mlir_aie_dir / "bin" / "aiecc"
    aiecc_py = mlir_aie_dir / "bin" / "aiecc.py"

    _write_file(fake_python)
    _write_file(aiecc_bin)
    _write_file(aiecc_py)

    mlir_source_path = tmp_path / "case.mlir"
    _write_file(mlir_source_path, "module {}\n")
    mlir_source = comp.SourceArtifact.new(mlir_source_path)
    xclbin = comp.XclbinArtifact.new(tmp_path / "case.xclbin", depends=[mlir_source])

    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(comp.subprocess, "run", fake_run)
    monkeypatch.setattr(comp.sys, "executable", str(fake_python))
    monkeypatch.setenv("PATH", "/usr/bin")

    rule = comp.AieccCompilationRule(build_dir, peano_dir, mlir_aie_dir)
    rule.compile([xclbin])

    assert captured["command"][0] == str(aiecc_bin)
    path_entries = captured["kwargs"]["env"]["PATH"].split(os.pathsep)
    assert path_entries[:3] == [
        str(fake_python.parent),
        str(mlir_aie_dir / "bin"),
        str(peano_dir / "bin"),
    ]


def test_aiecc_compilation_rule_falls_back_to_python_wrapper(tmp_path, monkeypatch):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    mlir_aie_dir = tmp_path / "mlir_aie"
    peano_dir = tmp_path / "llvm_aie"
    fake_python = tmp_path / "venv" / "bin" / "python3"
    aiecc_py = mlir_aie_dir / "bin" / "aiecc.py"

    _write_file(fake_python)
    _write_file(aiecc_py)

    mlir_source_path = tmp_path / "case.mlir"
    _write_file(mlir_source_path, "module {}\n")
    mlir_source = comp.SourceArtifact.new(mlir_source_path)
    xclbin = comp.XclbinArtifact.new(tmp_path / "case.xclbin", depends=[mlir_source])

    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(comp.subprocess, "run", fake_run)
    monkeypatch.setattr(comp.sys, "executable", str(fake_python))
    monkeypatch.setenv("PATH", "/usr/bin")

    rule = comp.AieccCompilationRule(build_dir, peano_dir, mlir_aie_dir)
    rule.compile([xclbin])

    assert captured["command"][:2] == [str(fake_python), str(aiecc_py)]

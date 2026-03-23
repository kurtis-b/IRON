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


def test_aiecc_compilation_rule_passes_xclbin_input_for_insts_only(
    tmp_path, monkeypatch
):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    mlir_aie_dir = tmp_path / "mlir_aie"
    peano_dir = tmp_path / "llvm_aie"
    fake_python = tmp_path / "venv" / "bin" / "python3"
    aiecc_bin = mlir_aie_dir / "bin" / "aiecc"

    _write_file(fake_python)
    _write_file(aiecc_bin)

    mlir_source_path = tmp_path / "case.mlir"
    _write_file(mlir_source_path, "module {}\n")
    mlir_source = comp.SourceArtifact.new(mlir_source_path)
    shared_xclbin = comp.XclbinArtifact.new(
        tmp_path / "shared.xclbin", depends=[mlir_source]
    )
    shared_xclbin.fake_available = True
    insts = comp.InstsBinArtifact.new(
        tmp_path / "case.bin",
        depends=[mlir_source],
        xclbin_input=shared_xclbin,
    )

    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(comp.subprocess, "run", fake_run)
    monkeypatch.setattr(comp.sys, "executable", str(fake_python))
    monkeypatch.setenv("PATH", "/usr/bin")

    rule = comp.AieccCompilationRule(build_dir, peano_dir, mlir_aie_dir)
    rule.compile([insts])

    assert "--no-compile" in captured["command"]
    assert f"--xclbin-input={shared_xclbin.path}" in captured["command"]
    assert f"--npu-insts-name={insts.path}" in captured["command"]


def test_aiecc_compilation_rule_passes_kernel_name_for_insts_only(
    tmp_path, monkeypatch
):
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    mlir_aie_dir = tmp_path / "mlir_aie"
    peano_dir = tmp_path / "llvm_aie"
    fake_python = tmp_path / "venv" / "bin" / "python3"
    aiecc_bin = mlir_aie_dir / "bin" / "aiecc"

    _write_file(fake_python)
    _write_file(aiecc_bin)

    mlir_source_path = tmp_path / "case.mlir"
    _write_file(mlir_source_path, "module {}\n")
    mlir_source = comp.SourceArtifact.new(mlir_source_path)
    shared_xclbin = comp.XclbinArtifact.new(
        tmp_path / "shared-kernels.xclbin",
        depends=[mlir_source],
    )
    shared_xclbin.fake_available = True
    insts = comp.InstsBinArtifact.new(
        tmp_path / "case-kernel.bin",
        depends=[mlir_source],
        xclbin_input=shared_xclbin,
        kernel_name="encoder_k_transpose",
    )

    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(comp.subprocess, "run", fake_run)
    monkeypatch.setattr(comp.sys, "executable", str(fake_python))
    monkeypatch.setenv("PATH", "/usr/bin")

    rule = comp.AieccCompilationRule(build_dir, peano_dir, mlir_aie_dir)
    rule.compile([insts])

    assert f"--xclbin-input={shared_xclbin.path}" in captured["command"]
    assert "--xclbin-kernel-name=encoder_k_transpose" in captured["command"]


def test_insts_artifact_invalidates_when_xclbin_input_is_newer(tmp_path):
    mlir_source_path = tmp_path / "case.mlir"
    shared_xclbin_path = tmp_path / "shared.xclbin"
    insts_path = tmp_path / "case.bin"

    _write_file(mlir_source_path, "module {}\n")
    _write_file(insts_path, "insts")
    _write_file(shared_xclbin_path, "xclbin")

    mlir_source = comp.SourceArtifact.new(mlir_source_path)
    insts = comp.InstsBinArtifact.new(
        insts_path,
        depends=[mlir_source],
        xclbin_input=comp.XclbinArtifact.new(shared_xclbin_path, depends=[mlir_source]),
    )

    older = 1000
    newer = 2000
    os.utime(mlir_source_path, (older, older))
    os.utime(insts_path, (older, older))
    os.utime(shared_xclbin_path, (newer, newer))

    assert not insts.is_available()


def test_xclbin_artifact_invalidates_when_xclbin_input_is_newer(tmp_path):
    mlir_source_path = tmp_path / "case.mlir"
    base_xclbin_path = tmp_path / "base.xclbin"
    chained_xclbin_path = tmp_path / "chained.xclbin"

    _write_file(mlir_source_path, "module {}\n")
    _write_file(base_xclbin_path, "base")
    _write_file(chained_xclbin_path, "chained")

    mlir_source = comp.SourceArtifact.new(mlir_source_path)
    base_xclbin = comp.XclbinArtifact.new(base_xclbin_path, depends=[mlir_source])
    chained_xclbin = comp.XclbinArtifact.new(
        chained_xclbin_path,
        depends=[mlir_source],
        xclbin_input=base_xclbin,
    )

    older = 1000
    newer = 2000
    os.utime(mlir_source_path, (older, older))
    os.utime(chained_xclbin_path, (older, older))
    os.utime(base_xclbin_path, (newer, newer))

    assert not chained_xclbin.is_available()

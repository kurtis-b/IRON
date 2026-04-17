# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import time
from pathlib import Path

from iron.common.compilation import CompilationArtifact, CompilationRule, compile


class FakeArtifact(CompilationArtifact):
    pass


class BuildNamedArtifactRule(CompilationRule):
    def __init__(self, name, *, touch_after=None):
        super().__init__()
        self.name = name
        self.touch_after = touch_after

    def matches(self, artifacts):
        return any(artifact.path.name == self.name for artifact in artifacts)

    def compile(self, artifacts):
        for artifact in artifacts:
            if artifact.path.name != self.name:
                continue
            artifact.path.parent.mkdir(parents=True, exist_ok=True)
            artifact.path.write_text(self.name, encoding="utf-8")
            if self.touch_after is not None:
                current_time = time.time() + 1.0
                os.utime(self.touch_after, (current_time, current_time))
        return artifacts


def test_compile_rediscovers_newly_stale_ancestor(tmp_path):
    CompilationArtifact._instances.clear()

    dep_a = FakeArtifact.new(tmp_path / "a")
    dep_b = FakeArtifact.new(tmp_path / "b", depends=[dep_a])
    dep_d = FakeArtifact.new(tmp_path / "d")
    root_c = FakeArtifact.new(tmp_path / "c", depends=[dep_b, dep_d])

    dep_a.path.write_text("a", encoding="utf-8")
    time.sleep(0.01)
    dep_b.path.write_text("b", encoding="utf-8")
    time.sleep(0.01)
    root_c.path.write_text("c", encoding="utf-8")

    assert dep_b.is_available()
    assert root_c.is_available() is False

    rules = [
        BuildNamedArtifactRule("d", touch_after=dep_a.path),
        BuildNamedArtifactRule("b"),
        BuildNamedArtifactRule("c"),
    ]

    compile(rules, [root_c])

    assert dep_a.is_available()
    assert dep_b.is_available()
    assert dep_d.is_available()
    assert root_c.is_available()


def test_kernel_object_requires_metadata_at_least_as_new_as_object(tmp_path):
    CompilationArtifact._instances.clear()

    source = FakeArtifact.new(tmp_path / "kernel.cc")
    source.path.write_text("// kernel", encoding="utf-8")
    obj = FakeArtifact.new(tmp_path / "kernel.o", depends=[source])

    # Reuse the real KernelObjectArtifact availability contract.
    from iron.common.compilation import KernelObjectArtifact

    CompilationArtifact._instances.pop(obj.path.absolute())
    obj = KernelObjectArtifact.new(obj.path, depends=[source])
    obj.path.write_text("object", encoding="utf-8")
    obj.metadata_path().write_text(
        '{\n  "artifact_type": "KernelObjectArtifact",\n  "extra_flags": [],\n  "rename_symbols": [],\n  "source_paths": ["%s"]\n}\n'
        % str(source.path.absolute()),
        encoding="utf-8",
    )
    assert obj.is_available()

    current_time = time.time() + 1.0
    os.utime(obj.path, (current_time, current_time))
    assert not obj.is_available()

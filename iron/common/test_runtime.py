#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from iron.common import aie_base as aie_base_module
from iron.common import aie_context as aie_context_module
from iron.common.aie_base import AIEOperatorBase


class _FakeBO:
    def __init__(self, size: int):
        self._storage = bytearray(size)
        self.sync_calls: list[str] = []

    def write(self, data, offset):
        data_bytes = bytes(np.asarray(data, dtype=np.uint8))
        self._storage[offset : offset + len(data_bytes)] = data_bytes

    def map(self):
        return memoryview(self._storage)

    def size(self):
        return len(self._storage)

    def sync(self, direction):
        self.sync_calls.append(direction)


class _FakeKernelRun:
    def __init__(self, completed_state):
        self._completed_state = completed_state

    def wait(self):
        return self._completed_state


class _FakeKernel:
    def __init__(self, completed_state):
        self._completed_state = completed_state
        self.calls = []

    def __call__(self, opcode, insts_bo, insts_len, *bos):
        self.calls.append((opcode, insts_bo, insts_len, bos))
        return _FakeKernelRun(self._completed_state)


class _FakeRunlist:
    def __init__(self):
        self.executed = 0
        self.waited = 0

    def execute(self):
        self.executed += 1

    def wait(self):
        self.waited += 1


class _FakePyxrt:
    class xclBOSyncDirection:
        XCL_BO_SYNC_BO_TO_DEVICE = "to_device"
        XCL_BO_SYNC_BO_FROM_DEVICE = "from_device"

    class ert_cmd_state:
        ERT_CMD_STATE_COMPLETED = "completed"

    class bo:
        host_only = object()

        def __new__(cls, device, size, flags, bank):
            del device, flags, bank
            return _FakeBO(size)


class _FakeContext:
    def __init__(self):
        self.static_data_pool = {}
        self.operators = []

    def register_operator(self, operator, skip_add_to_list=False):
        operator.context = self
        if not skip_add_to_list:
            self.operators.append(operator)


class _DummyOperator(AIEOperatorBase):
    def set_up_artifacts(self):
        pass

    def set_up_runtime(self):
        pass


def test_run_kernel_once_syncs_only_dirty_inputs_and_output(monkeypatch):
    monkeypatch.setattr(aie_base_module, "pyxrt", _FakePyxrt)

    context = _FakeContext()
    operator = _DummyOperator(context=context)
    operator.buffers = {"input": 8, "weights": 8, "output": 8}
    operator.buffer_static_data = {"weights": b"static"}
    operator.buffer_bos = {
        "input": _FakeBO(8),
        "weights": _FakeBO(8),
        "output": _FakeBO(8),
    }
    kernel = _FakeKernel(_FakePyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED)
    insts_bo = _FakeBO(16)
    operator.xrt_kernels = {
        "gemm": (object(), kernel, insts_bo, 4),
    }

    operator.write_buffer("input", np.arange(8, dtype=np.uint8))
    operator.write_buffer("output", np.zeros(8, dtype=np.uint8))

    operator.run_kernel_once("gemm", "input", "weights", "output")

    assert operator.buffer_bos["input"].sync_calls == [
        _FakePyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    ]
    assert operator.buffer_bos["weights"].sync_calls == []
    assert operator.buffer_bos["output"].sync_calls == [
        _FakePyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
        _FakePyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE,
    ]
    assert insts_bo.sync_calls == [
        _FakePyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    ]

    operator.run_kernel_once("gemm", "input", "weights", "output")

    assert operator.buffer_bos["input"].sync_calls == [
        _FakePyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    ]
    assert operator.buffer_bos["output"].sync_calls == [
        _FakePyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
        _FakePyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE,
        _FakePyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE,
    ]
    assert insts_bo.sync_calls == [
        _FakePyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    ]


def test_run_runlist_syncs_only_declared_inputs_and_outputs(monkeypatch):
    monkeypatch.setattr(aie_base_module, "pyxrt", _FakePyxrt)

    context = _FakeContext()
    operator = _DummyOperator(context=context)
    operator.buffers = {"input": 8, "tmp": 8, "output": 8}
    operator.buffer_bos = {
        "input": _FakeBO(8),
        "tmp": _FakeBO(8),
        "output": _FakeBO(8),
    }
    operator.runlist = [
        ("k1", "input", "tmp"),
        ("k2", "tmp", "output"),
    ]
    operator.device_input_buffer_names = ("input",)
    operator.host_output_buffer_names = ("output",)
    operator.xrt_runlist = _FakeRunlist()

    operator.write_buffer("input", np.arange(8, dtype=np.uint8))
    operator.write_buffer("output", np.zeros(8, dtype=np.uint8))

    operator.run_runlist()

    assert operator.buffer_bos["input"].sync_calls == [
        _FakePyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    ]
    assert operator.buffer_bos["tmp"].sync_calls == []
    assert operator.buffer_bos["output"].sync_calls == [
        _FakePyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE,
        _FakePyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE,
    ]
    assert operator.xrt_runlist.executed == 1
    assert operator.xrt_runlist.waited == 1


def test_prepare_runtime_syncs_static_and_instruction_bos_once(monkeypatch):
    monkeypatch.setattr(aie_context_module, "pyxrt", _FakePyxrt)
    monkeypatch.setattr(aie_base_module, "pyxrt", _FakePyxrt)

    class _FakeDeviceManager:
        def __init__(self):
            self.device = object()

        def get_kernel_handle(self, xclbin_path, kernel_name, insts_path):
            del xclbin_path, kernel_name, insts_path
            return SimpleNamespace(
                context=object(),
                kernel=_FakeKernel(_FakePyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED),
                insts_bo=_FakeBO(32),
                insts=b"insts",
            )

        def reset(self):
            pass

    monkeypatch.setattr(aie_context_module, "AIEDeviceManager", _FakeDeviceManager)

    class _PreparedOperator(AIEOperatorBase):
        def __init__(self, context):
            self.device_input_buffer_names = ("input",)
            self.host_output_buffer_names = ("output",)
            super().__init__(context=context)

        def set_up_artifacts(self):
            pass

        def set_up_runtime(self):
            xclbin = SimpleNamespace(
                path=Path("dummy.xclbin"), kernel_name="dummy_kernel"
            )
            insts = SimpleNamespace(path=Path("dummy.bin"))
            self.add_kernel("dummy", xclbin, "dummy_kernel", insts)
            self.add_buffer("input", 8, dtype=np.uint8)
            self.add_buffer(
                "weights",
                8,
                dtype=np.uint8,
                static_data=np.arange(8, dtype=np.uint8),
            )
            self.add_buffer("output", 8, dtype=np.uint8)
            self.add_to_runlist("dummy", "input", "weights", "output")

    context = aie_context_module.AIEContext(use_runlist=False)
    operator = _PreparedOperator(context=context)

    context.prepare_runtime()

    weights_bo = operator.buffer_bos["weights"]
    insts_bo = operator.xrt_kernels["dummy"][2]
    assert weights_bo.sync_calls == [
        _FakePyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    ]
    assert insts_bo.sync_calls == [
        _FakePyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
    ]

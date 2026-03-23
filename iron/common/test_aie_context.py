from pathlib import Path

import iron.common.aie_context as aie_context_module
from iron.common.compilation import InstsBinArtifact, XclbinArtifact


class _DummyHandle:
    def __init__(self, context, kernel, insts_bo, insts_len):
        self.context = context
        self.kernel = kernel
        self.insts_bo = insts_bo
        self.insts = bytes(insts_len)


class _DummyDeviceManager:
    def __init__(self):
        self.device = object()
        self.runtime = None
        self.calls = []
        self._shared_context = object()

    def get_kernel_handle(self, xclbin_path, kernel_name, insts_path):
        self.calls.append((xclbin_path, kernel_name, insts_path))
        return _DummyHandle(
            context=self._shared_context,
            kernel=object(),
            insts_bo=object(),
            insts_len=len(insts_path),
        )


class _DummyOperator:
    def __init__(self, xclbin, insts):
        self.kernels = {"gemm": (xclbin, "shared_kernel", insts)}
        self.buffers = {}
        self.buffer_static_data = {}
        self.buffer_aliases = {}
        self.runlist = []
        self.buffer_bos = {}
        self.xrt_kernels = {}
        self.xrt_runlist = None

    def set_up_runtime(self):
        return None


def test_prepare_runtime_reuses_kernel_object_for_shared_xclbin(monkeypatch, tmp_path):
    dummy_dm = _DummyDeviceManager()
    monkeypatch.setattr(
        aie_context_module,
        "AIEDeviceManager",
        lambda: dummy_dm,
    )

    ctx = aie_context_module.AIEContext(use_runlist=False)

    xclbin = XclbinArtifact.new(tmp_path / "shared.xclbin", depends=[])
    insts_a = InstsBinArtifact.new(tmp_path / "a.bin", depends=[])
    insts_b = InstsBinArtifact.new(tmp_path / "b.bin", depends=[])

    op_a = _DummyOperator(xclbin, insts_a)
    op_b = _DummyOperator(xclbin, insts_b)
    ctx.operators = [op_a, op_b]

    ctx.prepare_runtime(describe_runtime=False)

    a_ctx, a_kernel, a_insts_bo, a_insts_len = op_a.xrt_kernels["gemm"]
    b_ctx, b_kernel, b_insts_bo, b_insts_len = op_b.xrt_kernels["gemm"]

    assert a_ctx is b_ctx
    assert a_kernel is b_kernel
    assert a_insts_bo is not b_insts_bo
    assert a_insts_len == len(insts_a.path.as_posix())
    assert b_insts_len == len(insts_b.path.as_posix())
    assert len(dummy_dm.calls) == 2

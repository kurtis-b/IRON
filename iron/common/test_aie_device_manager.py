from types import SimpleNamespace

import iron.common.aie_device_manager as aie_device_manager_module


def test_get_kernel_handle_caches_identical_requests(monkeypatch):
    load_calls = []

    fake_runtime = SimpleNamespace(
        _device=object(),
        device=lambda: SimpleNamespace(resolve=lambda: SimpleNamespace(name="fake")),
        load=lambda npu_kernel: load_calls.append(npu_kernel) or object(),
    )

    monkeypatch.setattr(
        aie_device_manager_module,
        "DefaultNPURuntime",
        fake_runtime,
    )
    monkeypatch.setattr(
        aie_device_manager_module,
        "NPUKernel",
        lambda *, xclbin_path, insts_path, kernel_name: (
            xclbin_path,
            insts_path,
            kernel_name,
        ),
    )
    aie_device_manager_module.AIEDeviceManager._instance = None

    manager = aie_device_manager_module.AIEDeviceManager()

    first = manager.get_kernel_handle("shared.xclbin", "kernel", "a.insts.bin")
    second = manager.get_kernel_handle("shared.xclbin", "kernel", "a.insts.bin")
    third = manager.get_kernel_handle("shared.xclbin", "kernel", "b.insts.bin")

    assert first is second
    assert first is not third
    assert load_calls == [
        ("shared.xclbin", "a.insts.bin", "kernel"),
        ("shared.xclbin", "b.insts.bin", "kernel"),
    ]


def test_reset_recreates_runtime_and_clears_handle_cache(monkeypatch):
    cleanup_calls = []
    unregister_calls = []

    class FakeRuntime:
        next_device_id = 0

        def __init__(self):
            FakeRuntime.next_device_id += 1
            self.device_id = FakeRuntime.next_device_id
            self._device = object()
            self._cleanup_entry = lambda: None
            self._cleanup_insts_entry = lambda: None

        def device(self):
            return SimpleNamespace(resolve=lambda: SimpleNamespace(name="fake"))

        def load(self, npu_kernel):
            return object()

        def cleanup(self):
            cleanup_calls.append(self.device_id)

    monkeypatch.setattr(
        aie_device_manager_module,
        "DefaultNPURuntime",
        FakeRuntime(),
    )
    monkeypatch.setattr(
        aie_device_manager_module.atexit,
        "unregister",
        lambda callback: unregister_calls.append(callback),
    )
    aie_device_manager_module.AIEDeviceManager._instance = None

    manager = aie_device_manager_module.AIEDeviceManager()
    original_runtime = manager.runtime
    manager._kernel_handle_cache[("x", "k", "i")] = object()

    manager.reset()

    assert cleanup_calls == [original_runtime.device_id]
    assert len(unregister_calls) == 2
    assert manager.runtime is not original_runtime
    assert manager.device is manager.runtime._device
    assert manager.device_type.resolve().name == "fake"
    assert manager._kernel_handle_cache == {}

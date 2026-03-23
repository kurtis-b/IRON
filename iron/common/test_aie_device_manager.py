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

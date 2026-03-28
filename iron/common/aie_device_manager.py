# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Global AIE Device Manager for resource sharing and cleanup
"""

import atexit
import logging
import pyxrt
from aie.utils import DefaultNPURuntime
from aie.utils.npukernel import NPUKernel


class AIEDeviceManager:
    """Singleton manager for AIE XRT resources"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self.runtime = DefaultNPURuntime
        # Expose device for AIEContext buffer allocation
        # Accessing protected member _device as AIEContext needs pyxrt.device
        self.device = self.runtime._device
        self.device_type = self.runtime.device()
        if not hasattr(self, "_kernel_handle_cache"):
            self._kernel_handle_cache = {}

    def _instantiate_runtime(self):
        """Create a fresh host-runtime instance when possible."""
        runtime_cls = type(self.runtime)
        try:
            candidate = runtime_cls()
        except Exception:
            return self.runtime

        if not all(hasattr(candidate, attr) for attr in ("_device", "device", "load")):
            return self.runtime
        return candidate

    def get_kernel_handle(self, xclbin_path: str, kernel_name: str, insts_path: str):
        """Get kernel handle using HostRuntime"""
        cache_key = (xclbin_path, kernel_name, insts_path)
        handle = self._kernel_handle_cache.get(cache_key)
        if handle is None:
            npu_kernel = NPUKernel(
                xclbin_path=xclbin_path, insts_path=insts_path, kernel_name=kernel_name
            )
            handle = self.runtime.load(npu_kernel)
            self._kernel_handle_cache[cache_key] = handle
        return handle

    def device_str(self) -> str:
        return self.device_type.resolve().name

    def cleanup(self):
        """Clean up all XRT resources"""
        # HostRuntime handles cleanup
        pass

    def reset(self):
        """Reset host-runtime state between tests or isolated benchmark runs."""
        runtime = getattr(self, "runtime", None)
        if runtime is not None:
            for attr in ("_cleanup_entry", "_cleanup_insts_entry"):
                cleanup_entry = getattr(runtime, attr, None)
                if cleanup_entry is not None:
                    try:
                        atexit.unregister(cleanup_entry)
                    except Exception:
                        pass

            cleanup = getattr(runtime, "cleanup", None)
            if callable(cleanup):
                try:
                    cleanup()
                except Exception:
                    logging.exception("Failed to clean up AIE host runtime cleanly")

        self.runtime = self._instantiate_runtime()
        self.device = self.runtime._device
        self.device_type = self.runtime.device()
        self._kernel_handle_cache = {}

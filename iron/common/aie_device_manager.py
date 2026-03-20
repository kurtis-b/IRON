# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Global AIE Device Manager for resource sharing and cleanup
"""

import logging
import os
import sys
import gc
from pathlib import Path
from typing import Dict, Optional, Any
import pyxrt
import aie.utils
from aie.utils.npukernel import NPUKernel
from aie.iron.device import NPU1, NPU2


class AIEDeviceManager:
    """Singleton manager for AIE XRT resources"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._refresh_runtime()

    def _refresh_runtime(self):
        # Resolve the runtime lazily so reset() can recreate the mlir_aie singleton.
        self.runtime = aie.utils.DefaultNPURuntime
        # Expose device for AIEContext buffer allocation
        # Accessing protected member _device as AIEContext needs pyxrt.device
        self.device = self.runtime._device
        self.device_type = self.runtime.device()

    def get_kernel_handle(self, xclbin_path: str, kernel_name: str, insts_path: str):
        """Get kernel handle using HostRuntime"""
        npu_kernel = NPUKernel(
            xclbin_path=xclbin_path, insts_path=insts_path, kernel_name=kernel_name
        )
        return self.runtime.load(npu_kernel)

    def device_str(self) -> str:
        return self.device_type.resolve().name

    def cleanup(self):
        """Clean up all XRT resources"""
        runtime = getattr(self, "runtime", None)
        if runtime is not None and hasattr(runtime, "cleanup"):
            runtime.cleanup()

    def reset(self):
        """Reset the cached XRT runtime and reacquire the device."""
        runtime = getattr(self, "runtime", None)
        if runtime is not None and hasattr(runtime, "cleanup"):
            runtime.cleanup()

        # Drop the cached mlir_aie runtime so the next access recreates contexts.
        if hasattr(aie.utils, "_DefaultNPURuntime"):
            aie.utils._DefaultNPURuntime = None

        gc.collect()
        self._refresh_runtime()

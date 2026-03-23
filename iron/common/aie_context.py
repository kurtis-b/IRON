# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import gc
import logging
from pathlib import Path
import os

from .aie_device_manager import AIEDeviceManager, pyxrt
from . import compilation as comp
import aie.utils.config


class AIEContext:
    """Context for managing AIE operator compilation and runtime state"""

    def __init__(self, use_runlist=True, mlir_verbose=None):
        self.operators = []
        self.static_data_pool = {}
        self.device_manager = AIEDeviceManager()
        self.base_dir = Path(__file__).parent.parent.parent
        self.build_dir = Path(os.getcwd()) / "build"
        self.mlir_aie_dir = Path(aie.utils.config.root_path())
        self.peano_dir = Path(aie.utils.config.peano_install_dir())
        # Disable the XRT runlist sacrifices performance by executing kernels individually as separate xclbin invocations for easier debugging (can tell which part of runlist execution failed)
        self.use_runlist = use_runlist
        self.mlir_verbose = bool(mlir_verbose)
        self._runtime_prepared = False

    def register_operator(self, operator, skip_add_to_list=False):
        """Register an operator with this context"""
        if self._runtime_prepared:
            raise RuntimeError("Cannot register operators after runtime is prepared")
        operator.context = self
        if not skip_add_to_list:
            self.operators.append(operator)

    def compile_all(self):
        """Compile all registered operators"""
        self.build_dir.mkdir(parents=True, exist_ok=True)
        for op in self.operators:
            op.compile()

    def prepare_runtime(self, describe_runtime=True):
        """Setup XRT runtime for all registered operators"""
        if self._runtime_prepared:
            return

        if describe_runtime:
            for op in self.operators:
                op.set_up_runtime()

        # Pools of preallocated buffer objects; each buffer object is allocated
        # once at program start and then reused across operators where possible.
        bo_pools = {}
        page_sz = 4096
        get_pool_sz = lambda x: (x + page_sz - 1) // page_sz * page_sz

        # Allocate static buffers first
        for buffer_data in self.static_data_pool:
            logging.debug(
                f"Allocating static buffer with size {len(buffer_data)} bytes."
            )
            bo = pyxrt.bo(
                self.device_manager.device,
                len(buffer_data),
                pyxrt.bo.host_only,
                0x10000,
            )
            bo.write(np.frombuffer(buffer_data, dtype=np.uint8), 0)
            self.static_data_pool[buffer_data] = bo

        shared_kernel_handles = {}

        for op in self.operators:
            if len(op.kernels) == 0:
                continue

            logging.info(f"Preparing runtime for AIE operator: {op.__class__.__name__}")

            # Set up kernels
            lazy_kernel_loading = getattr(op, "lazy_kernel_loading", False)
            if not lazy_kernel_loading:
                for kernel_name, (
                    xclbin,
                    xclbin_kernel_name,
                    insts,
                ) in op.kernels.items():
                    handle = self.device_manager.get_kernel_handle(
                        str(xclbin.path), xclbin_kernel_name, str(insts.path)
                    )
                    shared_key = (str(xclbin.path), xclbin_kernel_name)
                    if shared_key in shared_kernel_handles:
                        context, kernel = shared_kernel_handles[shared_key]
                    else:
                        context, kernel = handle.context, handle.kernel
                        shared_kernel_handles[shared_key] = (context, kernel)
                    op.xrt_kernels[kernel_name] = (
                        context,
                        kernel,
                        handle.insts_bo,
                        len(handle.insts),
                    )
            else:
                op.xrt_kernels = {}

            # If multiple buffers (of the same binned size) are used in the
            # same kernel invocation OR across different invocations with shared
            # buffers, they require separate allocations.
            conflicting_buffers = {}  # map buffer -> {set of conflicting buffers}
            buffer_to_runlist_entries = {}  # map buffer -> set of runlist entry indices

            # First pass: track which buffers appear in which runlist entries
            for idx, (kernel, *args) in enumerate(op.runlist):
                for arg in args:
                    buffer_to_runlist_entries.setdefault(arg, set()).add(idx)

            # Second pass: determine conflicts
            for idx, (kernel, *args) in enumerate(op.runlist):
                for arg in args:
                    if arg in op.buffer_static_data:
                        # Static buffers never conflict
                        continue
                    pool_sz = get_pool_sz(op.buffers[arg])

                    # Buffers conflict if they're in the same runlist entry
                    conflicting_args = {
                        a for a in args if get_pool_sz(op.buffers[a]) == pool_sz
                    } - {arg}

                    # Also conflict with buffers in other runlist entries that share
                    # a buffer with this entry
                    for other_arg in args:
                        if other_arg == arg:
                            continue
                        for other_idx in buffer_to_runlist_entries.get(
                            other_arg, set()
                        ):
                            if other_idx != idx:
                                _, *other_args = op.runlist[other_idx]
                                conflicting_args.update(
                                    {
                                        a
                                        for a in other_args
                                        if get_pool_sz(op.buffers[a]) == pool_sz
                                        and a != arg
                                    }
                                )

                    conflicting_buffers[arg] = conflicting_buffers.get(
                        arg, set()
                    ).union(conflicting_args)

            # Allocate buffers
            buffer_allocations = {}
            alias_map = op.buffer_aliases
            for buffer_name, buffer_min_size in op.buffers.items():
                if buffer_name in alias_map:
                    # Alias BOs are resolved after target buffers are allocated.
                    continue
                if buffer_name in op.buffer_static_data:
                    static_data = op.buffer_static_data[buffer_name]
                    op.buffer_bos[buffer_name] = self.static_data_pool[static_data]
                    continue

                alloc_pool = get_pool_sz(buffer_min_size)
                alloc_idx = 0
                for conflict in conflicting_buffers.get(buffer_name, set()):
                    if conflict not in buffer_allocations:
                        continue
                    conflict_pool, conflict_idx = buffer_allocations[conflict]
                    alloc_idx = max(alloc_idx, conflict_idx + 1)

                assert 0 <= alloc_idx < len(bo_pools.get(alloc_pool, [])) + 1
                if alloc_idx == len(bo_pools.get(alloc_pool, [])):
                    bo = pyxrt.bo(
                        self.device_manager.device,
                        alloc_pool,
                        pyxrt.bo.host_only,
                        0x10000,
                    )
                    bo_pools.setdefault(alloc_pool, []).append(bo)

                buffer_allocations[buffer_name] = (alloc_pool, alloc_idx)
                op.buffer_bos[buffer_name] = bo_pools[alloc_pool][alloc_idx]

            # Resolve alias BOs after concrete allocations.
            for alias_name, target_name in alias_map.items():
                if target_name not in op.buffer_bos:
                    raise RuntimeError(
                        f"Alias target buffer '{target_name}' not allocated for alias '{alias_name}'."
                    )
                op.buffer_bos[alias_name] = op.buffer_bos[target_name]

            # Setup runlist
            if lazy_kernel_loading or not op.xrt_kernels:
                op.xrt_runlist = None
            else:
                context = next(iter(op.xrt_kernels.values()))[0]
                if self.use_runlist:
                    if any(
                        op.xrt_kernels[kernel_name][0] != context
                        for (kernel_name, *_) in op.runlist
                    ):
                        op.xrt_runlist = None
                        continue
                    op.xrt_runlist = pyxrt.runlist(context)
                    for i, (kernel_name, *buffer_args) in enumerate(op.runlist):
                        this_context, xrt_kernel, insts_bo, insts_len = op.xrt_kernels[
                            kernel_name
                        ]
                        assert this_context == context
                        opcode = 3
                        run = pyxrt.run(xrt_kernel)
                        run.set_arg(0, opcode)
                        run.set_arg(1, insts_bo)
                        run.set_arg(2, insts_len)
                        for j, buffer_arg in enumerate(buffer_args):
                            run.set_arg(j + 3, op.buffer_bos[buffer_arg])
                        op.xrt_runlist.add(run)
                else:
                    op.xrt_runlist = None

        # Log allocation info
        bo_count = sum(len(pool) for pool in bo_pools.values())
        bo_footprint = sum(len(pool) * pool_sz for pool_sz, pool in bo_pools.items())
        logging.info(
            f"Allocated {bo_count} total buffer objects with a total memory footprint of "
            + (
                f"{bo_footprint//1024//1024} MiB."
                if bo_footprint >= 1024 * 1024
                else f"{bo_footprint//1024} KiB."
            )
        )
        static_data_footprint = sum(len(data) for data in self.static_data_pool)
        logging.info(
            f"Allocated {len(self.static_data_pool)} static buffers with a total memory footprint of "
            + (
                f"{static_data_footprint//1024//1024} MiB."
                if static_data_footprint >= 1024 * 1024
                else f"{static_data_footprint//1024} KiB."
            )
        )

        self._runtime_prepared = True

    def reset_runtime(self):
        """Drop prepared XRT runtime state so it can be reloaded."""
        if not self._runtime_prepared:
            return

        for op in self.operators:
            op.buffer_bos = {}
            op.xrt_kernels = {}
            op.xrt_runlist = None

        # Drop Python references to XRT objects before the host runtime's
        # atexit cleanup runs. Calling CachedXRTRuntime.cleanup() here causes
        # a second cleanup pass later and can double-free when multiple staged
        # xclbins have been loaded in one process.
        gc.collect()

        self._runtime_prepared = False

# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import os
from pathlib import Path
from abc import ABC, abstractmethod
import logging
import time
import torch
from ml_dtypes import bfloat16

import aie.utils.config
from . import compilation as comp
from .aie_context import AIEContext
from .aie_device_manager import AIEDeviceManager, pyxrt
from .utils import numpy_to_torch, torch_to_numpy


class AIEOperatorBase(ABC):
    """Base class for AIE-accelerated operations"""

    @classmethod
    def get_default_context(cls):
        """One global 'default' context if none is specified"""
        if not hasattr(AIEOperatorBase, "_default_context"):
            AIEOperatorBase._default_context = AIEContext()
        return AIEOperatorBase._default_context

    def __init__(self, context=None, skip_add_to_list=False):
        # skip_add_to_list is for cases where the operator is a runlist implementation, which puts
        # togegther a sequence of operators, which also execute this constructor. Those operators should not register
        # themselves in the context's runlist again, otherwise the runtime setup will create kernels for them again.
        self.artifacts = (
            []
        )  # CompilationArtifact objects are uniqued within the context
        self.kernels = {}  # Name -> (xclbin_path, xclbin_kernel_name, insts_path)
        self.buffers = {}  # Name -> required buffer size in bytes
        self.buffer_static_data = {}
        self.runlist = (
            []
        )  # List of (kernel_name, buffers_name, buffer_name...), will be executed in sequence
        if not hasattr(self, "device_input_buffer_names"):
            self.device_input_buffer_names = ()
        if not hasattr(self, "host_output_buffer_names"):
            self.host_output_buffer_names = ()

        # AIE runtime state
        self.buffer_bos = {}  # Buffer name -> buffer object
        self.xrt_kernels = (
            {}
        )  # Kernel name -> (XRT context, XRT kernel object, instruction buffer object, instruction length)
        self.xrt_runlist = None
        self._buffer_dirty_to_device = {}
        self._insts_bos_synced_to_device = set()

        if context is None:
            context = self.get_default_context()
        context.register_operator(self, skip_add_to_list=skip_add_to_list)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def add_kernel(
        self,
        name: str,
        xclbin_artifact: comp.XclbinArtifact,
        xclbin_kernel_name: str,
        insts_artifact: comp.InstsBinArtifact,
    ):
        assert name not in self.kernels
        self.kernels[name] = (xclbin_artifact, xclbin_kernel_name, insts_artifact)

    def add_buffer(self, name, count, dtype=bfloat16, static_data=None):
        assert name not in self.buffers
        self.buffers[name] = count * np.dtype(dtype).itemsize
        if static_data is not None:
            assert (
                static_data.nbytes <= self.buffers[name]
            ), f"Static data for buffer {name} exceeds allocated size: expected {self.buffers[name]} bytes, got {static_data.nbytes} bytes."
            static_data_bytes = static_data.flatten().view(np.uint8).tobytes()
            if static_data_bytes not in self.context.static_data_pool:
                self.context.static_data_pool[static_data_bytes] = None
            self.buffer_static_data[name] = next(
                k
                for k, v in self.context.static_data_pool.items()
                if k == static_data_bytes
            )

    def add_to_runlist(self, kernel_name, *args):
        if kernel_name not in self.kernels:
            raise RuntimeError(f"No such kernel: {kernel_name}")
        for arg in args:
            if arg not in self.buffers:
                raise RuntimeError(f"No such buffer: {arg}")
        self.runlist.append((kernel_name, *args))

    def get_bo(self, buffer_name):
        return self.buffer_bos[buffer_name]

    def _normalize_runtime_buffer_names(self, buffer_names):
        ordered = []
        seen = set()
        for buffer_name in buffer_names:
            if buffer_name not in self.buffer_bos:
                raise RuntimeError(
                    f"Runtime sync metadata refers to unknown buffer '{buffer_name}'"
                )
            if buffer_name in seen:
                continue
            seen.add(buffer_name)
            ordered.append(buffer_name)
        return tuple(ordered)

    def _configured_device_input_buffers(self):
        if self.device_input_buffer_names:
            return self._normalize_runtime_buffer_names(self.device_input_buffer_names)
        return self._normalize_runtime_buffer_names(
            [
                buffer_arg
                for _, *buffer_args in self.runlist
                for buffer_arg in buffer_args
                if buffer_arg not in self.buffer_static_data
            ]
        )

    def _runtime_runlist_buffers(self):
        return self._normalize_runtime_buffer_names(
            [
                buffer_arg
                for _, *buffer_args in self.runlist
                for buffer_arg in buffer_args
                if buffer_arg not in self.buffer_static_data
            ]
        )

    def _configured_host_output_buffers(self):
        if self.host_output_buffer_names:
            return self._normalize_runtime_buffer_names(self.host_output_buffer_names)
        return self._normalize_runtime_buffer_names(
            [
                buffer_arg
                for _, *buffer_args in self.runlist
                for buffer_arg in buffer_args
                if buffer_arg not in self.buffer_static_data
            ]
        )

    def _runtime_output_buffers_for_call(self, buffer_args):
        configured_outputs = set(self._configured_host_output_buffers())
        matched_outputs = []
        seen = set()
        for buffer_arg in buffer_args:
            if buffer_arg not in configured_outputs or buffer_arg in seen:
                continue
            seen.add(buffer_arg)
            matched_outputs.append(buffer_arg)
        if matched_outputs:
            return tuple(matched_outputs)
        if not buffer_args:
            return ()
        return (buffer_args[-1],)

    def _mark_buffer_dirty_to_device(self, buffer_name):
        if buffer_name in self.buffer_static_data:
            return
        self._buffer_dirty_to_device[buffer_name] = True

    def _sync_buffer_to_device_if_needed(self, buffer_name):
        if buffer_name in self.buffer_static_data:
            return
        if not self._buffer_dirty_to_device.get(buffer_name, False):
            return
        self.buffer_bos[buffer_name].sync(
            pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
        )
        self._buffer_dirty_to_device[buffer_name] = False

    def _sync_buffer_from_device(self, buffer_name):
        if buffer_name in self.buffer_static_data:
            return
        self.buffer_bos[buffer_name].sync(
            pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
        )
        self._buffer_dirty_to_device[buffer_name] = False

    def _sync_insts_bo_to_device_if_needed(self, insts_bo):
        insts_bo_id = id(insts_bo)
        if insts_bo_id in self._insts_bos_synced_to_device:
            return
        insts_bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        self._insts_bos_synced_to_device.add(insts_bo_id)

    def buffer_view(self, buffer_name, shape, dtype=bfloat16):
        """Return a mapped numpy view into an existing BO without copying."""
        mv = self.get_bo(buffer_name).map()
        return np.frombuffer(mv, dtype=dtype, count=np.prod(shape)).reshape(shape)

    def read_buffer(self, buffer_name, shape, copy=False, dtype=bfloat16):
        """Read buffer and return values as a numpy array"""
        if copy:
            return np.array(
                self.buffer_view(buffer_name, shape, dtype=dtype),
                copy=True,
            )

        return self.buffer_view(buffer_name, shape, dtype=dtype)

    def read_buffer_as_torch(self, buffer_name, shape, dtype=bfloat16):
        # Detach the returned tensor from the live BO mapping. Several staged
        # encoder paths segfault when Torch keeps viewing XRT-backed memory
        # after the kernel returns.
        return numpy_to_torch(
            self.read_buffer(buffer_name, shape, copy=True, dtype=dtype)
        )

    def write_buffer(self, buffer_name, array):
        """Write buffer from a numpy array into a XRT buffer object"""
        if buffer_name in self.buffer_static_data:
            raise RuntimeError(f"Cannot write to static buffer: {buffer_name}")

        # Normalize the source
        if isinstance(array, torch.Tensor):
            src = torch_to_numpy(array)
        else:
            src = np.asarray(array)

        # Create a flattened 1D byte view of the source
        src_bytes = src.ravel().view(np.uint8)

        bo = self.get_bo(buffer_name)
        mv = bo.map()  # byte accessible memory view
        # Interpret the buffer as a 1-dimensional array
        dst_bytes = np.frombuffer(mv, dtype=np.uint8, count=bo.size())

        # The BO is an existing array, so copyto() can be called, which doesn't create a new array
        np.copyto(dst_bytes[: src_bytes.size], src_bytes, casting="no")
        self._mark_buffer_dirty_to_device(buffer_name)

    @abstractmethod
    def set_up_artifacts(self):
        """
        Subclasses should overwrite this method to set up their required dependenices and runtime runlist, kernels and buffers with calls to add_artifacts(), add_kernel(), add_buffer(), and add_to_runlist().
        Note: This method should only *describe* the required artifacts and runtime buffers, and not yet do any computation or compilation.
        Compilation will be handled automatically based on the provided description.
        """
        pass

    @abstractmethod
    def set_up_runtime(self):
        pass

    def compile(self, dry_run=None):
        """
        Set up the operator and compile any necessary artifacts.
        Subclasses are expected to overwrite set_up(); they may register any artifacts that they need to be compiled there.
        """
        context = self.context
        self.set_up_artifacts()
        self._move_artifact_paths()
        work_list = comp.get_work_list(self.artifacts)
        compilation_rules = [
            comp.GenerateMLIRFromPythonCompilationRule(dry_run=dry_run),
            comp.PeanoCompilationRule(
                context.peano_dir, context.mlir_aie_dir, dry_run=dry_run
            ),
            comp.ArchiveCompilationRule(context.peano_dir, dry_run=dry_run),
            comp.AieccCompilationRule(
                context.build_dir,
                context.peano_dir,
                context.mlir_aie_dir,
                dry_run=dry_run,
            ),
        ]
        if work_list:
            logging.info(
                f"Compiling {len(work_list)} new artifacts for AIE operator {self.__class__.__name__}: {', '.join(str(artifact.path.name) for artifact in work_list)}"
            )
        comp.compile(compilation_rules, self.artifacts)

    def add_artifacts(self, artifacts):
        self.artifacts.extend(artifacts)

    def _move_artifact_paths(self):
        """Make all artifacts paths point into the build directory (source artifacts into the ironclad source directory). This doesn't phyisically move files; this function is called before artifact generation."""
        context = self.context
        todo = self.artifacts.copy()
        while todo:
            artifact = todo[0]
            todo.pop(0)
            if isinstance(artifact, comp.SourceArtifact):
                artifact.set_path(context.base_dir / artifact.path)
            else:
                artifact.set_path(context.build_dir / artifact.path)
            todo.extend(artifact.depends)

    def run_runlist(self):
        elapsed = 0.0
        if self.xrt_runlist is None:
            # Execute as separate xclbin kernel invocations
            for i, (kernel_name, *buffer_args) in enumerate(self.runlist):
                elapsed += self.run_kernel_once(kernel_name, *buffer_args)
            if self.host_output_buffer_names:
                for buffer_name in self._configured_host_output_buffers():
                    self._sync_buffer_from_device(buffer_name)
        else:
            for buffer_name in self._runtime_runlist_buffers():
                self._sync_buffer_to_device_if_needed(buffer_name)
            start = time.perf_counter()
            self.xrt_runlist.execute()
            self.xrt_runlist.wait()
            stop = time.perf_counter()
            for buffer_name in self._configured_host_output_buffers():
                self._sync_buffer_from_device(buffer_name)
            elapsed = stop - start
        return elapsed

    def run_kernel_once(self, kernel_name, *buffer_args):
        _, xrt_kernel, insts_bo, insts_len = self.xrt_kernels[kernel_name]
        self._sync_insts_bo_to_device_if_needed(insts_bo)
        output_buffer_args = self._runtime_output_buffers_for_call(buffer_args)
        for buffer_arg in buffer_args:
            self._sync_buffer_to_device_if_needed(buffer_arg)
        bos = [self.buffer_bos[buffer_arg] for buffer_arg in buffer_args]
        start = time.perf_counter()
        result = xrt_kernel(3, insts_bo, insts_len, *bos).wait()
        stop = time.perf_counter()
        if result != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED:
            raise RuntimeError(
                f"Kernel {kernel_name} did not complete correctly: {result}"
            )
        for buffer_arg in output_buffer_args:
            self._sync_buffer_from_device(buffer_arg)
        return stop - start


class AIEOperatorConstraintError(RuntimeError):
    pass

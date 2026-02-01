# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ml_dtypes import bfloat16
from pathlib import Path
import numpy as np
import argparse
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker, Buffer
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2, Tile
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron.controlflow import range_
from aie.helpers.util import np_ndarray_type_get_shape
import aie.dialects.index as index
from aie.dialects.aiex import *


"""
This design computes weighted layer norm + eltwise add on AIE cores.
The data movement is written in such a way that the outputs can be used
to feed directly to a GEMM core. The s parameter is the microkernel
col dim layout parameter for the GEMM core that would consume the output of 
this design. It's ASSUMED that m == r of the microkernel row dim layout
for GEMM. The output stream through shim DMA is organized in tiles
of size (m x k), where within each tile the data is contiguously laid
out in the (m x s) subtiles required for the mmul API call for the AIE.
"""


def my_weighted_layer_norm(
    dev,
    M,
    K,
    m,
    k,
    s,
    num_columns,
    weight_file_path,
    kernel_archive_path,
    trace_size,
):
    static_weights = np.load(weight_file_path)
    if static_weights.shape[0] != K:
        raise ValueError(
            "Static weights length does not match the specified weight length"
        )
    assert M % (num_columns * m) == 0, "M must be multiple of num_columns * m"
    assert K % k == 0, "K must be multiple of k"
    assert k % s == 0, "k must be multiple of s"
    M_div_m = M // m
    K_div_k = K // k  # Will be used as the obj fifo depth
    iters_per_core = M_div_m // num_columns
    dtype = bfloat16
    # Define tensor types
    tensor_ty = np.ndarray[(M * K,), np.dtype[dtype]]
    weights_ty = np.ndarray[(K,), np.dtype[dtype]]
    tile_ty = np.ndarray[(m * K,), np.dtype[dtype]]
    out_ty = np.ndarray[(m * k,), np.dtype[dtype]]
    fifodepth = 2  # For double buffering

    # AIE-array data movement with object fifos
    of_in1s = [
        ObjectFifo(tile_ty, name=f"in1_{i}", depth=fifodepth)
        for i in range(num_columns)
    ]
    of_in2s = [
        ObjectFifo(tile_ty, name=f"in2_{i}", depth=fifodepth)
        for i in range(num_columns)
    ]

    of_out1s_l1l2 = [None] * num_columns
    of_out1s_l2l3 = [None] * num_columns
    for i in range(num_columns):
        dims_to_stream_out = [(k // s, s), (m, k), (s, 1)]
        of_out1s_l1l2[i] = ObjectFifo(
            out_ty,
            name=f"outl1l2_{i}",
            depth=1,
        )
        of_out1s_l2l3[i] = (
            of_out1s_l1l2[i]
            .cons(depth=fifodepth)
            .forward(
                obj_type=out_ty,
                depth=fifodepth,
                name=f"outl2l3_{i}",
                dims_to_stream=dims_to_stream_out,
            )
        )

    # AIE Core Function declaration
    fused_add_layer_norm_kernel = Kernel(
        "fused_add_layer_norm_1outs",
        kernel_archive_path,
        [tile_ty, tile_ty, weights_ty, tile_ty, np.int32, np.int32],
    )
    mem_copy_kernel = Kernel(
        "ln_passThroughTile_in",
        kernel_archive_path,
        [tile_ty, out_ty, np.int32, np.int32, np.int32, np.int32],
    )

    # Define a task that will run on a compute tile
    def core_body(
        of_in1, of_in2, weights, internal_out, of_out1, fused_add_layer_norm, copy
    ):
        # Number of sub-vector "tile" iterations
        for row_idx in range_(iters_per_core):
            elem_in1 = of_in1.acquire(1)
            elem_in2 = of_in2.acquire(1)
            fused_add_layer_norm(elem_in1, elem_in2, weights, internal_out, K, m)
            of_in1.release(1)
            of_in2.release(1)
            for col_idx in range_(K_div_k):
                col_i32 = index.casts(T.i32(), col_idx)
                elem_out = of_out1.acquire(1)
                copy(internal_out, elem_out, K, k, m, col_i32)
                of_out1.release(1)

    # Create workers to run the task on compute tiles,
    # one core for layer norm and another pipelined to do eltwise mul
    my_workers = []
    for i in range(num_columns):
        weights_buffer = Buffer(
            type=weights_ty,
            initial_value=static_weights,
            name=f"weights_buffer_{i}",
        )
        out_buffer = Buffer(
            type=tile_ty,
            name=f"internal_out_buffer_{i}",
        )
        my_workers.append(
            Worker(
                core_body,
                [
                    of_in1s[i].cons(),
                    of_in2s[i].cons(),
                    weights_buffer,
                    out_buffer,
                    of_out1s_l1l2[i].prod(),
                    fused_add_layer_norm_kernel,
                    mem_copy_kernel,
                ],
                stack_size=0xF00,
            )
        )

    # Create a TensorAccessPattern for each core
    # to describe the data movement
    # The pattern chops the data in equal chunks
    # and moves them in parallel across the cores.
    taps = [
        TensorAccessPattern(
            (1, M * K),
            iters_per_core * m * K * i,
            [1, 1, 1, iters_per_core * m * K],
            [0, 0, 0, 1],
        )
        for i in range(num_columns)
    ]

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (A, B, C):
        rt.start(*my_workers)

        # Initialize a group for parallel drain tasks, with fill resources free'd when drains complete.
        tg = rt.task_group()

        # Fill the input objectFIFOs with data
        for i in range(num_columns):
            rt.fill(
                of_in1s[i].prod(),
                A,
                taps[i],
                task_group=tg,
            )
            rt.fill(
                of_in2s[i].prod(),
                B,
                taps[i],
                task_group=tg,
            )
        # Drain the output objectFIFOs with data
        for i in range(num_columns):
            rt.drain(
                of_out1s_l2l3[i].cons(),
                C,
                taps[i],
                wait=True,
                task_group=tg,
            )
        rt.finish_task_group(tg)

    # Place program components (assign them resources on the device) and generate an MLIR module
    return Program(dev, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":

    def str_to_device(device: str):
        if device == "npu":
            return NPU1()
        elif device == "npu2":
            return NPU2()
        else:
            raise ValueError(f"Device name {device} is unknown.")

    p = argparse.ArgumentParser()
    # Parse command line arguments

    # Device name is required to select the AIE device: npu or npu2
    p.add_argument(
        "-d",
        "--dev",
        required=True,
        dest="device",
        help="AIE Device",
        type=str_to_device,
    )
    p.add_argument(
        "-M",
        required=True,
        dest="M",
        help="Number of rows",
    )
    p.add_argument(
        "-K",
        required=True,
        dest="K",
        help="Number of cols, also the weight vector length",
    )
    p.add_argument(
        "-m",
        required=True,
        dest="m",
        help="Number of tile rows",
    )
    p.add_argument(
        "-k",
        required=True,
        dest="k",
        help="Number of tile cols",
    )
    p.add_argument(
        "-s",
        required=True,
        dest="s",
        help="Number of subtile cols",
    )
    # Number of columns is required to define the number of columns to be used
    # It must be less than or equal to 4 for npu and 8 for npu2
    p.add_argument(
        "-co", "--columns", required=True, dest="cols", help="Number of columns"
    )
    # Static weights
    p.add_argument(
        "-wf",
        "--weight-file",
        required=True,
        default=None,
        dest="weight_file",
        help="Static weight values are expected to be saved in a numpy .npy file so they can be preloaded to buffers at compile time",
    )
    # Kernel archive
    p.add_argument(
        "-ka",
        "--kernel-archive",
        required=False,
        dest="kernel_archive",
        help="Kernel archive file path",
    )
    # Trace Size
    p.add_argument(
        "-ts", "--trace-size", required=True, dest="trace_size", help="Trace size"
    )
    p.add_argument(
        "--output-file-path",
        "-o",
        type=str,
        help="Output file path for the generated MLIR module",
    )

    opts = p.parse_args(sys.argv[1:])

    M = int(opts.M)
    K = int(opts.K)
    m = int(opts.m)
    k = int(opts.k)
    s = int(opts.s)
    columns = int(opts.cols)
    dev = opts.device  # Now this is already a device object!

    # Validate columns based on device type
    if isinstance(dev, NPU1) and columns > 4:
        raise ValueError("[ERROR] NPU device cannot allocate more than 4 columns")
    elif isinstance(dev, NPU2) and columns > 8:
        raise ValueError("[ERROR] NPU2 device cannot allocate more than 8 columns")

    # For add and norm: cores = columns * 2 (2 cores are used per set of inputs, one for each stage in pipeline)
    if (M % columns) != 0:
        print(
            "transfer rows ("
            + str(M)
            + ") must be a multiple of columns ("
            + str(columns)
            + ")"
        )
        raise ValueError
    # Load static weights
    weight_file_path = Path(opts.weight_file)
    trace_size = int(opts.trace_size) if opts.trace_size is not None else 0

    module = my_weighted_layer_norm(
        dev,
        M,
        K,
        m,
        k,
        s,
        columns,
        weight_file_path,
        opts.kernel_archive,
        trace_size,
    )

    output_file_path = Path(opts.output_file_path)

    with open(output_file_path, "w") as f:
        f.write(str(module))

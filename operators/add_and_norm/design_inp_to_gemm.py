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


"""
This design computes weighted layer norm + eltwise add on AIE cores.
The data movement is written in such a way that the outputs can be used
to feed directly to a GEMM core. The t parameter is the microkernel
col dim layout parameter for the GEMM core that would consume the output of 
this design. It's ASSUMED that m == r of the microkernel row dim layout
for GEMM. The output stream through shim DMA is organized in tiles
of size (m x k), where within each tile the data is contiguously laid
out in the (m x t) subtiles required for the mmul API call for the AIE.
"""


def my_weighted_layer_norm(
    dev,
    M,
    K,
    m,
    k,
    t,
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
    assert k % t == 0, "k must be multiple of t"
    M_div_m = M // m
    K_div_k = K // k  # Will be used as the obj fifo depth
    iters_per_core = M_div_m // num_columns
    dtype = bfloat16
    # Define tensor types
    tensor_ty = np.ndarray[(M * K,), np.dtype[dtype]]
    weights_ty = np.ndarray[(K,), np.dtype[dtype]]
    tile_ty = np.ndarray[(m * k,), np.dtype[dtype]]
    fifodepth = 2  # For double buffering

    # AIE-array data movement with object fifos
    of_in1s = [
        ObjectFifo(tile_ty, name=f"in1_{i}", depth=K_div_k * fifodepth)
        for i in range(num_columns)
    ]
    of_in2s = [
        ObjectFifo(tile_ty, name=f"in2_{i}", depth=K_div_k * fifodepth)
        for i in range(num_columns)
    ]
    dims_to_stream_out = [(k // t, t), (m, k), (t, 1)]
    of_out1s = [
        ObjectFifo(
            tile_ty,
            name=f"out2_{i}",
            depth=K_div_k * fifodepth,
            dims_to_stream=dims_to_stream_out,
        )
        for i in range(num_columns)
    ]

    # AIE Core Function declaration
    layer_norm_kernel = Kernel(
        "layer_norm", kernel_archive_path, [tile_ty, tile_ty, np.int32, np.int32]
    )
    eltwise_mul_kernel = Kernel(
        "eltwise_mul_bf16_vector",
        kernel_archive_path,
        [tile_ty, weights_ty, tile_ty, np.int32, np.int32],
    )
    eltwise_add_kernel = Kernel(
        "eltwise_add_bf16_vector",
        kernel_archive_path,
        [tile_ty, tile_ty, tile_ty, np.int32],
    )

    # Define a task that will run on a compute tile
    def core_body(of_in1, of_in2, of_out1, weights, layer_norm, eltwise_mul, add):
        # Number of sub-vector "tile" iterations
        for _ in range_(iters_per_core):
            elem_in1 = of_in1.acquire(K_div_k)
            elem_in2 = of_in2.acquire(K_div_k)
            elem_out = of_out1.acquire(K_div_k)
            # layer_norm(elem_in1, elem_out, K, m)
            # eltwise_mul(elem_out, weights, elem_out, K, m)
            # add(elem_out, elem_in2, elem_out, K * m)
            of_in1.release(K_div_k)
            of_in2.release(K_div_k)
            of_out1.release(K_div_k)

    # Create workers to run the task on compute tiles,
    # one core for layer norm and another pipelined to do eltwise mul
    my_workers = []
    for i in range(num_columns):
        weights_buffer = Buffer(
            type=weights_ty,
            initial_value=static_weights,
            name=f"weights_buffer_{i}",
        )
        my_workers.append(
            Worker(
                core_body,
                [
                    of_in1s[i].cons(),
                    of_in2s[i].cons(),
                    of_out1s[i].prod(),
                    weights_buffer,
                    layer_norm_kernel,
                    eltwise_mul_kernel,
                    eltwise_add_kernel,
                ],
            )
        )

    # Create a TensorAccessPattern for each core
    # to describe the data movement
    # The pattern chops the data in equal chunks
    # and moves them in parallel across the cores.
    taps = [
        TensorAccessPattern(
            (1, M * K),
            iters_per_core * K * i,
            [1, 1, 1, iters_per_core * K],
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
                of_out1s[i].cons(),
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
        "-t",
        required=True,
        dest="t",
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
    t = int(opts.t)
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
        t,
        columns,
        weight_file_path,
        opts.kernel_archive,
        trace_size,
    )

    output_file_path = Path(opts.output_file_path)

    with open(output_file_path, "w") as f:
        f.write(str(module))

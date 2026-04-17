# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU1, NPU2
from aie.iron.placers import SequentialPlacer


def my_eltwise_mul_broadcast_scalar(
    dev,
    num_elements,
    num_columns,
    num_channels,
    tile_size,
    scalar_broadcast,
    trace_size,
):
    del trace_size
    per_tile_elements = 4096 if tile_size > 4096 else tile_size
    total_cores = num_columns * num_channels
    n = per_tile_elements * total_cores
    if num_elements % n != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of {n}."
        )
    N_div_n = num_elements // n
    chunk = num_elements // total_cores
    dtype = bfloat16

    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(per_tile_elements,), np.dtype[dtype]]
    scalar_broadcasted_ty = np.ndarray[(16,), np.dtype[dtype]]

    of_in1s = [ObjectFifo(tile_ty, name=f"in1_{i}") for i in range(total_cores)]
    of_outs = [ObjectFifo(tile_ty, name=f"out_{i}") for i in range(total_cores)]

    eltwise_mul_bf16_vector = Kernel(
        "eltwise_mul_bf16_broadcasted_scalar",
        "mul.o",
        [tile_ty, scalar_broadcasted_ty, tile_ty, np.int32],
    )

    def core_body(of_in1, scalar, of_out, eltwise_mul):
        for _ in range_(N_div_n):
            elem_in1 = of_in1.acquire(1)
            elem_out = of_out.acquire(1)
            eltwise_mul(elem_in1, scalar, elem_out, per_tile_elements)
            of_in1.release(1)
            of_out.release(1)

    my_workers = []
    for i in range(total_cores):
        scalar_buffer = Buffer(
            type=scalar_broadcasted_ty,
            initial_value=np.full(16, scalar_broadcast, dtype=dtype),
            name=f"scalar_buffer_{i}",
        )
        my_workers.append(
            Worker(
                core_body,
                [
                    of_in1s[i].cons(),
                    scalar_buffer,
                    of_outs[i].prod(),
                    eltwise_mul_bf16_vector,
                ],
            )
        )

    taps = [
        TensorAccessPattern(
            (1, num_elements),
            chunk * i,
            [1, 1, 1, chunk],
            [0, 0, 0, 1],
        )
        for i in range(total_cores)
    ]

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (A, C):
        rt.start(*my_workers)
        tg = rt.task_group()
        for i in range(total_cores):
            rt.fill(of_in1s[i].prod(), A, taps[i], task_group=tg)
        for i in range(total_cores):
            rt.drain(of_outs[i].cons(), C, taps[i], wait=True, task_group=tg)
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())


def my_eltwise_mul(dev, num_elements, num_columns, num_channels, tile_size, trace_size):
    del num_channels, trace_size
    per_tile_elements = 4096 if tile_size > 4096 else tile_size
    n = per_tile_elements * num_columns
    if num_elements % n != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of {n}."
        )
    N_div_n = num_elements // n
    chunk = num_elements // num_columns
    dtype = bfloat16

    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(per_tile_elements,), np.dtype[dtype]]

    of_in1s = [ObjectFifo(tile_ty, name=f"in1_{i}") for i in range(num_columns)]
    of_in2s = [ObjectFifo(tile_ty, name=f"in2_{i}") for i in range(num_columns)]
    of_outs = [ObjectFifo(tile_ty, name=f"out_{i}") for i in range(num_columns)]

    eltwise_mul_bf16_vector = Kernel(
        "eltwise_mul_bf16_vector", "mul.o", [tile_ty, tile_ty, tile_ty, np.int32]
    )

    def core_body(of_in1, of_in2, of_out, eltwise_mul):
        for _ in range_(N_div_n):
            elem_in1 = of_in1.acquire(1)
            elem_in2 = of_in2.acquire(1)
            elem_out = of_out.acquire(1)
            eltwise_mul(elem_in1, elem_in2, elem_out, per_tile_elements)
            of_in1.release(1)
            of_in2.release(1)
            of_out.release(1)

    my_workers = [
        Worker(
            core_body,
            [
                of_in1s[i].cons(),
                of_in2s[i].cons(),
                of_outs[i].prod(),
                eltwise_mul_bf16_vector,
            ],
        )
        for i in range(num_columns)
    ]

    taps = [
        TensorAccessPattern(
            (1, num_elements),
            chunk * i,
            [1, 1, 1, chunk],
            [0, 0, 0, 1],
        )
        for i in range(num_columns)
    ]

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (A, B, C):
        rt.start(*my_workers)
        tg = rt.task_group()
        for i in range(num_columns):
            rt.fill(of_in1s[i].prod(), A, taps[i], task_group=tg)
            rt.fill(of_in2s[i].prod(), B, taps[i], task_group=tg)
        for i in range(num_columns):
            rt.drain(of_outs[i].cons(), C, taps[i], wait=True, task_group=tg)
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())


if __name__ == "__main__":

    def str_to_device(device: str):
        if device == "npu":
            return NPU1()
        if device == "npu2":
            return NPU2()
        raise ValueError(f"Device name {device} is unknown.")

    p = argparse.ArgumentParser()
    p.add_argument(
        "-d",
        "--dev",
        required=True,
        dest="device",
        help="AIE Device",
        type=str_to_device,
    )
    p.add_argument("-l", "--length", required=True, dest="length", help="Transfer size")
    p.add_argument(
        "-co", "--columns", required=True, dest="cols", help="Number of columns"
    )
    p.add_argument(
        "-ch", "--channels", required=True, dest="chans", help="Number of channels"
    )
    p.add_argument(
        "-ts",
        "--tile-size",
        required=False,
        dest="tile_size",
        default="1024",
        help="Tile size (elements per tile)",
    )
    p.add_argument(
        "-sb",
        "--scalar-broadcast",
        required=False,
        dest="scalar_broadcast",
        default=None,
        help="Scalar to broadcast to vectors for vector-scalar eltwise mul",
    )
    p.add_argument(
        "-t", "--trace-size", required=True, dest="trace_size", help="Trace size"
    )
    p.add_argument(
        "--output-file-path",
        "-o",
        type=str,
        help="Output file path for the generated MLIR module",
    )

    opts = p.parse_args(sys.argv[1:])
    length = int(opts.length)
    columns = int(opts.cols)
    dev = opts.device

    if isinstance(dev, NPU1) and columns > 4:
        raise ValueError("[ERROR] NPU device cannot allocate more than 4 columns")
    if isinstance(dev, NPU2) and columns > 8:
        raise ValueError("[ERROR] NPU2 device cannot allocate more than 8 columns")

    channels = int(opts.chans)
    if channels < 1 or channels > 2:
        raise ValueError("Number of channels must be 1 or 2")
    tile_size = int(opts.tile_size)
    if length % (tile_size * columns) != 0:
        raise ValueError(
            f"transfer size ({length}) must be a multiple of {tile_size * columns} (tile_size * columns)"
        )
    trace_size = int(opts.trace_size) if opts.trace_size is not None else 0
    scalar_broadcast = (
        None if opts.scalar_broadcast is None else float(opts.scalar_broadcast)
    )

    if scalar_broadcast is None:
        module = my_eltwise_mul(dev, length, columns, channels, tile_size, trace_size)
    else:
        module = my_eltwise_mul_broadcast_scalar(
            dev, length, columns, channels, tile_size, scalar_broadcast, trace_size
        )

    output_file_path = Path(opts.output_file_path)
    with open(output_file_path, "w") as f:
        f.write(str(module))

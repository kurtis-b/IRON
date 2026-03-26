# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import argparse
from pathlib import Path
import sys

import numpy as np
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2
from aie.iron import Buffer
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.helpers.dialects.scf import _for as range_
from ml_dtypes import bfloat16


def _softmax_short_rows(
    dev,
    num_elements,
    num_columns,
    num_channels,
    tile_size,
    kernel_object_name,
):
    per_tile_elements = tile_size
    n = per_tile_elements * num_columns
    if num_elements % n != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of {n}."
        )
    N_div_n = num_elements // n
    chunk = num_elements // num_columns // num_channels
    dtype = bfloat16

    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(per_tile_elements,), np.dtype[dtype]]

    fifodepth = 1 if tile_size > 4096 else 2

    of_in1s = [
        ObjectFifo(tile_ty, name=f"in1_{i}_{j}", depth=fifodepth)
        for i in range(num_columns)
        for j in range(num_channels)
    ]
    of_outs = [
        ObjectFifo(tile_ty, name=f"out_{i}_{j}", depth=fifodepth)
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    softmax_kernel = Kernel(
        "softmax_bf16",
        kernel_object_name,
        [tile_ty, tile_ty, np.int32],
    )

    def core_body(of_in1, of_out, softmax_kernel):
        for _ in range_(N_div_n):
            elem_in1 = of_in1.acquire(1)
            elem_out = of_out.acquire(1)
            softmax_kernel(elem_in1, elem_out, per_tile_elements)
            of_in1.release(1)
            of_out.release(1)

    my_workers = [
        Worker(
            core_body,
            [
                of_in1s[i * num_channels + j].cons(),
                of_outs[i * num_channels + j].prod(),
                softmax_kernel,
            ],
        )
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    taps = [
        TensorAccessPattern(
            (1, num_elements),
            chunk * i * num_channels + chunk * j,
            [1, 1, 1, chunk],
            [0, 0, 0, 1],
        )
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (A, C):
        rt.start(*my_workers)

        tg = rt.task_group()

        for i in range(num_columns):
            for j in range(num_channels):
                rt.fill(
                    of_in1s[i * num_channels + j].prod(),
                    A,
                    taps[i * num_channels + j],
                    task_group=tg,
                )
        for i in range(num_columns):
            for j in range(num_channels):
                rt.drain(
                    of_outs[i * num_channels + j].cons(),
                    C,
                    taps[i * num_channels + j],
                    wait=True,
                    task_group=tg,
                )
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())


def _softmax_long_rows(
    dev,
    num_elements,
    num_columns,
    num_channels,
    row_width,
    row_chunk_size,
    kernel_object_name,
):
    if num_elements % row_width != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be divisible by row_width "
            f"({row_width})."
        )
    if row_width % row_chunk_size != 0:
        raise ValueError(
            f"row_width ({row_width}) must be divisible by row_chunk_size "
            f"({row_chunk_size})."
        )

    rows = num_elements // row_width
    worker_count = num_columns * num_channels
    if rows % worker_count != 0:
        raise ValueError(
            f"rows ({rows}) must be divisible by worker_count ({worker_count})."
        )

    rows_per_worker = rows // worker_count
    num_row_chunks = row_width // row_chunk_size
    if num_row_chunks != 2:
        raise ValueError(
            "Long-row softmax currently supports exactly two chunks per row "
            f"(got num_row_chunks={num_row_chunks})."
        )
    chunk = num_elements // num_columns // num_channels
    dtype = bfloat16

    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(row_chunk_size,), np.dtype[dtype]]
    scale_ty = np.ndarray[(4,), np.dtype[dtype]]

    of_in1s = [
        ObjectFifo(tile_ty, name=f"in1_{i}_{j}", depth=1)
        for i in range(num_columns)
        for j in range(num_channels)
    ]
    of_outs = [
        ObjectFifo(tile_ty, name=f"out_{i}_{j}", depth=1)
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    init_scale_kernel = Kernel(
        "init_softmax_scale_buffer",
        kernel_object_name,
        [scale_ty, np.int32],
    )
    copy_kernel = Kernel(
        "copy_softmax_scale_bf16",
        kernel_object_name,
        [tile_ty, tile_ty, np.int32],
    )
    partial_softmax_kernel = Kernel(
        "partial_softmax_rows_bf16",
        kernel_object_name,
        [tile_ty, tile_ty, scale_ty, np.int32, np.int32],
    )
    normalize_softmax_kernel = Kernel(
        "normalize_softmax_rows_bf16",
        kernel_object_name,
        [tile_ty, scale_ty, tile_ty, np.int32, np.int32],
    )

    def core_body(
        of_in1,
        of_out,
        init_scale_kernel,
        copy_kernel,
        partial_softmax_kernel,
        normalize_softmax_kernel,
        scale_buffer,
        scratch_buffer,
    ):
        for _ in range_(rows_per_worker):
            init_scale_kernel(scale_buffer, 1)
            first_chunk = of_in1.acquire(1)
            partial_softmax_kernel(
                first_chunk,
                first_chunk,
                scale_buffer,
                row_chunk_size,
                1,
            )
            copy_kernel(first_chunk, scratch_buffer, row_chunk_size)
            of_in1.release(1)

            second_chunk = of_in1.acquire(1)
            partial_softmax_kernel(
                second_chunk,
                second_chunk,
                scale_buffer,
                row_chunk_size,
                1,
            )

            first_out = of_out.acquire(1)
            normalize_softmax_kernel(
                scratch_buffer,
                scale_buffer,
                first_out,
                row_chunk_size,
                1,
            )
            of_out.release(1)

            second_out = of_out.acquire(1)
            normalize_softmax_kernel(
                second_chunk,
                scale_buffer,
                second_out,
                row_chunk_size,
                1,
            )
            of_in1.release(1)
            of_out.release(1)

    my_workers = [
        Worker(
            core_body,
            [
                of_in1s[i * num_channels + j].cons(),
                of_outs[i * num_channels + j].prod(),
                init_scale_kernel,
                copy_kernel,
                partial_softmax_kernel,
                normalize_softmax_kernel,
                Buffer(
                    initial_value=np.zeros(shape=(4,), dtype=dtype),
                    name=f"scale_buffer_{i}_{j}",
                ),
                Buffer(
                    initial_value=np.zeros(shape=(row_chunk_size,), dtype=dtype),
                    name=f"scratch_buffer_{i}_{j}",
                ),
            ],
        )
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    taps = [
        TensorAccessPattern(
            (1, num_elements),
            chunk * i * num_channels + chunk * j,
            [1, 1, 1, chunk],
            [0, 0, 0, 1],
        )
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (A, C):
        rt.start(*my_workers)
        tg = rt.task_group()
        for i in range(num_columns):
            for j in range(num_channels):
                rt.fill(
                    of_in1s[i * num_channels + j].prod(),
                    A,
                    taps[i * num_channels + j],
                    task_group=tg,
                )
        for i in range(num_columns):
            for j in range(num_channels):
                rt.drain(
                    of_outs[i * num_channels + j].cons(),
                    C,
                    taps[i * num_channels + j],
                    wait=True,
                    task_group=tg,
                )
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program(SequentialPlacer())


def softmax(
    dev,
    num_elements,
    num_columns,
    num_channels,
    trace_size,
    row_width,
    tile_size,
    kernel_object_name="softmax.o",
):
    if row_width <= tile_size:
        return _softmax_short_rows(
            dev,
            num_elements,
            num_columns,
            num_channels,
            tile_size,
            kernel_object_name,
        )
    return _softmax_long_rows(
        dev,
        num_elements,
        num_columns,
        num_channels,
        row_width,
        tile_size,
        kernel_object_name,
    )


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
    # Transfer size is required to define the size of the data to be transferred
    # It must be a multiple of 1024 and divisible by the number of columns and 2 channels per column
    p.add_argument("-l", "--length", required=True, dest="length", help="Transfer size")
    # Number of columns is required to define the number of columns to be used
    # It must be less than or equal to 4 for npu and 8 for npu2
    p.add_argument(
        "-co", "--columns", required=True, dest="cols", help="Number of columns"
    )
    # Number of channels is required to define the number of channels to be used
    # It must be 1 or 2
    p.add_argument(
        "-ch", "--channels", required=True, dest="chans", help="Number of channels"
    )
    # Tile size (columns per tile) - defaults to 1024 for backward compatibility
    p.add_argument(
        "-ts",
        "--tile-size",
        required=False,
        dest="tile_size",
        default="1024",
        help="Tile size (columns per tile)",
    )
    # Trace Size
    p.add_argument(
        "-tr", "--trace-size", required=True, dest="trace_size", help="Trace size"
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
    dev = opts.device  # Now this is already a device object!

    # Validate columns based on device type
    if isinstance(dev, NPU1) and columns > 4:
        raise ValueError("[ERROR] NPU device cannot allocate more than 4 columns")
    elif isinstance(dev, NPU2) and columns > 8:
        raise ValueError("[ERROR] NPU2 device cannot allocate more than 8 columns")

    channels = int(opts.chans)
    if channels < 1 or channels > 2:
        raise ValueError("Number of channels must be 1 or 2")
    tile_size = int(opts.tile_size)
    if ((length % tile_size) % columns % channels) != 0:
        print(
            "transfer size ("
            + str(length)
            + ") must be a multiple of "
            + str(tile_size)
            + " and divisible by the number of columns and 2 channels per column"
        )
        raise ValueError
    trace_size = int(opts.trace_size) if opts.trace_size is not None else 0

    module = softmax(dev, length, columns, channels, trace_size, tile_size, tile_size)

    output_file_path = Path(opts.output_file_path)

    with open(output_file_path, "w") as f:
        f.write(str(module))

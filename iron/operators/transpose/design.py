# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ml_dtypes import bfloat16
from pathlib import Path
import numpy as np
import argparse
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron.controlflow import range_
from aie.helpers.util import np_ndarray_type_get_shape

DMA_BD_MAX_STRIDE = 1_048_576
DMA_BD_MAX_RANK = 3


def _normalize_dma_tap(
    tap: TensorAccessPattern, *, max_stride: int = DMA_BD_MAX_STRIDE
) -> TensorAccessPattern:
    keep_dims = [
        i for i, (size, stride) in enumerate(zip(tap.sizes, tap.strides)) if size != 1
    ]
    if not keep_dims:
        keep_dims = [len(tap.sizes) - 1]
    if len(keep_dims) == len(tap.sizes):
        return tap
    return TensorAccessPattern(
        tap.tensor_dims,
        offset=int(tap.offset),
        sizes=[tap.sizes[i] for i in keep_dims],
        strides=[tap.strides[i] for i in keep_dims],
    )


def _expand_dma_tap_for_bd_limits(
    tap: TensorAccessPattern,
    *,
    max_stride: int = DMA_BD_MAX_STRIDE,
    max_rank: int = DMA_BD_MAX_RANK,
) -> tuple[list[TensorAccessPattern], bool]:
    def _is_legal(current: TensorAccessPattern) -> bool:
        return len(current.sizes) <= max_rank and all(
            stride <= max_stride for stride in current.strides
        )

    def _expand_order_preserving(
        current: TensorAccessPattern,
    ) -> list[TensorAccessPattern]:
        current = _normalize_dma_tap(current, max_stride=max_stride)
        if _is_legal(current):
            return [current]
        split_size = current.sizes[0]
        if split_size == 1:
            raise ValueError(
                "Cannot legalize DMA tap while preserving transfer order: "
                f"sizes={current.sizes}, strides={current.strides}"
            )
        expanded: list[TensorAccessPattern] = []
        for split_index in range(split_size):
            split_sizes = [1, *current.sizes[1:]]
            split_offset = int(current.offset) + split_index * current.strides[0]
            expanded.extend(
                _expand_order_preserving(
                    TensorAccessPattern(
                        current.tensor_dims,
                        offset=split_offset,
                        sizes=split_sizes,
                        strides=current.strides,
                    )
                )
            )
        return expanded

    normalized = _normalize_dma_tap(tap, max_stride=max_stride)
    planned_taps = _expand_order_preserving(normalized)
    return planned_taps, len(planned_taps) > 1


def _emit_dma_transfer(
    rt: Runtime,
    *,
    fifo_handle,
    runtime_buffer,
    tap: TensorAccessPattern,
    task_group,
    is_fill: bool,
    wait: bool = False,
) -> None:
    planned_taps, was_split = _expand_dma_tap_for_bd_limits(tap)
    task_wait = wait or (was_split and not is_fill)
    for planned_tap in planned_taps:
        if is_fill:
            rt.fill(
                fifo_handle,
                runtime_buffer,
                tap=planned_tap,
                task_group=task_group,
                wait=task_wait,
            )
        else:
            rt.drain(
                fifo_handle,
                runtime_buffer,
                tap=planned_tap,
                task_group=task_group,
                wait=task_wait,
            )


def shuffle_transpose(dev, M, N, num_columns, num_channels, trace_size, m, n, s):
    num_elements = M * N
    per_tile_elements = m * n
    dtype = bfloat16

    if M % m != 0:
        raise ValueError(f"Matrix rows ({M}) must be a multiple of {m}.")
    if N % n != 0:
        raise ValueError(f"Matrix columns ({N}) must be a multiple of {n}.")
    if m % s != 0:
        raise ValueError(f"AIE tile rows ({m}) must be a multiple of {s}.")
    if n % s != 0:
        raise ValueError(f"AIE tile columns ({n}) must be a multiple of {s}.")
    if per_tile_elements > 8192:
        raise ValueError(
            f"Kernel tile size {per_tile_elements} needs to be below 8192 to fit within data memory."
        )

    # Minimum tile sizes required by the two kernels
    if s == 4 and (m <= 4 or n <= 4):
        raise ValueError(f"Kernel tile {s} needs AIE tile rows > 4 and columns > 4.")
    if s == 8 and (m <= 16 or n <= 16):
        raise ValueError(f"Kernel tile {s} needs AIE tile rows > 16 and columns > 16.")

    # Define tensor types
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(per_tile_elements,), np.dtype[dtype]]

    fifodepth = 1 if per_tile_elements > 4096 else 2

    # Create a TensorAccessPattern for each channel
    # to describe the data movement
    # The pattern chops the data in equal chunks
    # and moves them in parallel across the columns
    # and channels. Partially transposes the input
    # data so that the kernel only needs to
    # transpose s*s-sized sub-tiles.
    taps_in_L3L2 = [
        TensorAccessPattern(
            (M, N),
            (M // num_channels) * j * N + (N // num_columns) * i,
            [M // num_channels // m, N // num_columns // n, m, n],
            [m * N, n, N, 1],
        )
        for i in range(num_columns)
        for j in range(num_channels)
    ]
    taps_in_L2L1 = [
        TensorAccessPattern(
            (M, N),
            (M // num_channels) * j * N + (N // num_columns) * i,
            [m // s, s, n // s, s],
            [s, m, s * m, 1],
        )
        for i in range(num_columns)
        for j in range(num_channels)
    ]
    taps_out_L1L3 = [
        TensorAccessPattern(
            (N, M),
            (N // num_columns) * i * M + (M // num_channels) * j,
            [M // num_channels // m, N // num_columns // n, n, m],
            [m, n * M, M, 1],
        )
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    # AIE-array data movement with object fifos
    of_in1s_L3L2 = [
        ObjectFifo(tile_ty, name=f"of_in1s_L3L2_{i}_{j}", depth=fifodepth)
        for i in range(num_columns)
        for j in range(num_channels)
    ]
    of_in1s_L2L1 = [
        of_in1s_L3L2[i * num_channels + j]
        .cons(dims_from_stream=taps_in_L2L1[i * num_channels + j].transformation_dims)
        .forward(obj_type=tile_ty, name=f"of_in1s_L2L1_{i}_{j}", depth=fifodepth)
        for i in range(num_columns)
        for j in range(num_channels)
    ]
    of_outs = [
        ObjectFifo(tile_ty, name=f"out_{i}_{j}", depth=fifodepth)
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    # AIE Core Function declaration
    transpose_kernel = Kernel(
        f"transpose_{s}x{s}", f"transpose_{m}x{n}.o", [tile_ty, tile_ty]
    )

    # Define a task that will run on a compute tile
    def core_body(of_in1, of_out, transpose_kernel):
        # Number of sub-matrix "tile" iterations
        for _ in range_(N // n // num_columns):
            for _ in range_(M // m // num_channels):
                elem_in1 = of_in1.acquire(1)
                elem_out = of_out.acquire(1)
                transpose_kernel(elem_in1, elem_out)
                of_out.release(1)
                of_in1.release(1)

    # Create a worker to run the task on a compute tile
    my_workers = [
        Worker(
            core_body,
            [
                of_in1s_L2L1[i * num_channels + j].cons(),
                of_outs[i * num_channels + j].prod(),
                transpose_kernel,
            ],
        )
        for i in range(num_columns)
        for j in range(num_channels)
    ]

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (A, C):
        rt.start(*my_workers)

        # Initialize a group for parallel drain tasks, with fill resources free'd when drains complete.
        tg = rt.task_group()

        # Fill the input objectFIFOs with data
        for i in range(num_columns):
            for j in range(num_channels):
                _emit_dma_transfer(
                    rt,
                    fifo_handle=of_in1s_L3L2[i * num_channels + j].prod(),
                    runtime_buffer=A,
                    tap=taps_in_L3L2[i * num_channels + j],
                    task_group=tg,
                    is_fill=True,
                )
        # Drain the output objectFIFOs with data
        for i in range(num_columns):
            for j in range(num_channels):
                _emit_dma_transfer(
                    rt,
                    fifo_handle=of_outs[i * num_channels + j].cons(),
                    runtime_buffer=C,
                    tap=taps_out_L1L3[i * num_channels + j],
                    task_group=tg,
                    is_fill=False,
                    wait=True,
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
    # Transfer size is required to define the size of the data to be transferred
    p.add_argument(
        "-M", "--workload-rows", required=True, dest="work_rows", help="Number of rows"
    )
    p.add_argument(
        "-N",
        "--workload-columns",
        required=True,
        dest="work_cols",
        help="Number of columns",
    )
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
    # Tile size
    p.add_argument(
        "-m", "--tile-rows", required=True, dest="tile_rows", help="Outer tile rows"
    )
    p.add_argument(
        "-n",
        "--tile-columns",
        required=True,
        dest="tile_cols",
        help="Outer tile columns",
    )
    p.add_argument(
        "-s",
        "--kernel-dim",
        required=True,
        choices=["4", "8"],
        dest="kernel_dim",
        help="Inner tile dimension (square)",
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

    M = int(opts.work_rows)
    N = int(opts.work_cols)
    columns = int(opts.cols)

    dev = opts.device  # Already a device object from str_to_device

    # Validate columns based on device type
    if isinstance(dev, NPU1) and columns > 4:
        raise ValueError("[ERROR] Device NPU cannot allocate more than 4 columns")
    elif isinstance(dev, NPU2) and columns > 8:
        raise ValueError("[ERROR] Device NPU2 cannot allocate more than 8 columns")

    channels = int(opts.chans)
    if channels < 1 or channels > 2:
        raise ValueError("Number of channels must be 1 or 2")
    m = int(opts.tile_rows)
    n = int(opts.tile_cols)
    s = int(opts.kernel_dim)
    if (((M * N) % (m * n)) % columns % channels) != 0:
        print(
            "transfer size ("
            + str(M * N)
            + ") must be a multiple of "
            + str(m * n)
            + f" and divisible by the number of columns ({columns}) and {channels} channels per column"
        )
        raise ValueError
    trace_size = int(opts.trace_size) if opts.trace_size is not None else 0

    module = shuffle_transpose(dev, M, N, columns, channels, trace_size, m, n, s)

    output_file_path = Path(opts.output_file_path)

    with open(output_file_path, "w") as f:
        f.write(str(module))

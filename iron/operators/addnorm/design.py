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

DATA_MEM_SIZE = 65536  # L1 size in bytes


def get_rows_to_process(num_elements, weight_length):
    # Determine per-tile elements based on weight_length
    for i in range(1, num_elements // weight_length):
        per_tile_elements = weight_length * i
        # Conservative per-core estimate for the add -> norm -> mul pipeline.
        if (per_tile_elements * 3 + weight_length) * 2 * 2 > DATA_MEM_SIZE:
            return i - 1
    return num_elements // weight_length


def my_weighted_layer_norm(
    dev,
    num_elements,
    num_columns,
    weight_length,
    weight_file_path,
    kernel_archive_path,
    trace_size,
):
    static_weights = np.load(weight_file_path)
    if static_weights.shape[0] != weight_length:
        raise ValueError(
            "Static weights length does not match the specified weight length"
        )
    per_tile_elements = weight_length
    rows_to_process = get_rows_to_process(num_elements, weight_length)
    max_worker_lanes_per_column = 2
    # Find a tile size multiple that divides num_elements and can be evenly
    # partitioned across the per-column worker lanes.
    input_tile_size = 0
    worker_lanes_per_column = 1
    for candidate_rows_to_process in range(rows_to_process, 0, -1):
        candidate_tile_size = per_tile_elements * candidate_rows_to_process
        candidate_n = candidate_tile_size * num_columns
        if num_elements % candidate_n != 0:
            continue
        candidate_worker_lanes = min(
            max_worker_lanes_per_column, candidate_rows_to_process
        )
        while (
            candidate_worker_lanes > 1
            and candidate_rows_to_process % candidate_worker_lanes != 0
        ):
            candidate_worker_lanes -= 1
        rows_to_process = candidate_rows_to_process
        input_tile_size = candidate_tile_size
        worker_lanes_per_column = candidate_worker_lanes
        n = candidate_n
        break
    if input_tile_size == 0:
        raise ValueError(
            f"Couldn't find tile size multiple for number of elements ({num_elements})"
        )
    N_div_n = num_elements // n
    chunk = num_elements // num_columns
    worker_rows_to_process = rows_to_process // worker_lanes_per_column
    worker_tile_size = per_tile_elements * worker_rows_to_process
    split_offsets = [worker_tile_size * i for i in range(worker_lanes_per_column)]
    dtype = bfloat16
    # Define tensor types
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    weights_ty = np.ndarray[(per_tile_elements,), np.dtype[dtype]]
    column_tile_ty = np.ndarray[(input_tile_size,), np.dtype[dtype]]
    worker_tile_ty = np.ndarray[(worker_tile_size,), np.dtype[dtype]]

    # Set fifodepth based on weight_length
    fifodepth = 1 if weight_length > 4096 else 2

    # AIE-array data movement with object fifos
    of_in1s = [
        ObjectFifo(column_tile_ty, name=f"in1_L3L2_{i}", depth=fifodepth)
        for i in range(num_columns)
    ]
    of_in2s = [
        ObjectFifo(column_tile_ty, name=f"in2_L3L2_{i}", depth=fifodepth)
        for i in range(num_columns)
    ]
    of_outs = [
        ObjectFifo(column_tile_ty, name=f"out_L2L3_{i}", depth=fifodepth)
        for i in range(num_columns)
    ]
    of_in1_l2l1s = [[None] * worker_lanes_per_column for _ in range(num_columns)]
    of_in2_l2l1s = [[None] * worker_lanes_per_column for _ in range(num_columns)]
    of_adds = [[None] * worker_lanes_per_column for _ in range(num_columns)]
    of_out_l1l2s = [[None] * worker_lanes_per_column for _ in range(num_columns)]

    # AIE Core Function declaration
    eltwise_add_kernel = Kernel(
        "eltwise_add_bf16_vector",
        kernel_archive_path,
        [worker_tile_ty, worker_tile_ty, worker_tile_ty, np.int32],
    )
    layer_norm_kernel = Kernel(
        "layer_norm_rows",
        kernel_archive_path,
        [worker_tile_ty, worker_tile_ty, np.int32, np.int32],
    )
    eltwise_mul_kernel = Kernel(
        "eltwise_mul_bf16_vector_rows",
        kernel_archive_path,
        [worker_tile_ty, weights_ty, worker_tile_ty, np.int32, np.int32],
    )

    # Define a task that will run on a compute tile
    def core_body_stg1(of_in1, of_in2, of_add, add):
        # Number of sub-vector "tile" iterations
        for _ in range_(N_div_n):
            elem_in1 = of_in1.acquire(1)
            elem_in2 = of_in2.acquire(1)
            elem_out = of_add.acquire(1)
            add(
                elem_in1,
                elem_in2,
                elem_out,
                per_tile_elements * worker_rows_to_process,
            )
            of_in1.release(1)
            of_in2.release(1)
            of_add.release(1)

    def core_body_stg2(of_add, weights, of_out, layer_norm, eltwise_mul):
        # Number of sub-vector "tile" iterations
        for _ in range_(N_div_n):
            elem_in = of_add.acquire(1)
            elem_out = of_out.acquire(1)
            layer_norm(elem_in, elem_out, per_tile_elements, worker_rows_to_process)
            # Reuse the normalized output tile so the weighted layer norm stage
            # multiplies normalized values instead of the raw add result.
            eltwise_mul(
                elem_out,
                weights,
                elem_out,
                per_tile_elements,
                worker_rows_to_process,
            )
            of_add.release(1)
            of_out.release(1)

    # Split each column-wide tile through the memtile into per-lane subtiles,
    # then join the lane outputs back before draining to host memory.
    my_workers = []
    for i in range(num_columns):
        of_in1_l2l1s[i] = (
            of_in1s[i]
            .cons()
            .split(
                split_offsets,
                obj_types=[worker_tile_ty] * worker_lanes_per_column,
                names=[
                    f"in1_L2L1_{i}_{lane}" for lane in range(worker_lanes_per_column)
                ],
                depths=[fifodepth] * worker_lanes_per_column,
                placement=Tile(i, 1),
            )
        )
        of_in2_l2l1s[i] = (
            of_in2s[i]
            .cons()
            .split(
                split_offsets,
                obj_types=[worker_tile_ty] * worker_lanes_per_column,
                names=[
                    f"in2_L2L1_{i}_{lane}" for lane in range(worker_lanes_per_column)
                ],
                depths=[fifodepth] * worker_lanes_per_column,
                placement=Tile(i, 1),
            )
        )
        of_out_l1l2s[i] = (
            of_outs[i]
            .prod()
            .join(
                split_offsets,
                obj_types=[worker_tile_ty] * worker_lanes_per_column,
                names=[
                    f"out_L1L2_{i}_{lane}" for lane in range(worker_lanes_per_column)
                ],
                depths=[fifodepth] * worker_lanes_per_column,
                placement=Tile(i, 1),
            )
        )
        for lane in range(worker_lanes_per_column):
            of_adds[i][lane] = ObjectFifo(
                worker_tile_ty, name=f"add_{i}_{lane}", depth=fifodepth
            )
            weights_buffer = Buffer(
                type=weights_ty,
                initial_value=static_weights,
                name=f"weights_buffer_{i}_{lane}",
            )
            my_workers.append(
                Worker(
                    core_body_stg1,
                    [
                        of_in1_l2l1s[i][lane].cons(),
                        of_in2_l2l1s[i][lane].cons(),
                        of_adds[i][lane].prod(),
                        eltwise_add_kernel,
                    ],
                )
            )
            my_workers.append(
                Worker(
                    core_body_stg2,
                    [
                        of_adds[i][lane].cons(),
                        weights_buffer,
                        of_out_l1l2s[i][lane].prod(),
                        layer_norm_kernel,
                        eltwise_mul_kernel,
                    ],
                )
            )

    # Create a TensorAccessPattern for each core
    # to describe the data movement
    # The pattern chops the data in equal chunks
    # and moves them in parallel across the cores.
    taps = [
        TensorAccessPattern(
            (1, num_elements),
            chunk * i,
            [1, 1, 1, chunk],
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
                of_outs[i].cons(),
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
    # Transfer size is required to define the size of the data to be transferred
    # It must be a multiple of 1024 and divisible by the number of columns
    p.add_argument("-l", "--length", required=True, dest="length", help="Transfer size")
    # Number of columns is required to define the number of columns to be used
    # It must be less than or equal to 4 for npu and 8 for npu2
    p.add_argument(
        "-co", "--columns", required=True, dest="cols", help="Number of columns"
    )
    # Weight length
    p.add_argument(
        "-wl",
        "--weight-length",
        required=True,
        dest="weight_length",
        help="Weight vector length",
    )
    # Static weights
    p.add_argument(
        "-wf",
        "--weight-file",
        required=True,
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

    length = int(opts.length)
    columns = int(opts.cols)
    dev = opts.device  # Now this is already a device object!

    # Validate columns based on device type
    if isinstance(dev, NPU1) and columns > 4:
        raise ValueError("[ERROR] NPU device cannot allocate more than 4 columns")
    elif isinstance(dev, NPU2) and columns > 8:
        raise ValueError("[ERROR] NPU2 device cannot allocate more than 8 columns")

    weight_length = int(opts.weight_length)
    # For add and norm: cores = columns * 2 (2 cores are used per set of inputs, one for each stage in pipeline)
    if (length % (weight_length * columns)) != 0:
        print(
            "transfer size ("
            + str(length)
            + ") must be a multiple of weight_length * columns ("
            + str(weight_length * columns)
            + ")"
        )
        raise ValueError
    # Load static weights
    weight_file_path = Path(opts.weight_file)
    trace_size = int(opts.trace_size) if opts.trace_size is not None else 0

    module = my_weighted_layer_norm(
        dev,
        length,
        columns,
        weight_length,
        weight_file_path,
        opts.kernel_archive,
        trace_size,
    )

    output_file_path = Path(opts.output_file_path)

    with open(output_file_path, "w") as f:
        f.write(str(module))

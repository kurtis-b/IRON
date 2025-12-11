# SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from ml_dtypes import bfloat16
from pathlib import Path
import numpy as np
import argparse
import sys

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker, LocalBuffer
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1, NPU2, Tile
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron.controlflow import range_
from aie.helpers.util import np_ndarray_type_get_shape


DATA_MEM_SIZE = 65536 # L1 size in bytes

def get_rows_to_process(num_elements, weight_length):
    # Determine per-tile elements based on weight_length
    for i in range(1, num_elements // weight_length):
        per_tile_elements = weight_length * i
        # 1 input + 1 output + weight vector for second stage, bf16, double buffering
        if (per_tile_elements * 2 + weight_length) * 2 * 2 > DATA_MEM_SIZE:
            return i - 1
    return num_elements // weight_length

def my_weighted_layer_norm(
    dev, num_elements, num_columns, num_channels, weight_length, static_weights, trace_size
):
    per_tile_elements = weight_length
    rows_to_process = get_rows_to_process(num_elements, weight_length)
    total_cores = num_columns * num_channels  # For each core that does layer norm, another core will take its output to do eltwise mul
    input_tile_size = per_tile_elements * rows_to_process
    for _ in range(rows_to_process, 0, -1):
        n = input_tile_size * total_cores
        if num_elements % n == 0:
            break
        input_tile_size -= per_tile_elements
        rows_to_process -= 1
    if input_tile_size == 0:
        raise ValueError(
            f"Couldn't find tile size multiple for number of elements ({num_elements})"
        )
    N_div_n = num_elements // n
    chunk = num_elements // total_cores
    dtype = bfloat16
    # Define tensor types
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    weights_ty = np.ndarray[(per_tile_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(input_tile_size,), np.dtype[dtype]]

    # Set fifodepth based on weight_length
    fifodepth = 1 if weight_length > 4096 else 2

    # AIE-array data movement with object fifos
    of_in1s = [
        ObjectFifo(tile_ty, name=f"in1_{i}", depth=fifodepth)
        for i in range(total_cores)
    ]
    of_out1s = [
        ObjectFifo(tile_ty, name=f"out1_{i}", depth=fifodepth)
        for i in range(total_cores)
    ]
    of_out2s = [
        ObjectFifo(tile_ty, name=f"out2_{i}", depth=fifodepth)
        for i in range(total_cores)
    ]

    # AIE Core Function declaration
    layer_norm_kernel = Kernel(
        "layer_norm", "layer_norm_archive.a", [tile_ty, tile_ty, np.int32, np.int32]
    )
    eltwise_mul_kernel = Kernel(
        "eltwise_mul_bf16_vector",
        "layer_norm_archive.a",
        [tile_ty, weights_ty, tile_ty, np.int32, np.int32],
    )

    # Define a task that will run on a compute tile
    def core_body_norm(of_in1, of_out1, layer_norm):
        # Number of sub-vector "tile" iterations
        for _ in range_(N_div_n):
            elem_in1 = of_in1.acquire(1)
            elem_out = of_out1.acquire(1)
            layer_norm(elem_in1, elem_out, per_tile_elements, rows_to_process)
            of_in1.release(1)
            of_out1.release(1)

    def core_body_mul(of_in1, of_out2, eltwise_mul):
        weight_buffer = LocalBuffer(
            type=weights_ty,
            initial_value=static_weights,
        )
        # Number of sub-vector "tile" iterations
        for _ in range_(N_div_n):
            elem_in1 = of_in1.acquire(1)
            elem_out = of_out2.acquire(1)
            eltwise_mul(elem_in1, weight_buffer, elem_out, per_tile_elements, rows_to_process)
            of_in1.release(1)
            of_out2.release(1)

    # Create workers to run the task on compute tiles,
    # one core for layer norm and another pipelined to do eltwise mul
    my_workers = []
    for i in range(total_cores):
        my_workers.append(
            Worker(
                core_body_norm,
                [
                    of_in1s[i].cons(),
                    of_out1s[i].prod(),
                    layer_norm_kernel,
                ],
            )
        )
    for i in range(total_cores):
        my_workers.append(
            Worker(
                core_body_mul,
                [
                    of_out1s[i].cons(),
                    of_out2s[i].prod(),
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
        for i in range(total_cores)
    ]

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (A, C):
        rt.start(*my_workers)

        # Initialize a group for parallel drain tasks, with fill resources free'd when drains complete.
        tg = rt.task_group()

        # Fill the input objectFIFOs with data
        for i in range(total_cores):
            rt.fill(
                of_in1s[i].prod(),
                A,
                taps[i],
                task_group=tg,
                placement=Tile(i % 8, 0), # Place in 8 columns, interleaving rows across channels
            )
        # Drain the output objectFIFOs with data
        for i in range(total_cores):
            rt.drain(
                of_out2s[i].cons(),
                C,
                taps[i],
                wait=True,
                task_group=tg,
                placement=Tile(i % 8, 0), # Place in 8 columns, interleaving rows across channels
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
        help="Static weight values",
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

    channels = int(opts.chans)
    if channels < 1 or channels > 2:
        raise ValueError("Number of channels must be 1 or 2")
    weight_length = int(opts.weight_length)
    # For weighted LAYER norm: cores = columns * channels (weights are broadcasted)
    total_cores = columns * channels
    if (length % (weight_length * total_cores)) != 0:
        print(
            "transfer size ("
            + str(length)
            + ") must be a multiple of weight_length * total_cores ("
            + str(weight_length * total_cores)
            + ")"
        )
        raise ValueError
    # Load static weights
    weight_file_path = Path(opts.weight_file)
    static_weights = np.load(weight_file_path)
    print("weight_file_path: ", weight_file_path)
    print("static_weights: ", static_weights)
    if static_weights.shape[0] != weight_length:
        raise ValueError("Static weights length does not match the specified weight length")
    trace_size = int(opts.trace_size) if opts.trace_size is not None else 0

    module = my_weighted_layer_norm(
        dev, length, columns, channels, weight_length, static_weights, trace_size
    )

    output_file_path = Path(opts.output_file_path)

    with open(output_file_path, "w") as f:
        f.write(str(module))
